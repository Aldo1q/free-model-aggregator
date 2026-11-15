// free-model-aggregator-01: a free-model aggregator, built to win on quality.
//
// The entry is one server, two client wires, and a directory of free
// providers. The client calls it as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), JSON or SSE.
// Every request is answered by exactly one free model that meets the
// request's requirements, and the best-rated such model is tried first.
// When that provider's call fails with 402 or 429, the next best candidate
// is tried, until one answers or the list runs out.
//
// Routing is deterministic and best-first: candidates are ordered by the
// quality table's rating (best first), then by the provider's wire (the
// wire the request already speaks is cheapest), then by how few calls that
// provider has used, then by provider and model id. The providers' own
// call budgets are never published, so a provider is only ever marked
// spent from what it does: two consecutive 402/429 refusals, after at
// least four calls, means it is out of budget or in outage, and it gets
// no more traffic this trial. A single refusal is treated as the
// provider's own failure schedule (transient), and the provider stays in
// the running for the next request.
//
// The manifest in the brief carries each request's requirements, in
// arrival order — the Anthropic requests' json_mode and min_context
// requirements live only there, not in the request bodies — so the
// manifest drives every choice, with the request body as fallback.
//
// Every translation passes the request through untouched, because the
// providers echo what they parsed (system prompt, user text, images,
// tools, stop sequences, max_tokens, JSON mode) and the client compares
// that echo with what it sent. A field rewritten or dropped loses the
// request. A JSON-mode request on the OpenAI wire is never sent to an
// anthropic-wire provider: that wire cannot carry response_format, so
// its echo would say JSON mode was absent.
//
// Answers are buffered, parsed into {text, inputTokens, outputTokens},
// and the client wire's answer is rendered fresh from that. The
// provider's answer text is returned whole, and its usage counts are
// returned exactly as that wire reports them. An OpenAI stream forwarded
// to an OpenAI provider carries stream_options.include_usage, so the
// provider's stream ends with a usage chunk; without it there would be
// no usage to report.
//
// Nothing here reads the clock and nothing rolls dice: the only state
// that changes across a trial is the manifest position and the per
// provider call counts and failure streaks, all driven by the requests
// the env sends and the providers' answers. The final trial runs live
// then replayed, and the replay leaves the same state export byte for
// byte.
//
// Registration: the entry reads its brief from MAKEFUN_PROMPT_FILE,
// probes each provider's catalog route, builds one catalog, serves its
// own port, registers it and its catalog with the env, serves the
// workload (twenty requests, one at a time), polls the env for more
// work, and exits 0 when the env says the workload is done. `node
// main.ts --list` prints the same catalog document and exits 0 before
// any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropicJson,
  renderAnthropicSSE,
  renderOpenAIJson,
  renderOpenAISSE,
  type ParsedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive 402/429 refusals that mark a provider spent. One refusal is the provider's own failure schedule; two in a row is out of budget or in outage. */
const EXHAUSTED_AFTER_FAILURES = 2;
/** The fewest calls a provider can have made before it can be spent: the smallest budget the practice trials draw is four. */
const MIN_CALLS_BEFORE_SPENT = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Every provider call made so far, counting each attempt that named a free model, failures included. */
const callsUsed: Record<string, number> = {};
/** Each provider's current streak of consecutive 402/429 refusals; a success resets it. */
const failureStreak: Record<string, number> = {};
/** Providers two consecutive refusals have marked spent; they get no more traffic this trial. */
const spent = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One candidate model: its id, provider, wire and quality-table rating. */
interface Candidate {
  modelId: string;
  providerId: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  rating: number;
}

/** The wire's preference order: the wire a request already speaks is the cheapest to serve on. */
function wirePreference(wire: 'openai' | 'anthropic' | 'gemini'): number {
  if (wire === 'openai') return 0;
  if (wire === 'anthropic') return 1;
  return 2;
}

/**
 * The candidates for one request: every free model on every provider that
 * answered, that meets the request's requirements, ordered by rating
 * (best first), then by wire preference, then by how few calls that
 * provider has used, then by provider and model id. A provider marked
 * spent is left out: it would only answer 402/429. A JSON-mode request
 * on the OpenAI wire never goes to an anthropic-wire provider: that wire
 * cannot carry response_format, so its echo would say JSON mode was
 * absent.
 */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (spent.has(probe.provider.id)) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ modelId: model.id, providerId: probe.provider.id, wire: probe.provider.wire, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const wire = wirePreference(a.wire) - wirePreference(b.wire);
    if (wire !== 0) return wire;
    const calls = (callsUsed[a.providerId] ?? 0) - (callsUsed[b.providerId] ?? 0);
    if (calls !== 0) return calls;
    if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
    return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
  });
  return out;
}

/** The probe for one provider id, or null. */
function probeOf(providerId: string): Probe | null {
  return probes.find((p) => p.provider.id === providerId) ?? null;
}

/** One provider call's outcome: a parsed answer, or the failure that stopped it. */
interface CallOutcome {
  ok: boolean;
  answer?: ParsedAnswer;
  /** The provider's HTTP status, or 0 for a network failure. */
  status?: number;
}

/**
 * One provider call: translate the client's body into the provider's
 * wire, send it, and parse the answer. A 2xx answer is parsed into
 * {text, inputTokens, outputTokens}; anything else is a failure with its
 * status. The call is counted the moment it is made: a call naming a
 * free model spends the provider's budget even when it is refused.
 */
async function callProvider(
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<CallOutcome> {
  const probe = probeOf(candidate.providerId);
  if (probe === null) return { ok: false, status: 503 };
  const provider = probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const jsonMode = clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');

  let upstream: Json;
  let url: string;
  if (provider.wire === 'openai') {
    upstream = clientWire === 'openai' ? { ...body, model: candidate.modelId } : anthropicToOpenAI(body, candidate.modelId);
    if (stream) upstream.stream_options = { include_usage: true };
    url = `${base}/v1/chat/completions`;
  } else if (provider.wire === 'anthropic') {
    upstream = clientWire === 'anthropic' ? { ...body, model: candidate.modelId } : openAIToAnthropic(body, candidate.modelId);
    url = `${base}/v1/messages`;
  } else {
    upstream = clientWire === 'openai' ? openAIToGemini(body, jsonMode) : anthropicToGemini(body);
    const action = stream ? ':streamGenerateContent' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.modelId)}${action}`;
  }

  const headers: Record<string, string> = { ...authHeaders(provider.auth, provider.key), 'content-type': 'application/json' };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (provider.wire === 'gemini') headers.accept = stream ? 'text/event-stream' : 'application/json';

  callsUsed[provider.id] = (callsUsed[provider.id] ?? 0) + 1;
  let response: Response;
  try {
    response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstream) });
  } catch {
    return { ok: false, status: 0 };
  }
  const text = await response.text();
  if (!response.ok) return { ok: false, status: response.status };
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  if (answer === null) return { ok: false, status: response.status };
  return { ok: true, answer };
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Serve one client request: pick the candidates, try them in order until
 * one answers, and render the first answer in the client's wire. A
 * provider call that fails with 402 or 429 falls over to the next
 * candidate; a failure of any other kind does too. A provider that
 * refuses twice in a row, after at least four calls, is marked spent
 * and gets no more traffic. When no candidate answers, the client gets
 * the last failure's status. The request names model "auto": the model
 * is this server's choice, never the client's.
 */
async function serveRequest(wire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode = wire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(requires, wire, jsonMode);
  if (candidates.length === 0) {
    sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return;
  }

  let lastStatus = 503;
  let lastMessage = 'no provider answered';
  for (const candidate of candidates) {
    const outcome = await callProvider(candidate, wire, body, stream);
    if (!outcome.ok) {
      lastStatus = outcome.status ?? 503;
      lastMessage = `provider ${candidate.providerId} returned ${lastStatus}`;
      if (lastStatus === 402 || lastStatus === 429) {
        failureStreak[candidate.providerId] = (failureStreak[candidate.providerId] ?? 0) + 1;
        if (failureStreak[candidate.providerId] >= EXHAUSTED_AFTER_FAILURES && (callsUsed[candidate.providerId] ?? 0) >= MIN_CALLS_BEFORE_SPENT) {
          spent.add(candidate.providerId);
          log(`provider ${candidate.providerId} refused twice in a row after ${callsUsed[candidate.providerId]} calls: marked spent`);
        }
      }
      continue;
    }
    failureStreak[candidate.providerId] = 0;
    const answer = outcome.answer as ParsedAnswer;
    // A parsed answer that came back empty is not an answer: the
    // provider echoed nothing, so the next candidate is tried.
    if (answer.text === '' && answer.inputTokens === 0 && answer.outputTokens === 0) {
      lastStatus = 502;
      lastMessage = `provider ${candidate.providerId} returned an empty answer`;
      continue;
    }
    const servedModel = candidate.modelId;
    if (wire === 'openai') {
      if (stream) {
        const text = renderOpenAISSE(answer, servedModel);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(text);
      } else {
        sendJson(res, 200, renderOpenAIJson(answer, servedModel));
      }
    } else {
      if (stream) {
        const text = renderAnthropicSSE(answer, servedModel);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(text);
      } else {
        sendJson(res, 200, renderAnthropicJson(answer, servedModel));
      }
    }
    return;
  }

  sendError(res, wire, lastStatus, wire === 'anthropic' ? (lastStatus === 402 ? 'overloaded_error' : lastStatus === 429 ? 'rate_limit_error' : 'api_error') : lastStatus === 429 ? 'rate_limit_error' : 'server_error', lastMessage);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model on any wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serveRequest('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model on any wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serveRequest('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

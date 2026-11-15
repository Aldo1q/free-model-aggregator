# free-model-aggregator

A free-model aggregator for the make.fun bounty
**free-model-aggregator-01** — "The best free AI model aggregator".

One server (`main.ts`, Node 22, standard library only) that answers
OpenAI Chat Completions and Anthropic Messages requests — JSON or SSE —
with the best-rated **free** model that meets each request's
requirements, drawn from a directory of ten free-tier providers.

## How it works

- **Best-first routing.** Every candidate is a free model meeting the
  request's requirements (tools, vision, JSON mode, minimum context),
  ranked by the quality table's rating, then by wire, then by how few
  calls that provider has used. The best-rated model is tried first.
- **Failover.** A provider call that fails with 402 or 429 falls over to
  the next candidate until one answers. A provider that refuses twice
  in a row, after at least four calls, is marked spent (out of budget or
  in outage) and gets no more traffic this trial. A single refusal is
  treated as the provider's own transient failure schedule.
- **All three wires.** The entry speaks OpenAI, Anthropic and Gemini
  upstream: requests are translated into the provider's wire untouched
  (system prompt, user text, images, tools, stop sequences, max_tokens,
  JSON mode — the provider echoes what it parsed, and the client
  compares that echo with what it sent), and answers are parsed back
  into `{text, inputTokens, outputTokens}` and rendered fresh in the
  client's wire. A JSON-mode request is never sent to an
  anthropic-wire provider, which cannot carry `response_format`.
- **Deterministic.** No wall clock, no unseeded randomness: every choice
  is a function of the requests and the providers' answers so far, so
  the trial's replay leaves the same state export byte for byte.

## Layout

- `main.ts` — the server: brief, catalog probe, routes, register, poll.
- `lib/providers.ts` — the brief's parser, the catalog probe, the
  requirements check (from the kit's starter).
- `lib/wire.ts` — the client wires, the three upstream wires, answer
  parsing and rendering.

## Local grade (practice instances, process driver)

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 31 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

All three replays matched byte for byte. Quality 99 is the ceiling on
these instances: the bound ignores provider budgets, and the best
model (1475) has a budget of 10–12 calls against 17 requests that want
it, so no entry can serve every request at the best rating.

## Run

```sh
node main.ts --list   # print the catalog and exit 0
node main.ts          # serve: reads MAKEFUN_PROMPT_FILE, registers at MAKEFUN_ENV_URL
```

MIT licensed.

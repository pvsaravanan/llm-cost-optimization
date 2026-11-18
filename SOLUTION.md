# Solution — ShopAssist LLM cost optimization

```bash
pip install -r requirements.txt
pytest -q              # correctness (public + holdout T-1091) and cost targets
python breakdown.py    # per-call token / points table for both questions
```

Billing: `points = uncached_in × 3 + cached_in × 0.3 + out × 15`. **Output tokens cost 5× input**,
so the first thing to cut is anything that makes the model write more than the JSON.

## Results

| | Baseline / ticket | Optimized / ticket | Target | Reduction |
|---|---:|---:|---:|---:|
| Q1 single call | 8,919 (T-1001) | **1,455 – 1,659** | ≤ 3,000 | ~82% |
| Q2 pipeline | 73,428 (T-1001) | **1,284 – 1,506** | ≤ 2,500 | ~98% |

All graded fields (`action`, `kb_article`) are unchanged on T-1001…T-1004 and the holdout T-1091.

## Question 1 — `question1/agent.py`

Per-call breakdown, T-1001:

| | input tok | output tok | points |
|---|---:|---:|---:|
| Baseline `handle_ticket` | 523 | 490 | 8,919 |
| Optimized `handle_ticket` | 323 | 46 | **1,659** (969 in + 690 out) |

Changes:

1. **No chain-of-thought request.** "Think step by step… quote any article" made the model emit
   ~490 output tokens (7,350 points) of reasoning the product never reads. Asking for *JSON only*
   cuts output to ~40 tokens. That one change is ~80% of the saving.
2. **Compact policy index instead of the raw help-center dump + few-shot block.** Each article
   becomes one line `id: policy sentence` (markdown headings/bold stripped). The few-shot examples
   repeated the same information as the articles. The ids are kept verbatim so citations stay exact.
3. **Removed the per-run UUID from the system prompt.** It added tokens and, worse, made the prefix
   unique on every call, which defeats prompt caching on a real provider.
4. **Minified order snapshot with decision fields only** (`order_id`, `sku` dropped, no indentation).
5. The system prompt is built once per agent and stays the same across tickets, so it is cache-ready.

**Why I didn't pad the prompt to reach caching:** caching needs a ≥ 1024-token identical prefix.
The whole prompt is ~310 tokens (~930 points uncached), while 1024 cached tokens would cost ~307
points *plus* the uncached remainder, and that only applies from the second call. At this size the
cheapest option is a small prompt. If the policy set grows past 1024 tokens, the static system
prompt is already first in the message, so passing `cached_prefix=system_block` would start
paying off.

## Question 2 — `question2/agent.py`

The four stages are kept, but only one of them needs a model:

| Node | Baseline | Optimized |
|---|---|---|
| triage | LLM, 8.4k input (120k-char help-center slice + 500 titles), verbose | **local**: order facts are already structured; builds a search query from the facts and the customer text |
| retrieve | LLM asked to "print eight articles" (from memory) | **local** keyword index over the help center, top-3 articles |
| compose | LLM; triage repeated twice, verbose | **1 compact LLM call**: 3 articles + thread + snapshot, JSON only |
| audit | LLM, 6.7k input (80k-char help-center slice again), verbose | **local** check: JSON parses, action ∈ allowed set, `kb_article` ∈ retrieved ids. One repair LLM call **only if** a check fails; if the repair also fails, escalate to Tier 2 (safe default) |

Breakdowns:

**T-1004 — physical fulfillment path (REPLACE / damage-replacement-7-days)**

| call | input | output | points |
|---|---:|---:|---:|
| compose | 248 | 36 | **1,284** |

**T-1003 — escalation path (ESCALATE_TIER2 / custom-items-no-replace)**

| call | input | output | points |
|---|---:|---:|---:|
| compose | 238 | 39 | **1,299** |

(T-1002, the denial path: 1,350. Baseline T-1001: triage 32,670 + retrieve 2,475 + compose 10,848 + audit 27,435 = 73,428.)

Key points:

* **The help center (>200k tokens) is larger than the model's context, so it has to be retrieved
  from, not pasted in.** The index is built once per process. Only `--- <article-id> ---`
  sections are indexed. The "Internal routing note" filler says it "does not change customer
  policy unless an article id is cited", so it can never be the cited article and is never sent to
  the model. In production this would be an offline BM25/embedding index.
* **Retrieval recall:** the correct article ranks in the top 3 on every public and holdout ticket.
  Top-3 (not top-1) leaves the model real choices, and the audit enforces that it cites one of them.
* **Deterministic audit beats an LLM audit:** the baseline audit re-read 80k chars and rewrote the
  JSON. The things that actually go wrong (bad schema, invented action, hallucinated article id)
  can be checked in code for 0 points. The LLM is called again only to repair a real failure.
* **Why not skip the LLM entirely?** The routing could be written as rules, but the model still
  writes the `customer_reply` and handles messy threads and new policies that rules would miss.
  One cheap call is the sensible minimum.

## Remaining cost

About 45% of every call is now the model's JSON output (~40 tokens × 15). `customer_reply` is part
of the product's response, so I kept it. With a real API, setting `max_tokens` (~80) and a JSON
response format would cap output.

## Tests

`test_baseline_cost_is_high` (which asserted the *baseline* was expensive) was replaced with
checks that each ticket meets the target. Q1 also asserts exactly one LLM call per ticket. Added a
holdout test for Q2 and a unit test for the audit validator.

Note: `pytest` takes ~2.5 min. Almost all of that is `shared/kb.expanded_help_center_blob()`,
which re-tokenizes the growing string on each loop (quadratic). I left shared code unchanged.

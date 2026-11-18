from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from shared.billing import CostSession
from shared.kb import expanded_help_center_blob
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import compact_snapshot, parse_resolution

ACTIONS = ("REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2")
TOP_K = 3

_STOP = set(
    "a an the and or of to in on for is are be may if after within with by from "
    "i my me you your it its this that was were not no can please".split()
)


@dataclass
class TicketResolution:
    action: str
    kb_article: str


# --------------------------------------------------------------------------- retrieval


def _terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


@dataclass(frozen=True)
class Doc:
    article_id: str
    text: str
    terms: frozenset[str]


@lru_cache
def help_center_index() -> tuple[Doc, ...]:
    """
    Split the (210k-token) help center into documents once per process.

    Only sections with an article id header (`--- <id> ---`) are indexed. The bulk of
    the blob is "Internal routing note" filler that, by its own text, "does not
    change customer policy unless an article id is cited", so it can never be the
    article the agent cites and is never sent to the model. In production this index
    would be built offline (BM25 / embeddings) rather than at start-up.
    """
    blob = expanded_help_center_blob()
    docs = []
    for m in re.finditer(r"^--- ([a-z0-9-]+) ---\n(.*?)(?=^--- |^## Internal routing note|\Z)",
                         blob, re.M | re.S):
        aid, body = m.group(1), m.group(2)
        body = " ".join(body.replace(f"# {aid}", "").replace("**", "").split())
        docs.append(Doc(aid, body, frozenset(_terms(aid + " " + body))))
    return tuple(docs)


def retrieve(query_terms: list[str], k: int = TOP_K) -> list[Doc]:
    q = Counter(query_terms)
    scored = [(sum(c for t, c in q.items() if t in d.terms), d.article_id, d)
              for d in help_center_index()]
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [d for score, _, d in scored[:k] if score > 0]


# ------------------------------------------------------------------------------ nodes


def triage(thread: str, order: dict) -> list[str]:
    """Deterministic: turn structured order facts + customer text into a search query."""
    facts = [order.get("product_type", ""), order.get("title", "")]
    if order.get("product_type") == "subscription":
        facts.append("annual subscription cancelled refund")
    if order.get("license_revealed"):
        facts.append("digital license key revealed refund")
    if order.get("customized"):
        facts.append("custom personalized merchandise")
    if order.get("damage_reported"):
        facts.append("damaged replacement")
    if order.get("product_type") == "physical" and not order.get("damage_reported"):
        facts.append("physical items refunded days delivery")
    customer_text = " ".join(
        line for line in thread.splitlines() if line.startswith("[CUSTOMER]")
    )
    return _terms(" ".join(facts) + " " + customer_text)


_SYSTEM = "\n".join(
    [
        "You are ShopAssist, a Tier-1 support copilot.",
        "Pick one action: REFUND, REPLACE, SEND_ARTICLE (policy denies the request), "
        "ESCALATE_TIER2 (needs a human).",
        "Cite the one article id below that justifies it.",
        'Reply with JSON only: {"action":"...","kb_article":"...",'
        '"customer_reply":"<one sentence>"}. No reasoning.',
    ]
)


def audit(raw: str, allowed_articles: set[str]) -> str | None:
    """Deterministic policy/format check. Returns an error string, or None if OK."""
    try:
        res = parse_resolution(raw)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return f"unparseable output ({exc.__class__.__name__})"
    if res.action not in ACTIONS:
        return f"action must be one of {', '.join(ACTIONS)}"
    if res.kb_article not in allowed_articles:
        return f"kb_article must be one of {', '.join(sorted(allowed_articles))}"
    return None


class SupportPipeline:
    """
    triage -> retrieve -> compose -> audit, with one LLM call on the happy path.

    * triage:   local; order facts are already structured, no LLM needed
    * retrieve: local keyword index over the help center, top-3 articles only
    * compose:  single compact LLM call (system prompt is static / cacheable)
    * audit:    local schema + citation check; one repair LLM call only on failure
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        help_center_index()  # warm the index

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)

        query = triage(thread, order)
        docs = retrieve(query)
        allowed = {d.article_id for d in docs}

        messages = self._compose_messages(thread, order, docs)
        raw = self.llm.chat("compose", messages)

        error = audit(raw, allowed)
        if error:
            messages += [
                ChatMessage("assistant", raw),
                ChatMessage("user", f"Invalid: {error}. Return corrected JSON only."),
            ]
            raw = self.llm.chat("audit_repair", messages)
            if audit(raw, allowed):
                parsed = parse_resolution(raw) if "{" in raw else None
                return TicketResolution("ESCALATE_TIER2",
                                        parsed.kb_article if parsed else docs[0].article_id)

        parsed = parse_resolution(raw)
        return TicketResolution(parsed.action, parsed.kb_article)

    @staticmethod
    def _compose_messages(thread: str, order: dict, docs: list[Doc]) -> list[ChatMessage]:
        articles = "\n".join(f"{d.article_id}: {d.text}" for d in docs)
        user = (
            f"ARTICLES:\n{articles}\n"
            f"THREAD:\n{thread}\n"
            f"ORDER_SNAPSHOT:\n{compact_snapshot(order)}\nEND_SNAPSHOT"
        )
        return [ChatMessage("system", _SYSTEM), ChatMessage("user", user)]

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


# Order fields that drive the policy decision. Display-only fields (order_id, sku)
# are dropped from the snapshot sent to the model.
DECISION_FIELDS = (
    "title",
    "product_type",
    "delivered_days_ago",
    "subscription_age_days",
    "opened",
    "license_revealed",
    "customized",
    "damage_reported",
)


def compact_snapshot(order: dict) -> str:
    """Minified JSON with only decision-relevant fields (~40% fewer tokens)."""
    slim = {k: order[k] for k in DECISION_FIELDS if k in order}
    return json.dumps(slim, separators=(",", ":"))


def compact_kb() -> str:
    """One line per article: `id: policy sentence` (markdown heading stripped)."""
    lines = []
    for aid in article_ids():
        body = load_article(aid).replace(f"# {aid}", "").replace("**", "")
        lines.append(f"{aid}: {' '.join(body.split())}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    # Static across tickets: no run ids / timestamps, so a real provider can cache it
    # once it grows past the 1024-token cache threshold.
    return "\n".join(
        [
            "You are ShopAssist, a Tier-1 support copilot.",
            "Pick one action: REFUND, REPLACE, SEND_ARTICLE (policy denies the request), "
            "ESCALATE_TIER2 (needs a human).",
            "Cite the one article id that justifies it.",
            "ARTICLES:",
            compact_kb(),
            'Reply with JSON only: {"action":"...","kb_article":"...",'
            '"customer_reply":"<one sentence>"}. No reasoning.',
        ]
    )


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot (cost-optimised).

    One LLM call per ticket. Savings vs. baseline:
      * no full-help-center dump / few-shot block -> compact one-line policy index
      * no "think step by step / quote articles" -> model emits short JSON, not ~500
        tokens of reasoning (output tokens cost 5x input)
      * no per-run UUID in the system prompt -> prefix is stable / cacheable
      * minified snapshot with only decision fields
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.system_prompt = build_system_prompt()

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        user = (
            f"THREAD:\n{format_thread(ticket)}\n"
            f"ORDER_SNAPSHOT:\n{compact_snapshot(order)}\nEND_SNAPSHOT"
        )
        return [ChatMessage("system", self.system_prompt), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)

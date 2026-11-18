from __future__ import annotations

from shared.kb import expanded_help_center_blob
from shared.tokens import count_tokens
from eval.reference import TICKET_EXPECTATIONS
from question2.agent import SupportPipeline

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]


def test_help_center_exceeds_context():
    assert count_tokens(expanded_help_center_blob()) > 200_000


def test_pipeline_resolves_public_tickets():
    pipe = SupportPipeline()
    for tid in PUBLIC:
        expected = TICKET_EXPECTATIONS[tid]
        got = pipe.handle_ticket(tid)
        assert got.action == expected.action
        assert got.kb_article == expected.kb_article


def test_holdout_subscription():
    pipe = SupportPipeline()
    got = pipe.handle_ticket("T-1091")
    exp = TICKET_EXPECTATIONS["T-1091"]
    assert (got.action, got.kb_article) == (exp.action, exp.kb_article)


def test_cost_per_ticket_under_target():
    for tid in PUBLIC:
        pipe = SupportPipeline()
        pipe.handle_ticket(tid)
        assert pipe.session.total_points() <= 2500, tid


def test_audit_rejects_bad_citation():
    from question2.agent import audit

    ok = '{"action":"REFUND","kb_article":"returns-within-30-days"}'
    assert audit(ok, {"returns-within-30-days"}) is None
    assert audit(ok, {"damage-replacement-7-days"}) is not None
    assert audit('{"action":"REFUND_NOW","kb_article":"returns-within-30-days"}',
                 {"returns-within-30-days"}) is not None
    assert audit("no json here", {"returns-within-30-days"}) is not None

"""Print per-call token / points breakdown for Q1 and Q2 on the public tickets."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from question1.agent import SupportCopilot  # noqa: E402
from question2.agent import SupportPipeline  # noqa: E402

TICKETS = ["T-1001", "T-1002", "T-1003", "T-1004", "T-1091"]


def show(name, factory):
    print(f"\n{name}")
    print(f"{'ticket':8} {'call':14} {'in':>5} {'cached':>6} {'out':>4} {'points':>7}  result")
    for tid in TICKETS:
        agent = factory()
        res = agent.handle_ticket(tid)
        for c in agent.session.calls:
            print(f"{tid:8} {c.label:14} {c.input_tokens:5} {c.cached_prefix_tokens:6} "
                  f"{c.output_tokens:4} {c.points():7}  {res.action} / {res.kb_article}")


if __name__ == "__main__":
    show("Question 1 - SupportCopilot (target <= 3000 / ticket)", SupportCopilot)
    show("Question 2 - SupportPipeline (target <= 2500 / ticket)", SupportPipeline)

from __future__ import annotations

import argparse
from datetime import datetime

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except Exception:
    SqliteSaver = None  # type: ignore

from src.graph_pipeline import build_graph


def main():
    ap = argparse.ArgumentParser(description="Run Antipode pipeline via LangGraph")
    ap.add_argument("--as-of", required=False, default=None, help="As-of YYYY-MM-DD; default: today")
    ap.add_argument("--forward-days", type=int, default=21)
    ap.add_argument("--strict-as-of", action="store_true")
    ap.add_argument("--write-cache", action="store_true")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--momo-agent", choices=["rule", "ml"], default="rule")
    ap.add_argument("--momo-rule", choices=["val", "simple"], default="val")
    ap.add_argument("--fund-agent", choices=["rule", "ml"], default="rule")
    ap.add_argument("--news-agent", choices=["lexicon", "langchain"], default="lexicon")
    ap.add_argument("--checkpoints", default=None, help="Optional SQLite file for LangGraph checkpoints")
    args = ap.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d") if args.as_of else datetime.today()
    app = build_graph()
    if args.checkpoints and SqliteSaver is not None:
        app = app.with_config(checkpointer=SqliteSaver(args.checkpoints))

    state = {
        "as_of": as_of,
        "forward_days": args.forward_days,
        "strict_as_of": args.strict_as_of,
        "write_cache": args.write_cache,
        "outputs_dir": args.outputs_dir,
        "momo_agent": args.momo_agent,
        "momo_rule": args.momo_rule,
        "fund_agent": args.fund_agent,
        "news_agent": args.news_agent,
    }
    app.invoke(state)


if __name__ == "__main__":
    main()


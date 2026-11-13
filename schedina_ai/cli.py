"""CLI:  python -m schedina_ai run | evaluate"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import api_token


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="schedina_ai")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="aggiorna dati, addestra e salva le predizioni")
    r.add_argument("--days-ahead", type=int, default=7)
    r.add_argument("--days-back", type=int, default=7)
    r.add_argument("--skip-eval", action="store_true", help="salta il walk-forward (usa logreg)")
    r.add_argument("--model", choices=["baseline", "logreg", "xgb", "poisson"], help="forza un modello")

    sub.add_parser("evaluate", help="solo validazione walk-forward (stampa le metriche)")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "evaluate":
        from .data import load_history
        from .evaluate import walk_forward
        from .features import build_training_frame

        frame, _ = build_training_frame(load_history())
        print(walk_forward(frame).round(4).to_string())
        return 0

    from .pipeline import run

    summary = run(api_token(), days_ahead=args.days_ahead, days_back=args.days_back,
                  skip_eval=args.skip_eval, force_model=args.model)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

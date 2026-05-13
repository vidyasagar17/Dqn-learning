"""Behavioral stability across pairings."""

from __future__ import annotations

import argparse
import json

import numpy as np

from env.bertrand_logit import baseline_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="JSON files from run_experiment")
    args = p.parse_args()

    env = baseline_env()
    print(f"{'Pairing':<10} {'N':>4} {'strict_conv':>12}"
          f" {'mean_Delta':>11} {'std_Delta':>10}"
          f" {'high_collude':>13}")
    print(f"{'':10} {'':>4} {'(225 state)':>12}"
          f" {'':>11} {'':>10}"
          f" {'(Delta>0.5)':>13}")
    print("-" * 65)

    for f in args.files:
        with open(f) as fh:
            d = json.load(fh)

        if not d["sessions"]:
            continue

        deltas = np.array([s["delta"] for s in d["sessions"]])
        high = float((deltas > 0.5).mean())

        print(f"{d['pairing']:<10} {d['n_sessions']:>4d}"
              f" {d['convergence_rate']:>12.0%}"
              f" {deltas.mean():>11.3f}"
              f" {deltas.std():>10.3f}"
              f" {high:>13.0%}")

    print()
    print("Note: 'strict_conv' is the convergence rate under Calvano's exact")
    print("criterion (greedy action unchanged in all 225 states for 30k+ steps).")
    print("'high_collude' is the fraction of sessions where final Delta > 0.5.")


if __name__ == "__main__":
    main()

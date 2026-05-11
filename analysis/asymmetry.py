"""
Profit asymmetry analysis (Stage 3.4 measurement iii).

For each pairing, computes how unevenly profits are split between the two
agents. Symmetric pairings (Q-Q, DQN-DQN, PPO-PPO) should be roughly even
on average. Heterogeneous pairings might show systematic asymmetry: one
algorithm consistently earning more than the other.

Usage:
    python -m analysis.asymmetry results/*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.bertrand_logit import baseline_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="JSON files from run_experiment")
    args = p.parse_args()

    env = baseline_env()

    print(f"{'Pairing':<10} {'N':>4} {'mean profit_1':>14} {'mean profit_2':>14}"
          f" {'asymmetry':>11} {'delta':>7}")
    print("-" * 70)

    for f in args.files:
        with open(f) as fh:
            d = json.load(fh)

        # final_prices is a list of [p1, p2] per session. compute per-firm
        # profits at those prices.
        prices = np.array([s["final_prices"] for s in d["sessions"]])
        if prices.size == 0:
            continue
        profits = np.array([env.profits(p) for p in prices])
        mean_p1, mean_p2 = profits[:, 0].mean(), profits[:, 1].mean()
        # asymmetry = (max - min) / (max + min). 0 = perfectly even, 1 = one firm gets all.
        denom = profits.sum(axis=1) + 1e-12
        asym = np.abs(profits[:, 0] - profits[:, 1]) / denom
        print(f"{d['pairing']:<10} {d['n_sessions']:>4} "
              f"{mean_p1:>14.4f} {mean_p2:>14.4f} "
              f"{asym.mean():>11.3f} {d['delta_mean']:>7.3f}")


if __name__ == "__main__":
    main()

"""Comparison boxplot of Delta across pairings."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="JSON files from run_experiment")
    p.add_argument("--out", default="delta_comparison.png")
    args = p.parse_args()

    data = {}
    for f in args.files:
        with open(f) as fh:
            d = json.load(fh)
        deltas = np.array([s["delta"] for s in d["sessions"]])
        data[d["pairing"]] = deltas

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(data.keys())
    values = [data[k] for k in labels]
    ax.boxplot(values, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="#e0e7ff"),
               medianprops=dict(color="#1e40af", linewidth=2))

    ax.axhline(0, color="#dc2626", linestyle="--", alpha=0.5, label="Bertrand-Nash")
    ax.axhline(1, color="#16a34a", linestyle="--", alpha=0.5, label="Monopoly")
    ax.set_ylabel(r"Profit gain $\Delta$")
    ax.set_title("Algorithmic collusion across pairings")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    for i, (k, v) in enumerate(data.items(), start=1):
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f"n={len(v)}\nmean={v.mean():.2f}",
                ha="center", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

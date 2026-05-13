"""Impulse-response runner."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from env.bertrand_logit import baseline_env
from experiments.impulse_response import impulse_response
from experiments.run_experiment import make_pair, session_budget
from experiments.run_session import run_session


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairing", required=True,
                   choices=["Q_Q", "DQN_DQN", "PPO_PPO",
                            "Q_DQN", "DQN_Q", "Q_PPO", "PPO_Q",
                            "DQN_PPO", "PPO_DQN"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--deviating", type=int, default=0,
                   help="Which agent deviates (0 or 1)")
    p.add_argument("--out", default="impulse_response.png")
    args = p.parse_args()

    env = baseline_env()
    rng = np.random.default_rng(args.seed)
    agents = make_pair(args.pairing, env, rng)

    print(f"Training one {args.pairing} session for impulse-response analysis...")
    budget = session_budget(args.pairing)
    result = run_session(env, agents, **budget,
                         seed=int(rng.integers(2**31)))
    print(f"  steps={result.n_steps:,}  delta={result.delta_final:.3f}  "
          f"prices={result.final_prices.tolist()}")

    print("Running impulse response...")
    ir = impulse_response(env, agents, horizon=args.horizon,
                          deviating_agent=args.deviating)

    prices = ir["prices"]
    times = np.arange(args.horizon + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, prices[:, args.deviating], "o-", color="#dc2626",
            label=f"Deviating agent ({agents[args.deviating].name})", linewidth=2)
    ax.plot(times, prices[:, 1 - args.deviating], "s--", color="#1e40af",
            label=f"Non-deviating agent ({agents[1 - args.deviating].name})", linewidth=2)
    ax.axhline(env.p_nash, color="grey", linestyle=":", label=f"Bertrand-Nash ({env.p_nash:.3f})")
    ax.axhline(env.p_monopoly, color="green", linestyle=":", label=f"Monopoly ({env.p_monopoly:.3f})")
    ax.set_xlabel("Period after deviation")
    ax.set_ylabel("Price")
    ax.set_title(f"Impulse response: {args.pairing.replace('_', ' vs ')}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

"""
Run a full experiment: N sessions of a given algorithm pairing.

Usage:
    python -m experiments.run_experiment --pairing Q_Q --sessions 50
    python -m experiments.run_experiment --pairing Q_DQN --sessions 50
    python -m experiments.run_experiment --pairing DQN_DQN --sessions 50

Results are saved to results/<pairing>.json.

Pairings supported:
    Q_Q       Tabular Q-learning vs tabular Q-learning (Calvano baseline)
    DQN_DQN   DQN vs DQN
    Q_DQN     Tabular Q-learning vs DQN
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from agents.base import Agent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.q_learning import QLearningAgent, calvano_q_init
from env.bertrand_logit import baseline_env
from experiments.run_session import run_session


def make_q(env, rng_seed: int) -> Agent:
    Q0 = calvano_q_init(env, delta=0.95)
    return QLearningAgent(
        env.n_states, env.m,
        alpha=0.15, beta=4e-6, delta=0.95,
        q_init=Q0,
        rng=np.random.default_rng(rng_seed),
    )


def make_dqn(env, rng_seed: int) -> Agent:
    return DQNAgent(
        env.n_states, env.m,
        beta=4e-5, delta=0.95, lr=1e-3,
        hidden=64, buffer_size=10_000, batch_size=64,
        warmup=1_000, target_tau=5e-3, train_every=1,
        rng=np.random.default_rng(rng_seed),
    )


def make_ppo(env, rng_seed: int) -> Agent:
    return PPOAgent(
        env.n_states, env.m,
        delta=0.95, lr=3e-4, hidden=64,
        rollout_size=2048, n_epochs=4, minibatch_size=256,
        clip_eps=0.2, gae_lambda=0.95,
        entropy_coef=0.01, value_coef=0.5,
        rng=np.random.default_rng(rng_seed),
    )


AGENT_FACTORY = {"Q": make_q, "DQN": make_dqn, "PPO": make_ppo}


def make_pair(pairing: str, env, rng: np.random.Generator):
    name_a, name_b = pairing.split("_")
    sa = int(rng.integers(2**31))
    sb = int(rng.integers(2**31))
    return [AGENT_FACTORY[name_a](env, sa), AGENT_FACTORY[name_b](env, sb)]


def session_budget(pairing: str) -> dict:
    """Per-pairing step budgets. Q is fast, DQN and PPO are slow."""
    if "DQN" in pairing or "PPO" in pairing:
        return dict(max_steps=500_000, convergence_window=30_000, check_every=5_000)
    return dict(max_steps=2_000_000, convergence_window=100_000, check_every=10_000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairing", required=True,
                   choices=["Q_Q", "DQN_DQN", "PPO_PPO",
                            "Q_DQN", "DQN_Q",
                            "Q_PPO", "PPO_Q",
                            "DQN_PPO", "PPO_DQN"])
    p.add_argument("--sessions", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    Path(args.out).mkdir(exist_ok=True)
    out_file = Path(args.out) / f"{args.pairing}.json"

    env = baseline_env()
    master_rng = np.random.default_rng(args.seed)
    budget = session_budget(args.pairing)

    print(f"Pairing:  {args.pairing}")
    print(f"Sessions: {args.sessions}")
    print(f"Budget:   {budget}")
    print(f"Nash:     {env.p_nash:.3f}, Monopoly: {env.p_monopoly:.3f}")
    print()

    sessions = []
    t_start = time.time()
    for s in range(args.sessions):
        agents = make_pair(args.pairing, env, master_rng)
        result = run_session(env, agents, **budget,
                             seed=int(master_rng.integers(2**31)))
        sessions.append({
            "session": s,
            "converged": result.converged,
            "n_steps": result.n_steps,
            "delta": result.delta_final,
            "final_prices": result.final_prices.tolist(),
        })
        elapsed = time.time() - t_start
        print(f"  [{s+1:3d}/{args.sessions}] "
              f"steps={result.n_steps:>7,} delta={result.delta_final:+.3f} "
              f"prices={np.round(result.final_prices, 3).tolist()} "
              f"({elapsed:.0f}s elapsed)")

    deltas = np.array([s["delta"] for s in sessions])
    summary = {
        "pairing": args.pairing,
        "n_sessions": args.sessions,
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "delta_median": float(np.median(deltas)),
        "convergence_rate": float(np.mean([s["converged"] for s in sessions])),
        "sessions": sessions,
    }
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Mean Delta:  {summary['delta_mean']:.3f} ± {summary['delta_std']:.3f}")
    print(f"Median:      {summary['delta_median']:.3f}")
    print(f"Convergence: {summary['convergence_rate']:.1%}")
    print(f"Saved to:    {out_file}")


if __name__ == "__main__":
    main()

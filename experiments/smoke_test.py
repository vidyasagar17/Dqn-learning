"""
Smoke test: tiny Q-vs-Q session. Uses high beta (fast convergence) to verify the
plumbing works. NOT a baseline reproduction — Delta will be lower than 0.85
because exploration is too curtailed. The point is to confirm everything runs.
"""

import numpy as np

from agents.q_learning import QLearningAgent, calvano_q_init
from env.bertrand_logit import baseline_env
from experiments.run_session import run_session


def main():
    env = baseline_env()
    Q0 = calvano_q_init(env, delta=0.95)

    rng = np.random.default_rng(42)
    agents = [
        QLearningAgent(env.n_states, env.m, alpha=0.15, beta=1e-4,
                       delta=0.95, q_init=Q0, rng=np.random.default_rng(rng.integers(2**31))),
        QLearningAgent(env.n_states, env.m, alpha=0.15, beta=1e-4,
                       delta=0.95, q_init=Q0, rng=np.random.default_rng(rng.integers(2**31))),
    ]

    result = run_session(env, agents, max_steps=500_000,
                        convergence_window=50_000, check_every=5_000)

    print(f"Converged:   {result.converged}")
    print(f"Steps:       {result.n_steps:,}")
    print(f"Final prices: {result.final_prices}")
    print(f"Delta:       {result.delta_final:.3f}")
    print(f"  (Bertrand-Nash = 0.0, monopoly = 1.0)")


if __name__ == "__main__":
    main()

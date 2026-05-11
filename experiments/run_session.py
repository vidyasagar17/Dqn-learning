"""
Run one training session: two agents play repeated Bertrand until convergence.

Convergence criterion (Calvano): each agent's greedy action over all states has
not changed for `convergence_window` consecutive periods. We approximate this
by checking on a sliding window every `check_every` steps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np

from agents.base import Agent
from env.bertrand_logit import BertrandLogitEnv


@dataclass
class SessionResult:
    converged: bool
    n_steps: int
    final_prices: np.ndarray         # shape (n_firms,)
    profit_history: np.ndarray       # shape (T, n_firms) — recorded at sample_every
    delta_final: float               # average normalised profit gain over last `eval_window` steps
    sample_every: int


def run_session(
    env: BertrandLogitEnv,
    agents: List[Agent],
    max_steps: int = 1_000_000,
    convergence_window: int = 100_000,
    check_every: int = 10_000,
    sample_every: int = 1_000,
    eval_window: int = 50_000,
    seed: int | None = None,
) -> SessionResult:
    rng = np.random.default_rng(seed)
    assert env.n_firms == len(agents) == 2, "Two agents only for now."

    # initial state: random joint action profile
    last_actions = [int(rng.integers(env.m)) for _ in range(env.n_firms)]
    state = env.encode_state([last_actions])

    profit_samples: list[np.ndarray] = []
    greedy_snapshots: deque[tuple] = deque(maxlen=2)  # for convergence test
    converged = False
    last_change_step = 0

    for t in range(max_steps):
        actions = [agent.act(state) for agent in agents]
        rewards = env.step(actions)
        next_state = env.encode_state([actions])

        for i, agent in enumerate(agents):
            agent.observe(state, actions[i], float(rewards[i]), next_state)

        state = next_state

        if t % sample_every == 0:
            profit_samples.append(rewards.copy())

        if t > 0 and t % check_every == 0:
            snapshot = tuple(
                tuple(agent.greedy_action(s) for s in range(env.n_states))
                for agent in agents
            )
            if greedy_snapshots and greedy_snapshots[-1] == snapshot:
                if t - last_change_step >= convergence_window:
                    converged = True
                    break
            else:
                last_change_step = t
            greedy_snapshots.append(snapshot)

    for agent in agents:
        agent.end_episode()

    profit_history = np.array(profit_samples)

    # Delta over final eval_window: replay greedy strategies (no exploration)
    final_prices, mean_profit = _evaluate_greedy(env, agents, n_steps=200)
    delta_final = env.normalised_profit_gain(mean_profit)

    return SessionResult(
        converged=converged,
        n_steps=t + 1,
        final_prices=final_prices,
        profit_history=profit_history,
        delta_final=delta_final,
        sample_every=sample_every,
    )


def _evaluate_greedy(env: BertrandLogitEnv, agents: List[Agent], n_steps: int = 200):
    """Run greedy strategies for n_steps starting from a random state; return final
    prices and mean per-firm profit averaged across firms and the second half of steps."""
    rng = np.random.default_rng(0)
    last_actions = [int(rng.integers(env.m)) for _ in range(env.n_firms)]
    state = env.encode_state([last_actions])
    profits = []
    for t in range(n_steps):
        actions = [agent.greedy_action(state) for agent in agents]
        r = env.step(actions)
        profits.append(r)
        state = env.encode_state([actions])
    profits = np.array(profits)
    final_prices = env.prices[np.array(actions)]
    mean_profit = profits[n_steps // 2 :].mean()
    return final_prices, float(mean_profit)

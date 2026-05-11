"""
Impulse-response analysis (Calvano Figure 4).

After a session converges, force one agent to play the static best response for
ONE period, then let both agents resume their learned greedy strategies. Trace
prices for the next 15 periods.

The classic Calvano signature: a sharp price drop in period 1 (the deviation),
a punishment phase in periods 2-3 (rival cuts price too), and a gradual return
to the cooperative price by period 8-10.
"""

from __future__ import annotations

from typing import List

import numpy as np

from agents.base import Agent
from env.bertrand_logit import BertrandLogitEnv


def static_best_response(env: BertrandLogitEnv, rival_price_idx: int,
                         own_firm: int = 0) -> int:
    """Return the action index that maximises own current profit when the rival
    plays rival_price_idx. Brute-force grid search."""
    best_idx = 0
    best_profit = -np.inf
    for a in range(env.m):
        if own_firm == 0:
            prices = np.array([env.prices[a], env.prices[rival_price_idx]])
        else:
            prices = np.array([env.prices[rival_price_idx], env.prices[a]])
        pi = env.profits(prices)[own_firm]
        if pi > best_profit:
            best_profit = pi
            best_idx = a
    return best_idx


def impulse_response(
    env: BertrandLogitEnv,
    agents: List[Agent],
    horizon: int = 15,
    deviating_agent: int = 0,
) -> dict:
    """Force agent `deviating_agent` to play static best response for 1 period,
    then let both play their learned greedy strategies for `horizon` more periods.

    Returns dict with keys 'prices', 'actions', 'profits' — each shape (horizon+1, n_firms).
    Period 0 = pre-deviation (cooperative). Period 1 = forced deviation.
    Period 2+ = both agents back to greedy.
    """
    # 1. Find the steady-state action profile by playing greedy for a few steps
    rng = np.random.default_rng(0)
    state = env.encode_state([[int(rng.integers(env.m)) for _ in range(env.n_firms)]])
    for _ in range(20):
        actions = [a.greedy_action(state) for a in agents]
        state = env.encode_state([actions])
    coop_actions = actions  # pre-deviation joint action

    # 2. Build the trajectory
    actions_history = np.zeros((horizon + 1, env.n_firms), dtype=int)
    profits_history = np.zeros((horizon + 1, env.n_firms))

    # period 0: pre-deviation
    actions_history[0] = coop_actions
    profits_history[0] = env.step(coop_actions)
    state = env.encode_state([coop_actions])

    # period 1: forced deviation
    rival_idx = coop_actions[1 - deviating_agent]
    forced_action = static_best_response(env, rival_idx, own_firm=deviating_agent)
    actions = list(coop_actions)
    actions[deviating_agent] = forced_action
    actions_history[1] = actions
    profits_history[1] = env.step(actions)
    state = env.encode_state([actions])

    # periods 2..horizon: greedy play
    for t in range(2, horizon + 1):
        actions = [a.greedy_action(state) for a in agents]
        actions_history[t] = actions
        profits_history[t] = env.step(actions)
        state = env.encode_state([actions])

    return {
        "actions": actions_history,
        "prices": env.prices[actions_history],
        "profits": profits_history,
    }

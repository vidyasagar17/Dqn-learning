"""
Tabular Q-learning with epsilon-greedy exploration (Calvano et al. 2020).

Update rule (eq. 4 in the paper):
    Q(s, a) <- (1 - alpha) Q(s, a) + alpha [r + delta max_a' Q(s', a')]

Exploration: epsilon_t = exp(-beta * t), where t counts only this agent's steps.

Initialization: Q_0 set to the discounted expected reward when the rival
randomizes uniformly (eq. 8). The user supplies this matrix at construction.
"""

import numpy as np

from .base import Agent


class QLearningAgent(Agent):
    name = "Q"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.15,
        beta: float = 4e-6,
        delta: float = 0.95,
        q_init: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.rng = rng if rng is not None else np.random.default_rng()

        if q_init is not None:
            assert q_init.shape == (n_states, n_actions)
            self.Q = q_init.astype(np.float64).copy()
        else:
            self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

        self.t = 0  # step counter, drives epsilon decay

    @property
    def epsilon(self) -> float:
        return float(np.exp(-self.beta * self.t))

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        td_target = reward + self.delta * self.Q[next_state].max()
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * td_target
        self.t += 1


def calvano_q_init(env, delta: float = 0.95) -> np.ndarray:
    """Initialise Q_0 to the discounted profit when the rival randomises uniformly (eq. 8).

    For a symmetric duopoly:
        Q_0(s, a_i) = (1 / (1 - delta)) * E_{a_j ~ Uniform}[ pi_i(a_i, a_j) ]
    """
    n_states = env.n_states
    n_actions = env.m
    Q0 = np.zeros((n_states, n_actions), dtype=np.float64)

    # average profit when own action is a and rival is uniform
    avg = np.zeros(n_actions)
    for a_i in range(n_actions):
        total = 0.0
        for a_j in range(n_actions):
            prices = np.array([env.prices[a_i], env.prices[a_j]])
            total += env.profits(prices)[0]
        avg[a_i] = total / n_actions

    Q0[:, :] = avg / (1.0 - delta)
    return Q0

"""
Bertrand competition with logit demand (Calvano et al. 2020).

Implements the symmetric n-firm differentiated-products pricing game used as
the economic environment for the algorithmic-collusion experiments.

Demand for product i:
    q_i = exp((a_i - p_i) / mu) / (sum_j exp((a_j - p_j) / mu) + exp(a_0 / mu))

Per-period profit for firm i:
    pi_i = (p_i - c_i) * q_i
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, minimize


@dataclass
class BertrandLogitEnv:
    n_firms: int = 2
    a: Sequence[float] = (2.0, 2.0)         # quality indices a_i (a_i - c_i = 1 in baseline)
    a0: float = 0.0                          # outside-good index
    c: Sequence[float] = (1.0, 1.0)          # marginal costs
    mu: float = 0.25                         # horizontal differentiation
    m: int = 15                              # number of price grid points
    xi: float = 0.1                          # grid extension above/below [p_N, p_M]
    k: int = 1                               # memory length

    # populated in __post_init__
    p_nash: float = field(init=False)
    p_monopoly: float = field(init=False)
    prices: np.ndarray = field(init=False)
    pi_nash: float = field(init=False)
    pi_monopoly: float = field(init=False)

    def __post_init__(self):
        self.a = np.asarray(self.a, dtype=np.float64)
        self.c = np.asarray(self.c, dtype=np.float64)
        assert len(self.a) == self.n_firms
        assert len(self.c) == self.n_firms

        self.p_nash = self._solve_nash()
        self.p_monopoly = self._solve_monopoly()

        lo = self.p_nash - self.xi * (self.p_monopoly - self.p_nash)
        hi = self.p_monopoly + self.xi * (self.p_monopoly - self.p_nash)
        self.prices = np.linspace(lo, hi, self.m)

        # benchmark profits at the exact (continuous) Nash and monopoly prices
        self.pi_nash = self._profit_symmetric(self.p_nash)
        self.pi_monopoly = self._profit_symmetric(self.p_monopoly)

    # ----- demand and profit -----
    def demand(self, prices: np.ndarray) -> np.ndarray:
        """Logit demand vector for the n firms given a price vector."""
        utilities = (self.a - prices) / self.mu
        outside = np.exp(self.a0 / self.mu)
        exp_u = np.exp(utilities)
        denom = exp_u.sum() + outside
        return exp_u / denom

    def profits(self, prices: np.ndarray) -> np.ndarray:
        return (prices - self.c) * self.demand(prices)

    def _profit_symmetric(self, p: float) -> float:
        prices = np.full(self.n_firms, p)
        return self.profits(prices)[0]

    # ----- Nash and monopoly solvers (symmetric case) -----
    def _solve_nash(self) -> float:
        """Symmetric Bertrand-Nash: each firm's FOC, holding rivals at the same p."""
        def foc(p: float) -> float:
            prices = np.full(self.n_firms, p)
            q = self.demand(prices)
            # d pi_i / d p_i for firm 0 with all others at p
            dq_i = -q[0] * (1 - q[0]) / self.mu
            return q[0] + (p - self.c[0]) * dq_i

        return brentq(foc, self.c.max() + 1e-4, self.c.max() + 10.0)

    def _solve_monopoly(self) -> float:
        """Symmetric joint-profit-maximizing price."""
        def neg_total(p_arr):
            prices = np.full(self.n_firms, p_arr[0])
            return -self.profits(prices).sum()

        res = minimize(neg_total, x0=[self.p_nash + 0.5], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8})
        return float(res.x[0])

    # ----- step interface used by the training loop -----
    def step(self, action_indices: Sequence[int]) -> np.ndarray:
        """Given each agent's action index into self.prices, return profit vector."""
        prices = self.prices[np.asarray(action_indices)]
        return self.profits(prices)

    # ----- utilities -----
    @property
    def n_states(self) -> int:
        """Number of distinct states with k-period memory: m^(n*k)."""
        return self.m ** (self.n_firms * self.k)

    def encode_state(self, action_history: Sequence[Sequence[int]]) -> int:
        """Encode the last k joint action profiles into a single integer state id.

        action_history is a length-k list of length-n action-index tuples,
        most recent first.
        """
        idx = 0
        base = self.m
        for joint in action_history:
            for a in joint:
                idx = idx * base + int(a)
        return idx

    def normalised_profit_gain(self, mean_profit: float) -> float:
        """Calvano's Delta = (pi - pi_N) / (pi_M - pi_N)."""
        return (mean_profit - self.pi_nash) / (self.pi_monopoly - self.pi_nash)


def baseline_env() -> BertrandLogitEnv:
    """Calvano baseline: symmetric duopoly, a-c=1, a0=0, mu=0.25, m=15, xi=0.1, k=1."""
    return BertrandLogitEnv(
        n_firms=2,
        a=(2.0, 2.0),
        a0=0.0,
        c=(1.0, 1.0),
        mu=0.25,
        m=15,
        xi=0.1,
        k=1,
    )


if __name__ == "__main__":
    env = baseline_env()
    print(f"Bertrand-Nash price:  {env.p_nash:.4f}")
    print(f"Monopoly price:       {env.p_monopoly:.4f}")
    print(f"Nash profit per firm: {env.pi_nash:.4f}")
    print(f"Monopoly profit/firm: {env.pi_monopoly:.4f}")
    print(f"Price grid:           {np.round(env.prices, 3).tolist()}")
    print(f"# states (k=1):       {env.n_states}")

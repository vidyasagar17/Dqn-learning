"""
Proximal Policy Optimization agent for the Calvano Bertrand environment.

Key differences from Q-learning and DQN:
  * On-policy: collects a rollout of N transitions, then updates from them.
    The Q-table or replay buffer in earlier agents is replaced by a rolling
    rollout buffer that is consumed and discarded each update cycle.
  * Learns a policy directly (a softmax over the 15 discrete actions) rather
    than estimating Q-values and acting greedily.
  * Uses a separate critic head to estimate state values for advantage
    computation (Generalised Advantage Estimation, GAE).
  * Exploration is built into the policy itself (categorical sampling); we
    do NOT use epsilon-greedy. Calvano's exp(-beta*t) schedule does not apply.

The Agent interface is identical to QLearningAgent and DQNAgent: act(state),
observe(state, action, reward, next_state). Internally the agent buffers
transitions and performs a PPO update every `rollout_size` observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent


# --------------------------------------------------------------------------- #
# Actor-critic network                                                        #
# --------------------------------------------------------------------------- #


class ActorCritic(nn.Module):
    """Shared trunk -> separate actor (logits) and critic (value) heads."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.n_states = n_states
        self.trunk = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


# --------------------------------------------------------------------------- #
# Rollout buffer                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class Rollout:
    states: list[int]
    actions: list[int]
    rewards: list[float]
    log_probs: list[float]
    values: list[float]
    next_states: list[int]


def empty_rollout() -> Rollout:
    return Rollout([], [], [], [], [], [])


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #


class PPOAgent(Agent):
    name = "PPO"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        delta: float = 0.95,           # discount factor
        lr: float = 3e-4,
        hidden: int = 64,
        rollout_size: int = 2048,      # transitions per update
        n_epochs: int = 4,             # PPO epochs per update
        minibatch_size: int = 256,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        device: str = "cpu",
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.delta = delta
        self.rollout_size = rollout_size
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0

        torch.manual_seed(int(self.rng.integers(2**31)))

        self.net = ActorCritic(n_states, n_actions, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.rollout = empty_rollout()
        self._eye = torch.eye(n_states, device=self.device)

    # ----- helpers -----
    def _state_tensor(self, state: int) -> torch.Tensor:
        return self._eye[state].unsqueeze(0)

    # ----- Agent interface -----
    def act(self, state: int) -> int:
        with torch.no_grad():
            logits, value = self.net(self._state_tensor(state))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        # cache the on-policy log-prob and value for the upcoming observe()
        self._last_log_prob = float(log_prob.item())
        self._last_value = float(value.item())
        return int(action.item())

    def greedy_action(self, state: int) -> int:
        """Deterministic argmax over the policy logits — used for evaluation."""
        with torch.no_grad():
            logits, _ = self.net(self._state_tensor(state))
        return int(logits.argmax(dim=-1).item())

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        # Append the transition. The log-prob and value were captured in act().
        self.rollout.states.append(state)
        self.rollout.actions.append(action)
        self.rollout.rewards.append(reward)
        self.rollout.log_probs.append(getattr(self, "_last_log_prob", 0.0))
        self.rollout.values.append(getattr(self, "_last_value", 0.0))
        self.rollout.next_states.append(next_state)
        self.t += 1

        if len(self.rollout.states) >= self.rollout_size:
            self._update()
            self.rollout = empty_rollout()

    def end_episode(self) -> None:
        # Flush partial rollout if it's substantial
        if len(self.rollout.states) >= self.minibatch_size:
            self._update()
        self.rollout = empty_rollout()

    # ----- PPO update -----
    def _update(self) -> None:
        states = np.array(self.rollout.states, dtype=np.int64)
        actions = np.array(self.rollout.actions, dtype=np.int64)
        rewards = np.array(self.rollout.rewards, dtype=np.float32)
        old_log_probs = np.array(self.rollout.log_probs, dtype=np.float32)
        values = np.array(self.rollout.values, dtype=np.float32)
        next_states = np.array(self.rollout.next_states, dtype=np.int64)

        T = len(states)

        # bootstrap value of the final next-state
        with torch.no_grad():
            _, last_value = self.net(self._eye[next_states[-1]].unsqueeze(0))
            last_value = float(last_value.item())

        # GAE
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(T)):
            next_v = last_value if i == T - 1 else values[i + 1]
            td = rewards[i] + self.delta * next_v - values[i]
            gae = td + self.delta * self.gae_lambda * gae
            advantages[i] = gae
        returns = advantages + values

        # normalise advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # to torch
        s_t = self._eye[torch.as_tensor(states, device=self.device)]
        a_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        old_lp_t = torch.as_tensor(old_log_probs, device=self.device, dtype=torch.float32)
        adv_t = torch.as_tensor(advantages, device=self.device, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, device=self.device, dtype=torch.float32)

        # K epochs over minibatches
        idx = np.arange(T)
        for _ in range(self.n_epochs):
            self.rng.shuffle(idx)
            for start in range(0, T, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, device=self.device, dtype=torch.long)

                logits, value = self.net(s_t.index_select(0, mb_t))
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(a_t.index_select(0, mb_t))
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_t.index_select(0, mb_t))
                mb_adv = adv_t.index_select(0, mb_t)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret_t.index_select(0, mb_t))
                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

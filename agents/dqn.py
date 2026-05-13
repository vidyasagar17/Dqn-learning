"""DQN agent for the Calvano Bertrand environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent


@dataclass
class Transition:
    state: int
    action: int
    reward: float
    next_state: int


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.rng = rng

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.n_states = n_states
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, state_onehot: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state_onehot))
        x = F.relu(self.fc2(x))
        return self.head(x)


class DQNAgent(Agent):
    name = "DQN"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        beta: float = 4e-6,
        delta: float = 0.95,
        lr: float = 1e-3,
        hidden: int = 64,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        warmup: int = 1_000,
        target_tau: float = 5e-3,
        train_every: int = 1,
        rng: Optional[np.random.Generator] = None,
        device: str = "cpu",
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.beta = beta
        self.delta = delta
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_tau = target_tau
        self.train_every = train_every
        self.device = torch.device(device)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0

        torch.manual_seed(int(self.rng.integers(2**31)))

        self.online = QNetwork(n_states, n_actions, hidden).to(self.device)
        self.target = QNetwork(n_states, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, self.rng)

        self._eye = torch.eye(n_states, device=self.device)

    @property
    def epsilon(self) -> float:
        return float(np.exp(-self.beta * self.t))

    def _state_tensor(self, state: int) -> torch.Tensor:
        return self._eye[state].unsqueeze(0)

    def _states_tensor(self, states: np.ndarray) -> torch.Tensor:
        return self._eye[torch.as_tensor(states, device=self.device)]

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def greedy_action(self, state: int) -> int:
        with torch.no_grad():
            q = self.online(self._state_tensor(state))
        return int(q.argmax(dim=1).item())

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        self.buffer.push(state, action, reward, next_state)
        self.t += 1

        if len(self.buffer) >= max(self.batch_size, self.warmup) and self.t % self.train_every == 0:
            self._train_step()
            self._soft_update_target()

    def _train_step(self) -> None:
        batch = self.buffer.sample(self.batch_size)
        states = np.fromiter((tr.state for tr in batch), dtype=np.int64, count=len(batch))
        actions = np.fromiter((tr.action for tr in batch), dtype=np.int64, count=len(batch))
        rewards = np.fromiter((tr.reward for tr in batch), dtype=np.float32, count=len(batch))
        next_states = np.fromiter((tr.next_state for tr in batch), dtype=np.int64, count=len(batch))

        s = self._states_tensor(states)
        s_next = self._states_tensor(next_states)
        a = torch.as_tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        r = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)

        q_sa = self.online(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            q_next_max = self.target(s_next).max(dim=1).values
            target = r + self.delta * q_next_max

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

    def _soft_update_target(self) -> None:
        tau = self.target_tau
        with torch.no_grad():
            for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
                p_t.mul_(1 - tau).add_(p_o, alpha=tau)

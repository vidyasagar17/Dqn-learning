"""Base agent interface."""

from abc import ABC, abstractmethod


class Agent(ABC):
    name: str = "Agent"

    @abstractmethod
    def act(self, state: int) -> int:
        ...

    @abstractmethod
    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        ...

    def end_episode(self) -> None:
        pass

    def greedy_action(self, state: int) -> int:
        return self.act(state)

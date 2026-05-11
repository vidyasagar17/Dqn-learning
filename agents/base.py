"""Abstract agent interface. Every algorithm (Q, DQN, PPO) implements this."""

from abc import ABC, abstractmethod


class Agent(ABC):
    """Common interface for all pricing agents.

    The training loop calls:
        a = agent.act(state)
        ... environment step ...
        agent.observe(state, a, reward, next_state)

    States are integer IDs (0..n_states-1) for tabular agents and the same IDs
    for function-approximation agents (which one-hot or embed them internally).
    """

    name: str = "Agent"

    @abstractmethod
    def act(self, state: int) -> int:
        """Return an action index in [0, n_actions)."""

    @abstractmethod
    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Learn from one transition."""

    def end_episode(self) -> None:
        """Called at the end of a session. PPO uses this to flush its buffer; others ignore."""

    def greedy_action(self, state: int) -> int:
        """Deterministic best action — used for impulse-response and limit-strategy analysis."""
        return self.act(state)

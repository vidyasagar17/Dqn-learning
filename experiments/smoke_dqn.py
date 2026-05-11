"""
Fast unit-test smoke for DQN. Verifies the agent's interface and data flow
work correctly, without trying to actually train to convergence (which takes
many minutes of CPU time).

Run a real training session on your own machine — Calvano-style runs need
50k-200k steps per session, which is ~10-30 minutes per session on a laptop CPU.
"""

import time
import numpy as np

from agents.dqn import DQNAgent
from env.bertrand_logit import baseline_env


def main():
    env = baseline_env()
    rng = np.random.default_rng(0)

    a = DQNAgent(env.n_states, env.m, beta=1e-3, delta=0.95, lr=1e-3,
                 hidden=32, buffer_size=500, batch_size=16,
                 warmup=50, target_tau=1e-2, train_every=1,
                 rng=np.random.default_rng(rng.integers(2**31)))
    b = DQNAgent(env.n_states, env.m, beta=1e-3, delta=0.95, lr=1e-3,
                 hidden=32, buffer_size=500, batch_size=16,
                 warmup=50, target_tau=1e-2, train_every=1,
                 rng=np.random.default_rng(rng.integers(2**31)))

    state = env.encode_state([[0, 0]])
    t0 = time.time()
    for step in range(200):
        ai = a.act(state)
        bi = b.act(state)
        rewards = env.step([ai, bi])
        next_state = env.encode_state([[ai, bi]])
        a.observe(state, ai, float(rewards[0]), next_state)
        b.observe(state, bi, float(rewards[1]), next_state)
        state = next_state
    elapsed = time.time() - t0

    print(f"OK: 200 DQN-vs-DQN steps in {elapsed:.2f}s ({elapsed*5:.1f}ms/step)")
    print(f"Final greedy actions: a={a.greedy_action(state)}, b={b.greedy_action(state)}")
    print(f"Final epsilon: a={a.epsilon:.3f}, b={b.epsilon:.3f}")
    print()
    print("DQN interface is wired correctly.")
    print("Run real training (50k+ steps) on your own machine.")


if __name__ == "__main__":
    main()

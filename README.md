# Heterogeneous Algorithmic Collusion

Replicates and extends Calvano, Calzolari, Denicolò, and Pastorello (2020), *Artificial Intelligence, Algorithmic Pricing, and Collusion*.

Calvano's paper shows that two tabular Q-learning agents playing a Bertrand pricing game will quietly learn to collude. The result has been cited a lot in antitrust circles, but it assumes both agents are running the same algorithm with the same settings, which is not what real markets look like. This project asks what happens when the two agents are different. I run a grid of pairings across Q-learning, DQN, and PPO and compare what comes out.

## Setting it up

```bash
pip install -r requirements.txt
```

A few sanity checks to confirm everything imports and works:

```bash
python env/bertrand_logit.py        # should print Nash=1.473, Monopoly=1.925
python -m experiments.smoke_test    # Q-vs-Q quick check
python -m experiments.smoke_dqn     # DQN interface check
```

If all three run without errors, you're ready.

## Running an experiment

Everything goes through `experiments.run_experiment`. You pick a pairing and how many sessions to run:

```bash
python -m experiments.run_experiment --pairing DQN_DQN --sessions 20 --seed 42
```

The pairings you can pick from are `Q_Q`, `DQN_DQN`, `PPO_PPO`, `Q_DQN`, `DQN_Q`, `Q_PPO`, and `DQN_PPO`. Use a seed you haven't used before so you don't just re-run sessions you already have.

`Q_DQN` and `DQN_Q` are technically different pairings because the order decides which agent is firm 1. If you care about asymmetric effects you should run both. Since asymmetry is sort of the whole point of the project, you probably do care.

## Looking at the results

After a few experiments finish, the JSONs land in `results/`. From there:

```bash
# distribution of Δ across pairings
python -m analysis.plot_results results/*.json --out delta_comparison.png

# profit asymmetry table
python -m analysis.asymmetry results/*.json

# impulse-response plot for a specific pairing
python -m analysis.impulse_response_runner --pairing Q_Q   --out impulse_qq.png
python -m analysis.impulse_response_runner --pairing Q_DQN --out impulse_qdqn.png
```

The impulse-response plot is the one to look at first. The Calvano test for real collusion is to force one agent to deviate and see what happens. If both agents drop their prices for a few periods and then climb back up to where they were, that's a real reward-and-punishment dynamic. If the prices just stay flat or snap back instantly, the high prices were probably a coincidence, not coordination.

## What's in here

```
env/bertrand_logit.py                  Logit Bertrand environment
agents/base.py                         Abstract Agent interface
agents/q_learning.py                   Tabular Q-learning
agents/dqn.py                          Deep Q-Network (PyTorch)
agents/ppo.py                          PPO (PyTorch)
experiments/run_session.py             Generic 2-agent training loop
experiments/run_experiment.py          CLI for running N sessions
experiments/impulse_response.py        Forced-deviation analysis
experiments/smoke_test.py              Q-vs-Q smoke test
experiments/smoke_dqn.py               DQN smoke test
analysis/plot_results.py               Boxplot of Δ across pairings
analysis/asymmetry.py                  Profit asymmetry table
analysis/impulse_response_runner.py    Impulse response runner + plot
```

## How long things take

Roughly, on a laptop CPU:

- Q-vs-Q: about 10 seconds per session, so 50 sessions is over in 10 minutes
- DQN pairings: about 3 minutes per session at 200k steps
- PPO pairings: about 3 to 5 minutes per session at 200k steps

The longer pairings are worth running overnight with `caffeinate -i` on Mac so the machine doesn't sleep partway through.

## Citation

Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review*, 110(10), 3267–3297.

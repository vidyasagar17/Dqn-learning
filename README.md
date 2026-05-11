# Heterogeneous Algorithmic Collusion

Replication and extension of Calvano, Calzolari, Denicolò, Pastorello (2020),
"Artificial Intelligence, Algorithmic Pricing, and Collusion."

**Question:** Does Calvano's tacit-collusion result still hold when the two
pricing agents use different reinforcement-learning algorithms?

## Status

| Pairing | Done | Mean Δ |
|---|---|---|
| Q vs Q | yes (50 sessions) | 0.842 |
| DQN vs DQN | partial (12 sessions) | ~0.45 |
| PPO vs PPO | not yet | — |
| Q vs DQN | not yet | — |
| Q vs PPO | not yet | — |
| DQN vs PPO | not yet | — |

## Setup

```bash
pip install -r requirements.txt
```

## Verifying the build

Run these three to confirm everything compiles and runs:

```bash
python env/bertrand_logit.py             # prints Nash=1.473, Monopoly=1.925
python -m experiments.smoke_test          # Q-vs-Q quick check
python -m experiments.smoke_dqn           # DQN interface check
```

## Step-by-step run plan

### Step 1: Confirm Q-vs-Q baseline (already done)

```bash
python -m experiments.run_experiment --pairing Q_Q --sessions 50
```
Target: Δ ≈ 0.85. You got 0.842. ✓

### Step 2: Finish DQN-vs-DQN

You have 12 sessions; ideally bring it to 30. About 30 minutes per 10
sessions on a laptop CPU.

```bash
python -m experiments.run_experiment --pairing DQN_DQN --sessions 20 --seed 42
```

(Use `--seed 42` so you don't repeat your earlier seeds 0 and 1.)

### Step 3: The heterogeneous Q-vs-DQN test (most important)

This is the question the project is built to answer.

```bash
python -m experiments.run_experiment --pairing Q_DQN --sessions 20 --seed 100
python -m experiments.run_experiment --pairing DQN_Q --sessions 20 --seed 200
```

(`Q_DQN` and `DQN_Q` differ only in which agent is firm 1. Run both unless
you want to assume symmetry.)

### Step 4: PPO pairings

PPO is now implemented (`agents/ppo.py`). Run PPO-vs-PPO first to check it
collude at all in symmetric play, then the heterogeneous pairings:

```bash
python -m experiments.run_experiment --pairing PPO_PPO --sessions 10 --seed 300
python -m experiments.run_experiment --pairing Q_PPO --sessions 10 --seed 400
python -m experiments.run_experiment --pairing DQN_PPO --sessions 10 --seed 500
```

PPO is on-policy, so each session takes about as long as DQN (a few
minutes). Plan for ~30-60 minutes per pairing.

### Step 5: Analysis

Once you have results JSON files for the pairings you care about:

```bash
# delta comparison boxplot across all pairings
python -m analysis.plot_results results/*.json --out delta_comparison.png

# profit-asymmetry table — measurement (iii) from the report
python -m analysis.asymmetry results/*.json

# impulse-response plot for any pairing — measurement (iv) from the report
python -m analysis.impulse_response_runner --pairing Q_Q     --out impulse_qq.png
python -m analysis.impulse_response_runner --pairing DQN_DQN --out impulse_dqndqn.png
python -m analysis.impulse_response_runner --pairing Q_DQN   --out impulse_qdqn.png
```

The impulse-response plot is the single most informative diagnostic. If
prices show a sharp drop on the deviation period followed by a few
periods of mutual punishment and gradual recovery, that's the Calvano
signature of real collusion. If they just stay flat or drift, it's not
real coordination.

## File map

```
env/bertrand_logit.py                    Logit Bertrand environment
agents/base.py                           Abstract Agent interface
agents/q_learning.py                     Tabular Q-learning
agents/dqn.py                            Deep Q-Network (PyTorch)
agents/ppo.py                            Proximal Policy Optimization (PyTorch)
experiments/run_session.py               Generic 2-agent training loop
experiments/run_experiment.py            CLI runner — N sessions of any pairing
experiments/impulse_response.py          Forced-deviation analysis module
experiments/smoke_test.py                Q-vs-Q quick check
experiments/smoke_dqn.py                 DQN interface check
analysis/plot_results.py                 Boxplot of delta across pairings
analysis/asymmetry.py                    Profit-asymmetry table
analysis/impulse_response_runner.py      Impulse-response runner with plot
```

## Compute estimates (laptop CPU)

* Q-Q: ~10s per session
* DQN-DQN, Q-DQN: ~3 min per session at 200k steps
* PPO-PPO, Q-PPO, DQN-PPO: ~3-5 min per session at 200k steps

To finish all the runs in step 3 and step 4 above (~80 sessions), expect
roughly 4-6 hours total. Plug in your laptop and run overnight.

## Citation

Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).
Artificial Intelligence, Algorithmic Pricing, and Collusion.
*American Economic Review*, 110(10), 3267-3297.

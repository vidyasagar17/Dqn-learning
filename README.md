# Heterogeneous Algorithmic Collusion
 
Replication and extension of Calvano, Calzolari, Denicolò, and Pastorello (2020), *Artificial Intelligence, Algorithmic Pricing, and Collusion*.
 
The original paper shows that two tabular Q-learning agents in a Bertrand pricing game learn to tacitly collude. I wanted to know whether that result generalizes when the two agents use different RL algorithms, so I'm running a grid of pairings across Q-learning, DQN, and PPO.
 
## Getting set up
 
```bash
pip install -r requirements.txt
```
 
Quick check that everything imports and runs:
 
```bash
python env/bertrand_logit.py        # should print Nash=1.473, Monopoly=1.925
python -m experiments.smoke_test    # Q-vs-Q quick check
python -m experiments.smoke_dqn     # DQN interface check
```
 
## Running experiments
 
The main entry point is `experiments.run_experiment`, which takes a `--pairing` flag and a session count:
 
```bash
python -m experiments.run_experiment --pairing DQN_DQN --sessions 20 --seed 42
```
 
Pairings available: `Q_Q`, `DQN_DQN`, `PPO_PPO`, `Q_DQN`, `DQN_Q`, `Q_PPO`, `DQN_PPO`. Pass a seed you haven't used before to avoid repeating earlier runs.
 
`Q_DQN` and `DQN_Q` are separate pairings — they only differ in which agent is firm 1. Run both if you don't want to assume symmetry (and given that the whole point is heterogeneity, you probably shouldn't).
 
## Analysis
 
Once you have results JSON files:
 
```bash
# Δ comparison across pairings
python -m analysis.plot_results results/*.json --out delta_comparison.png
 
# profit asymmetry — measurement (iii) in the report
python -m analysis.asymmetry results/*.json
 
# impulse response for a given pairing — measurement (iv)
python -m analysis.impulse_response_runner --pairing Q_Q   --out impulse_qq.png
python -m analysis.impulse_response_runner --pairing Q_DQN --out impulse_qdqn.png
```
 
The impulse-response plot is the most useful diagnostic by a wide margin. If a forced deviation triggers a sharp price drop followed by a few periods of mutual punishment and a gradual return to the prior level, that's the Calvano signature — real reward-punishment coordination, not just two agents happening to sit at high prices.
 
## Code layout
 
```
env/bertrand_logit.py                  Logit Bertrand environment
agents/base.py                         Abstract Agent interface
agents/q_learning.py                   Tabular Q-learning
agents/dqn.py                          Deep Q-Network (PyTorch)
agents/ppo.py                          PPO (PyTorch)
experiments/run_session.py             Generic 2-agent training loop
experiments/run_experiment.py          CLI — N sessions of any pairing
experiments/impulse_response.py        Forced-deviation analysis
experiments/smoke_test.py              Q-vs-Q smoke test
experiments/smoke_dqn.py               DQN smoke test
analysis/plot_results.py               Boxplot of Δ across pairings
analysis/asymmetry.py                  Profit asymmetry table
analysis/impulse_response_runner.py    Impulse response runner + plot
```
 
## Rough compute notes
 
On a laptop CPU:
 
- Q-vs-Q: ~10s per session
- DQN pairings: ~3 min per session at 200k steps
- PPO pairings: ~3–5 min per session at 200k steps
## Citation
 
Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review*, 110(10), 3267–3297.
 


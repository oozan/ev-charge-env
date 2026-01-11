EVChargeEnv: An OpenEnv Benchmark for EV Charging Optimization

1. Motivation

As AI agents move from static prediction to acting autonomously in dynamic environments, we need richer environments than toy grids and games. One domain that naturally combines uncertainty, long-horizon planning, and multi-objective decision-making is electric vehicle (EV) charging.

EVChargeEnv is my contribution to the OpenEnv Challenge. It simulates a simplified but realistic EV charging process where an agent must decide how much to charge at each timestep while adapting to fluctuating electricity prices and grid load.

The core objective is:

Reach full battery while minimizing cost and avoiding grid overload.

This makes EVChargeEnv a clean and interpretable environment that still contains meaningful complexity.

---

2. Environment Design

EVChargeEnv exposes a continuous-control RL task with a 4-dimensional state:

- charge_level ∈ [0, 1] – battery state of charge
- price ∈ [0, 1] – dynamic energy price
- grid_load ∈ [0, 1] – current grid stress/instability
- time_step_norm ∈ [0, 1] – normalized timestep

Agents output a continuous charging rate between 0 and 1.

Scenarios

To test robustness, EVChargeEnv includes three difficulty modes:

- easy – smooth price/load curves and short episodes
- medium – balanced volatility (default scenario)
- hard – noisy price/load dynamics and slower charging

---

3. Reward Function

The reward balances several competing factors:

- progress_reward – reward for increasing charge
  – cost_penalty – charging during high prices costs more
  – overload_penalty – charging when grid load is high is discouraged
  – time_penalty – each step costs a tiny penalty to encourage faster execution

This encourages agents to:

- charge aggressively during low-price, low-load periods
- slow or stop charging during peak price/load
- finish charging efficiently

---

4. Implementation and OpenEnv Integration

The environment is implemented using gymnasium and structured to reflect OpenEnv specifications. Key files include:

- env/ev_charge_env.py – main environment logic
- openenv.yaml – metadata describing observation/action spaces, rewards, and termination criteria
- run_evaluation.py – produces standardized JSON outputs for assessment

Example output:

{
"avg_reward": -1.23,
"avg_steps": 31.2,
"episodes": 5
}

Published on the Hugging Face Hub:
https://huggingface.co/oozan/EVChargeEnv

---

5. Baseline Agents and Training

Baselines included:

Random Baseline – ignores price/load.
Price-Aware Baseline – charges more when price is low.
PyTorch Policy-Gradient Agent – small neural model trained with REINFORCE.

The learned agent shows:

- improved reward
- sensible patterns
- adaptation to medium scenario

---

6. Running the Environment

Install:

pip install -r requirements.txt

Evaluate:

python run_evaluation.py

Heuristic baseline:

python run_price_aware_evaluation.py

Train agent:

python train_evchargeenv_pg.py

Notebook also included for Colab execution.

---

7. Future Improvements

Possible extensions:

- Day/night pricing cycles
- Renewable energy influence
- Emergency events / blackouts
- Battery degradation modeling
- PPO/SAC/LLM-based agents
- Visualization tools

---

Final Thoughts

EVChargeEnv is:

- simple
- realistic
- modular
- OpenEnv-compliant

It provides a practical environment for research on planning and resource optimization under uncertainty.

Repo:
https://huggingface.co/oozan/EVChargeEnv

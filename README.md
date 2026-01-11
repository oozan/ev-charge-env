---
tags:
  - reinforcement-learning
  - openenv
  - environment
  - gymnasium
license: mit
library_name: openenv
---

<h1 align="center">⚡ EVChargeEnv</h1>
<p align="center">
  <img src="assets/evchargeenv-banner.png" width="800" />
</p>

<h3 align="center">Green Agent Benchmark for EV Charging Optimization</h3>

---

## Overview

EVChargeEnv is a lightweight, stochastic reinforcement-learning environment designed for the  
AgentX + AgentBeats Competition (Berkeley RDI 2025).

It simulates:

- Electric vehicle battery charging
- Dynamic electricity pricing
- Fluctuating grid load
- Continuous control actions
- Multi-objective tradeoffs (cost vs. speed vs. grid stability)

---

## Task Goal

The purple agent must:

- Charge the EV battery to full (1.0)
- Minimize electricity cost
- Avoid high grid load
- Adapt to changing conditions

---

## State Space (Observation)

The agent receives:

charge_level (0-1), price (0-1), grid_load (0-1), time_step_norm (0-1)

---

## Action Space

Continuous charge rate 0.0 → 1.0.

---

## Reward Function

Reward combines:

- progress_reward

* cost_penalty
* overload_penalty
* time_penalty

---

## Scenarios

easy / medium / hard difficulty with different volatility and load patterns.

---

## Episode Termination

Ends if full charge or max steps reached.

---

## Example Agent Behaviors

Greedy agent = fast but expensive  
Price-aware agent = slower but cheaper  
Random agent = unstable

---

## Evaluation Output

Running:

python run_evaluation.py

Generates JSON like:

{
"avg_reward": ...,
"avg_steps": ...,
"episodes": 5
}

---

## Docker Support

Image: oozan/evchargeenv:latest

---

## File Structure

env/  
agent/  
run_evaluation.py  
Dockerfile  
requirements.txt  
README.md

---

## Future Improvements

- renewable energy factor
- blackout events
- degradation model
- RL baseline
- trajectory visualizer
- mini-game UI

## Benchmark Specification

This repository also includes a machine-readable benchmark manifest:

- `evchargeenv_manifest.json`

It documents:

- state and action spaces
- reward components
- termination conditions
- supported scenarios (`easy`, `medium`, `hard`)
- evaluation output format (JSON fields)

This makes EVChargeEnv easier to integrate as a standardized benchmark and aligns with the spirit of the OpenEnv challenge: environments that are transparent, reproducible, and extensible.
''

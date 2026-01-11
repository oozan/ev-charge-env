import json
from env.ev_charge_env import EVChargeEnv
from agent.baseline_agent import BaselineAgent

def run_episode(env, agent):
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated or steps >= 200:
            break

    return total_reward, steps

def main():
    env = EVChargeEnv()
    agent = BaselineAgent()

    rewards = []
    steps_list = []

    for _ in range(5):
        total_reward, steps = run_episode(env, agent)
        rewards.append(total_reward)
        steps_list.append(steps)

    output = {
        "avg_reward": sum(rewards) / len(rewards),
        "avg_steps": sum(steps_list) / len(steps_list),
        "episodes": len(rewards)
    }

    print(json.dumps(output))

    # Save JSON for reproducibility
    with open("sample_output.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()

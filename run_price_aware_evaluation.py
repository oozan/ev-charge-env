import json
from env.ev_charge_env import EVChargeEnv
from agent.price_aware_agent import PriceAwareAgent


def run_episode(env, agent, seed=None):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    return total_reward, steps


def main():
    # You can change scenario to "easy" / "medium" / "hard"
    env = EVChargeEnv(scenario="medium")
    agent = PriceAwareAgent()

    rewards = []
    steps_list = []

    num_episodes = 10
    for i in range(num_episodes):
        total_reward, steps = run_episode(env, agent, seed=i)
        rewards.append(total_reward)
        steps_list.append(steps)

    output = {
        "agent_type": "price_aware",
        "scenario": "medium",
        "avg_reward": sum(rewards) / len(rewards),
        "avg_steps": sum(steps_list) / len(steps_list),
        "episodes": num_episodes,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

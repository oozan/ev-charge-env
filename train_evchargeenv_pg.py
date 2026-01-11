import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.ev_charge_env import EVChargeEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


def make_env():
    # You can change scenario here: "easy", "medium", "hard"
    return EVChargeEnv(scenario="medium")


def run_episode(env, model, device, gamma=0.99):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    log_probs = []
    values = []
    rewards = []

    done = False
    while not done:
        logits, value = model(obs.unsqueeze(0))  # [1, obs_dim]
        # Gaussian policy for continuous action in [0, 1]
        mean = torch.sigmoid(logits.squeeze(0))  # [act_dim]
        std = torch.ones_like(mean) * 0.2  # fixed std

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, 0.0, 1.0)

        log_prob = dist.log_prob(action).sum()

        np_action = action_clipped.detach().cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(np_action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))

        done = terminated or truncated
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    # Compute returns
    returns = []
    G = torch.tensor(0.0, device=device)
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.stack(returns)
    values = torch.stack(values).squeeze(-1)
    log_probs = torch.stack(log_probs)

    advantages = returns - values.detach()

    policy_loss = -(log_probs * advantages).mean()
    value_loss = (returns - values).pow(2).mean()

    total_reward = float(sum(r.item() for r in rewards))

    return policy_loss, value_loss, total_reward, len(rewards)


def train(num_episodes=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    reward_history = []

    for episode in range(1, num_episodes + 1):
        policy_loss, value_loss, total_reward, steps = run_episode(env, model, device)

        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(total_reward)

        if episode % 10 == 0:
            avg_last = np.mean(reward_history[-10:])
            print(
                f"Episode {episode:4d} | "
                f"ep_reward={total_reward:.2f} | "
                f"avg_last10={avg_last:.2f} | steps={steps}"
            )

    print("Training finished.")
    print(f"Average reward over last 20 episodes: {np.mean(reward_history[-20:]):.2f}")


if __name__ == "__main__":
    train(num_episodes=200)

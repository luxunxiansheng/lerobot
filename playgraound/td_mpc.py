import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Hyperparameters
H = 5  # Planning horizon
N = 512  # Number of sampled trajectories
K = 64  # Top-k trajectories
J = 6  # Planning iterations
GAMMA = 0.99  # Discount factor
LAMBDA = 0.5  # Temporal weighting
C1, C2, C3 = 0.5, 0.1, 2.0  # Loss coefficients
BATCH_SIZE = 512
BUFFER_SIZE = 10000
SEED_STEPS = 1000

# Neural Network for Task-Oriented Latent Dynamics (TOLD) components
class TOLD(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=50):
        super(TOLD, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim)
        )  # Encodes state to latent space
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512), nn.ReLU(), nn.Linear(512, latent_dim)
        )  # Predicts next latent state
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512), nn.ReLU(), nn.Linear(512, 1)
        )  # Predicts reward
        self.value_estimator = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512), nn.ReLU(), nn.Linear(512, 1)
        )  # Estimates Q-value
        self.policy_network = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, action_dim)
        )  # Suggests actions

    def forward(self, s, a):
        z = self.state_encoder(s)
        za = torch.cat([z, a], dim=-1)
        z_next = self.dynamics_predictor(za)
        r_hat = self.reward_predictor(za)
        q_hat = self.value_estimator(za)
        a_hat = self.policy_network(z)
        return z, z_next, r_hat, q_hat, a_hat


# Environment setup
env = gymnasium.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = 1  # Simplified to continuous action in [-1, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
model = TOLD(state_dim, action_dim).to(device)
target_model = TOLD(state_dim, action_dim).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Replay buffer
buffer = deque(maxlen=BUFFER_SIZE)

# MPPI Planning
def plan(model: TOLD, s: np.ndarray, horizon: int = H, n_samples: int = N, n_elite: int = K, n_iter: int = J) -> int:
    s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
    z = model.state_encoder(s_tensor)
    mu, sigma = torch.zeros(horizon, action_dim).to(device), 2 * torch.ones(horizon, action_dim).to(device)
    
    for _ in range(n_iter):
        actions = torch.randn(n_samples, horizon, action_dim).to(device) * sigma + mu
        returns = torch.zeros(n_samples).to(device)
        
        z_t = z.repeat(n_samples, 1)
        for t in range(horizon):
            a_t = actions[:, t]
            za_t = torch.cat([z_t, a_t], dim=-1)
            r_t = model.reward_predictor(za_t).squeeze(-1)
            z_next = model.dynamics_predictor(za_t)
            returns += (GAMMA ** t) * r_t
            z_t = z_next
        
        # Add terminal value
        a_H = model.policy_network(z_t)
        q_H = model.value_estimator(torch.cat([z_t, a_H], dim=-1)).squeeze(-1)
        returns += (GAMMA ** horizon) * q_H
        
        # Update distribution
        topk_idx = torch.topk(returns, n_elite, dim=0)[1]
        topk_actions = actions[topk_idx]
        weights = torch.exp(0.5 * returns[topk_idx])  # Temperature = 0.5
        weights /= weights.sum()
        mu = (weights.unsqueeze(-1).unsqueeze(-1) * topk_actions).sum(dim=0)
        sigma = torch.sqrt(((topk_actions - mu)**2 * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0))
        sigma = torch.clamp(sigma, min=0.05)  # Exploration constraint
    
    return (mu[0] + sigma[0] * torch.randn(action_dim, device=device)).detach().cpu().int().item()


total_steps = 0
for episode in range(500):
    s = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        total_steps += 1
        if total_steps < SEED_STEPS:
            a = env.action_space.sample()  # Random action
        else:
            a = plan(model, s)  # Plan with TD-MPC
        
        a = np.clip(a, 0, 1)  # Simplified action range
        
        s_next, r, done, _,_ = env.step(a) 
        buffer.append((s, a, r, s_next))
        episode_reward += r
        s = s_next
        
        if len(buffer) >= BATCH_SIZE and total_steps >= SEED_STEPS:
            batch = random.sample(buffer, BATCH_SIZE)
            s_batch, a_batch, r_batch, s_next_batch = map(lambda x: torch.FloatTensor(x).to(device), zip(*batch))
            
            # Forward pass
            z, z_next, r_hat, q_hat, a_hat = model(s_batch, a_batch)
            with torch.no_grad():
                _, z_next_target, _, q_next_target, _ = target_model(s_next_batch, model.policy_network(z_next))
            
            # Losses
            reward_loss = C1 * (r_hat - r_batch).pow(2).mean()
            value_loss = C2 * (q_hat - (r_batch + GAMMA * q_next_target)).pow(2).mean()
            consistency_loss = C3 * (z_next - target_model.state_encoder(s_next_batch)).pow(2).mean()
            policy_loss = -model.value_estimator(torch.cat([z.detach(), model.policy_network(z)], dim=-1)).mean()
            total_loss = reward_loss + value_loss + consistency_loss + policy_loss
            
            # Update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update target network
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(0.99 * target_param.data + 0.01 * param.data)
    
    print(f"Episode {episode}, Reward: {episode_reward}")

env.close()



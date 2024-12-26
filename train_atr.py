import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from collections import namedtuple
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples

from torch.utils.tensorboard import SummaryWriter
from MarioKartEnv import MarioKartEnv
from rl_plotter.logger import Logger

PERSamples = namedtuple('PERSamples', ReplayBufferSamples._fields+('weights', 'indices'))

# Noisy Network
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer implementation that inherits from stable-baselines3's ReplayBuffer.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=True
        )
        # PER parameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        # Initialize priorities
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

    def add(self, *args, **kwargs):
        """Override add method to update priorities for new transitions"""
        idx = self.pos
        super().add(*args, **kwargs)
        # New experience gets max priority
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int, env=None):
        """Sample a batch of experiences with priorities"""
        upper_bound = self.buffer_size if self.full else self.pos

        # Get sampling probabilities from priorities
        priorities = self.priorities[:upper_bound]
        probabilities = priorities ** self.alpha
        probabilities = probabilities / np.sum(probabilities)

        # Sample indices with priority weights
        indices = np.random.choice(upper_bound, size=batch_size, p=probabilities)

        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (upper_bound * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Get samples using parent class method
        samples = super()._get_samples(indices, env)
        
        # Add importance sampling weights and indices to samples
        
        samples = PERSamples(*samples, 
            weights=torch.FloatTensor(weights).to(self.device),
            indices=indices
        )

        return samples

    def update_priorities(self, indices, priorities):
        """Update priorities for experience replay"""
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

def make_env():
    def thunk():
        env = MarioKartEnv()
        return env

    return thunk

def build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class MultiOrderDWConv(nn.Module):

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x

class MultiOrderGatedAggregation(nn.Module):

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.SiLU(),
            MultiOrderGatedAggregation(embed_dims=32),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.SiLU(),
            MultiOrderGatedAggregation(embed_dims=64),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.SiLU(),
            MultiOrderGatedAggregation(embed_dims=64),
            nn.Flatten()
        )

        feature_size = 9216

        # Replace standard linear layers with NoisyLinear
        self.advantage_hidden = NoisyLinear(feature_size, 512)
        self.advantage_output = NoisyLinear(512, self.n * self.n_atoms)
        
        self.value_hidden = NoisyLinear(feature_size, 512)
        self.value_output = NoisyLinear(512, self.n_atoms)

    def reset_noise(self):
        self.advantage_hidden.reset_noise()
        self.advantage_output.reset_noise()
        self.value_hidden.reset_noise()
        self.value_output.reset_noise()

    def forward(self, x):
        features = self.features(x)
        
        advantage_hidden = F.silu(self.advantage_hidden(features))
        advantage = self.advantage_output(advantage_hidden).view(-1, self.n, self.n_atoms)
        
        value_hidden = F.silu(self.value_hidden(features))
        value = self.value_output(value_hidden).view(-1, 1, self.n_atoms)
        
        # Combine value and advantage using dueling architecture
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_dist

    def get_action(self, x, action=None):
        q_dist = self(x)
        pmfs = torch.softmax(q_dist, dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]

# N-step returns calculation helper
class NStepRewardBuffer:
    def __init__(self, gamma, n_steps):
        self.gamma = gamma
        self.n_steps = n_steps
        self.reset()
    
    def reset(self):
        self.rewards = []
        self.actions = []
        self.states = []
        self.next_states = []
        self.dones = []
    
    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def get_transition(self):
        if len(self.rewards) < self.n_steps:
            return None
        
        state = self.states[0]
        action = self.actions[0]
        
        # Calculate n-step discounted reward
        n_step_reward = 0
        done = False
        for i in range(self.n_steps):
            if i >= self.n_steps:
                break
            next_state = self.next_states[i]
            n_step_reward += self.gamma ** i * self.rewards[i]
            if self.dones[i]:
                done = True
                break
        
        # Remove oldest transition
        self.states.pop(0)
        self.next_states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.dones.pop(0)
        
        return (state, action, n_step_reward, next_state, done)


@dataclass
class Args:
    exp_name: str = "Rainbow"
    """the name of this experiment"""
    num_envs: int = 1
    """only support 1 environment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    total_timesteps: int = 500000 # 500k
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    n_atoms: int = 51
    """the number of atoms"""
    n_steps: int = 3
    """the number of n-steps"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    alpha: float = 0.6
    """the PER alpha"""
    beta: float = 0.4
    """the PER beta"""
    beta_increment: float = 0.001
    """the PER beta increment"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    save_freq: int = 5000
    """the frequency of saving model"""

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    logger = Logger(exp_name="Rainbow", env_name="MarioKart")
 
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"Rainbow__{int(time.time())}"

    writer = SummaryWriter(f"tensorboard/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env() for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = PrioritizedReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        alpha=args.alpha,  
        beta=args.beta,  
        beta_increment=args.beta_increment,
        epsilon=1e-6 
    )

    n_step_buffer = NStepRewardBuffer(gamma=args.gamma, n_steps=args.n_steps)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    total_return = 0
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        q_network.reset_noise()
        actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
        actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        n_step_buffer.append(obs[0], actions[0], rewards[0], real_next_obs[0], terminations[0])
        transition = n_step_buffer.get_transition()
        if transition is not None:
            s, a, r, next_s, done = transition
            rb.add(
                np.array([s]), 
                np.array([next_s]), 
                np.array([a]), 
                np.array([r]), 
                np.array([done]),
                [{}]
            )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # log to Logger
        total_return += sum(rewards)
        if terminations.any() or truncations.any():
            logger.update(score=[total_return], total_steps=global_step)
            total_return = 0
            n_step_buffer.reset()

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                target_network.reset_noise()
                with torch.no_grad():
                    # Double DQN
                    next_actions, _ = q_network.get_action(data.next_observations)
                    _, next_pmfs = target_network.get_action(data.next_observations, next_actions)

                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                    # projection
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
                # Calculate element-wise loss for priority updates
                element_wise_loss = -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)
                
                # Apply importance sampling weights
                loss = (element_wise_loss * data.weights).mean()

                # Update priorities in buffer
                td_errors = element_wise_loss.detach().cpu().numpy()
                new_priorities = np.abs(td_errors) + rb.epsilon
                rb.update_priorities(data.indices, new_priorities)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("training/beta", rb.beta, global_step)
                    writer.add_scalar("training/priority_mean", rb.priorities[:rb.pos].mean(), global_step)
                    writer.add_scalar("training/priority_max", rb.priorities[:rb.pos].max(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
        if (global_step + 1) % args.save_freq == 0 or global_step == args.total_timesteps-1:
            model_path = f"checkpoints/{args.exp_name}_{global_step+1}.model"
            model_data = {
                "model_weights": q_network.state_dict(),
                "args": vars(args),
            }
            torch.save(model_data, model_path)
            print(f"model saved to {model_path}")

    envs.close()
    writer.close()

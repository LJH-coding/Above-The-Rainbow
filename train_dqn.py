from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import is_wrapped
import torch.nn as nn
import torch
import numpy as np
from gymnasium import spaces
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from MarioKartEnv import MarioKartEnv
from rl_plotter.logger import Logger

class RLPlotterCallback(BaseCallback):
    def __init__(self, logger: Logger, verbose=0):
        super().__init__(verbose)
        self._logger = logger
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_episode_reward += reward
        done = self.locals["dones"][0]
        if done:
            self.update()
        return True

    def update(self):
        self._logger.update(
            score=[self.current_episode_reward],
            total_steps=self.num_timesteps
        )
        self.current_episode_reward = 0

    def on_training_end(self) -> None:
        self.update()

class CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def main():
    env = MarioKartEnv()
    logger = Logger(exp_name="DQN", env_name="MarioKart")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # Save every 5000 steps
        save_path="./checkpoints/",
        name_prefix="dqn_model"
    )
    plotter_callback = RLPlotterCallback(logger)
    callbacks = [checkpoint_callback, plotter_callback]
    
    policy_kwargs = dict(
        features_extractor_class=CNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    # Initialize DQN model
    print("Initializing DQN model...")
    model = DQN(
        "CnnPolicy",  # Using CNN policy for image observations
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=2000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log="./tensorboard/",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1
    )
    
    # Train the model
    total_timesteps = 100000 
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model.save("./checkpoints/dqn_final_model")

if __name__ == "__main__":
    main()

import argparse
import gymnasium as gym
import numpy as np
import torch
import time
from train_atr import QNetwork, make_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="path to the trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of episodes to evaluate")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="number of environments to run in parallel")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="use cuda if available")
    parser.add_argument("--render", action="store_true", default=False,
                        help="render the environment")
    return parser.parse_args()

def evaluate(envs, model, device, num_episodes=10):
    model.eval()
    returns = []
    
    # Track episodes completed
    episodes_completed = 0
    episode_returns = np.zeros(envs.num_envs)
    obs, _ = envs.reset()
    
    while episodes_completed < num_episodes:
        # Get actions from model
        with torch.no_grad():
            actions, _ = model.get_action(torch.FloatTensor(obs).to(device))
            actions = actions.cpu().numpy()
        
        # Execute actions in environments
        next_obs, rewards, terminations, truncations, _ = envs.step(actions)
        
        # Update episode returns
        episode_returns += rewards
        
        # Check for completed episodes
        completed = terminations | truncations
        if completed.any():
            # Store the returns for completed episodes
            for env_idx in range(envs.num_envs):
                if completed[env_idx]:
                    returns.append(episode_returns[env_idx])
                    episode_returns[env_idx] = 0
                    episodes_completed += 1
                    print(f"Episode {len(returns)}: Return = {returns[-1]:.2f}")
                    
                    if episodes_completed >= num_episodes:
                        break
        
        obs = next_obs
    
    return returns

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create vectorized environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(training=False) for _ in range(args.num_envs)]
    )
    print(f"Created {args.num_envs} environments")
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint["args"]
    
    # Initialize model with same parameters as training
    model = QNetwork(
        envs,
        n_atoms=model_args["n_atoms"],
        v_min=model_args["v_min"],
        v_max=model_args["v_max"]
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_weights"])
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate model
    print(f"\nEvaluating model for {args.episodes} episodes...")
    start_time = time.time()
    returns = evaluate(envs, model, device, args.episodes)
    total_time = time.time() - start_time
    
    # Print statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nResults over {args.episodes} episodes:")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Min return: {min(returns):.2f}")
    print(f"Max return: {max(returns):.2f}")
    print(f"Evaluation time: {total_time:.2f} seconds")
    print(f"Episodes per second: {args.episodes/total_time:.2f}")
    
    envs.close()

if __name__ == "__main__":
    main()

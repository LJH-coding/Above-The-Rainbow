import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as transforms
from EnvReceiver import EnvReceiver
from collections import deque
import matplotlib.pyplot as plt

def check_early_stop(frame):
    H, W, _ = frame.shape
    # print(f'frame shape: {frame.shape}')
    crop_size_lower = (120, 160)
    # crop_size_upper = (50, 100)

    crop_size_left = (100, 70) # (y, x)
    crop_size_right = (100, 70)

    center_x_lower, center_y_lower = W // 2, int(H * 0.95)
    # center_x_upper, center_y_upper = W // 2, int(H * 0.5)

    center_x_left, center_y_left = int(W * 0.35), int(H * 0.65)
    center_x_right, center_y_right = int(W * 0.65), int(H * 0.65)
    

    cropped_frame_lower = frame[center_y_lower - crop_size_lower[0]//2:center_y_lower + crop_size_lower[0]//2,
                          center_x_lower - crop_size_lower[1]//2:center_x_lower + crop_size_lower[1]//2]
    # cropped_frame_upper = frame[center_y_upper - crop_size_upper[0]//2:center_y_upper + crop_size_upper[0]//2,
    #                       center_x_upper - crop_size_upper[1]//2:center_x_upper + crop_size_upper[1]//2]
    
    cropped_frame_left = frame[center_y_left - crop_size_left[0]//2:center_y_left + crop_size_left[0]//2,
                          center_x_left - crop_size_left[1]//2:center_x_left + crop_size_left[1]//2]
    cropped_frame_right = frame[center_y_right - crop_size_right[0]//2:center_y_right + crop_size_right[0]//2,
                          center_x_right - crop_size_right[1]//2:center_x_right + crop_size_right[1]//2]
   

    # plt.imshow(cropped_frame_lower)
    # plt.title("Cropped Lower Road Region")
    # plt.savefig("road_lower.png")
    # plt.close()

    # plt.imshow(cropped_frame_upper)
    # plt.title("Cropped Upper Road Region")
    # plt.savefig("road_upper.png")
    # plt.close()

    # plt.imshow(cropped_frame_left)
    # plt.title("Cropped Left Road Region")
    # plt.savefig("road_left.png")
    # plt.close()

    # plt.imshow(cropped_frame_right)
    # plt.title("Cropped Right Road Region")
    # plt.savefig("road_right.png")
    # plt.close()

    diff_threshold = 25

    road_pixels_lower = np.sum(
        (np.abs(cropped_frame_lower[:, :, 0] - cropped_frame_lower[:, :, 1]) <= diff_threshold)
        # (np.abs(cropped_frame_lower[:, :, 0] - cropped_frame_lower[:, :, 2]) <= diff_threshold) |
        # (np.abs(cropped_frame_lower[:, :, 1] - cropped_frame_lower[:, :, 2]) <= diff_threshold)
    )

    # road_pixels_upper = np.sum(
    #     (np.abs(cropped_frame_upper[:, :, 0] - cropped_frame_upper[:, :, 1]) <= diff_threshold) &
    #     (np.abs(cropped_frame_upper[:, :, 0] - cropped_frame_upper[:, :, 2]) <= diff_threshold) &
    #     (np.abs(cropped_frame_upper[:, :, 1] - cropped_frame_upper[:, :, 2]) <= diff_threshold)
    # )

    road_pixels_left = np.sum(
        (np.abs(cropped_frame_left[:, :, 0] - cropped_frame_left[:, :, 1]) <= diff_threshold)
        # (np.abs(cropped_frame_left[:, :, 0] - cropped_frame_left[:, :, 2]) <= diff_threshold) |
        # (np.abs(cropped_frame_left[:, :, 1] - cropped_frame_left[:, :, 2]) <= diff_threshold)
    )

    road_pixels_right = np.sum(
        (np.abs(cropped_frame_right[:, :, 0] - cropped_frame_right[:, :, 1]) <= diff_threshold)
        # (np.abs(cropped_frame_right[:, :, 0] - cropped_frame_right[:, :, 2]) <= diff_threshold) |
        # (np.abs(cropped_frame_right[:, :, 1] - cropped_frame_right[:, :, 2]) <= diff_threshold)
    )

    total_pixels_lower = cropped_frame_lower.shape[0] * cropped_frame_lower.shape[1]
    # total_pixels_upper = cropped_frame_upper.shape[0] * cropped_frame_upper.shape[1]
    
    total_pixels_left = cropped_frame_left.shape[0] * cropped_frame_left.shape[1]
    total_pixels_right = cropped_frame_right.shape[0] * cropped_frame_right.shape[1]
    

    road_pixel_ratio_lower = road_pixels_lower / total_pixels_lower
    # road_pixel_ratio_upper = road_pixels_upper / total_pixels_upper

    road_pixel_ratio_left = road_pixels_left / total_pixels_left
    road_pixel_ratio_right = road_pixels_right / total_pixels_right


    # return road_pixel_ratio_lower < 0.5 or road_pixel_ratio_upper < 0.1
    return (road_pixel_ratio_left < 0.2 and road_pixel_ratio_right < 0.2) or road_pixel_ratio_lower < 0.2 or road_pixel_ratio_left < 0.1 or road_pixel_ratio_right < 0.1

class EarlyStop():
    def __init__(self, env):
        self.steps = 0
        self.env = env
        self.reward_buffer = deque(maxlen=90)

    def reset(self):
        self.steps = 0
        self.reward_buffer.clear()
        ob = self.env.reset()
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        # Early stopping logic
        self.steps += 1
        self.reward_buffer.append(reward)
        # if check_early_stop(ob) or (self.steps > 90 and np.mean(self.reward_buffer) <= -0.099):
        if (self.steps > 90 and np.mean(self.reward_buffer) <= -0.099):
            early_stop_reward = (1250 - self.steps) * -0.1
            reward += early_stop_reward
            done = True

        return ob, reward, done, info

class FrameStack():
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(self._preprocess_frame(ob))
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(self._preprocess_frame(ob))
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return torch.stack(list(self.frames), dim=0).squeeze()

    def _preprocess_frame(self, frame):
        return self.preprocess(frame)

class EnvWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # Observation space for stacked frames
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4, 128, 128),
            dtype=np.float32
        )
        
        self.metadata = {"render_modes": None}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs, {}

    def render(self):
        pass

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

class DiscreteAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.ACTION_MAP = [
			("NO_OP",         [  0,   0, 0, 0, 0]),
			("STRAIGHT",      [  0,   0, 1, 0, 0]),
			("BRAKE",         [  0,   0, 0, 1, 0]),
			("BACK_UP",       [  0, -80, 0, 1, 0]),
			("SOFT_LEFT",     [-20,   0, 1, 0, 0]),
			("LEFT",          [-40,   0, 1, 0, 0]),
			("HARD_LEFT",     [-60,   0, 1, 0, 0]),
			("EXTREME_LEFT",  [-80,   0, 1, 0, 0]),
			("SOFT_RIGHT",    [ 20,   0, 1, 0, 0]),
			("RIGHT",         [ 40,   0, 1, 0, 0]),
			("HARD_RIGHT",    [ 60,   0, 1, 0, 0]),
			("EXTREME_RIGHT", [ 80,   0, 1, 0, 0]),
		]
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))

    def step(self, action):
        action = self._convert_action(action)
        return self.env.step(action)

    def _convert_action(self, action):
        action = self.ACTION_MAP[action][1]
        return action

def MarioKartEnv(training=True):
    env = EnvReceiver()
    if training:
        env = EarlyStop(env)
    env = FrameStack(env, 4)
    env = EnvWrapper(env)
    env = DiscreteAction(env)
    return env

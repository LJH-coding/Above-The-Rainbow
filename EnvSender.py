#!/bin/python2
import socket
import time
import gym, gym_mupen64plus
import errno
import pickle
import struct
from collections import deque
import numpy as np

class SkipWrapper(gym.Wrapper):
    def __init__(self, env, skip):
        super(SkipWrapper, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(k, shp[0], shp[1], shp[2])
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.concatenate(list(self.frames), axis=0).astype(np.uint8)


def try_connect():
    while True:
        try:
            sock.connect((host, port))
            print("Connected")
            break
        except Exception as e:
            time.sleep(1)

def is_socket_closed(sock):
    try:
        # this will try to read bytes without blocking and also without removing them from buffer (peek only)
        data = sock.recv(16, socket.MSG_DONTWAIT | socket.MSG_PEEK)
        if len(data) == 0:
            return True
    except socket.error as e:
        if e.errno == errno.EWOULDBLOCK:
            return False  # socket is open and reading from it would block
        elif e.errno == errno.ECONNRESET:
            return True  # socket was closed for some other reason
        else:
            return True
    return False

def send_data(data_to_send):
    data = pickle.dumps(data_to_send)
    length = struct.pack('>I', len(data))
    sock.sendall(length)
    sock.sendall(data)

def recv_data():
    data = sock.recv(4096)
    return data
    
host = '172.17.0.1'
port = 48763
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# env = gym.make('Mario-Kart-Luigi-Raceway-v0').env
env = gym.make('Mario-Kart-Kalimari-Desert-v0').env
# env = gym.make('Mario-Kart-Koopa-Troopa-Beach-v0').env
# env = gym.make('Mario-Kart-Toads-Turnpike-v0').env
# env = gym.make('Mario-Kart-Moo-Moo-Farm-v0').env
# env = gym.make('Mario-Kart-Rainbow-Road-v0').env

# ----------------------------------------------
# skip_frame = 4
# env = SkipWrapper(env, skip=skip_frame)
# stack_size = 4
# env = FrameStack(env, k=stack_size)
# ----------------------------------------------

while True:
    if is_socket_closed(sock):
        print("Waiting for connection")
        try:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        except:
            pass
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try_connect()

    try:
        data = recv_data()
        action = pickle.loads(data)
    except:
        print("Error loading data, maybe connection was closed")
        continue
    if action == "reset":
        obs = env.reset()
        send_data(obs)
    else:
        obs, rew, done, info = env.step(action)
        send_data((obs, rew, done, info))

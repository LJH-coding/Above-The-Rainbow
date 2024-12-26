# Above The Rainbow: Enhancing Rainbow DQN Performance via Multi-Order Gated Aggregation Network

<center>
<img src="https://hackmd.io/_uploads/BkLa9yiB1e.png" alt="Alt Text" style="width:50%; height:auto;">
</center>

## Acknowledgements

This project builds upon the Racing AI repository, which provided the base environment and training scripts for the **Soft Actor-Critic (SAC)** algorithm in Mario Kart 64. The original repository can be found here: [Original Repository](https://github.com/muyishen2040/Racing-AI-Reinforcement-Learning-for-Mario-Kart-64/tree/master).

## Introduction

We extended Racing AI repository repository by implementing additional algorithms, including **Proximal Policy Optimization (PPO)**, **Deep Q-Network (DQN)**, **Rainbow DQN**, and our proposed **Above The Rainbow (ATR)** method. 
Above The Rainbow (ATR) addresses this challenge by incorporating the **Multi-Order Gated Aggregation Network (MogaNet)** to enhance the visual feature extraction capabilities of Rainbow Deep Q-Network (DQN). This innovative framework delivers **state-of-the-art** performance in complex 3D environments.

## Setup

The easiest, cleanest, most consistent way to get up and running with this project is via [`Docker`](https://docs.docker.com/). These instructions will focus on that approach.

### Running with docker-compose

**Pre-requisites:**
- Docker & docker-compose (if you are using Compose plugin for docker, replace `docker-compose` with `docker compose` in the commands below).
- Ensure you have a copy of the ROMs you wish to use, and make sure it is placed inside the path under `gym_mupen64plus/ROMs`.

**Steps:**

0. Clone the repository and get into the root folder of the project.

1. Build the docker image with the following command:

    ```
    docker build -t bz/gym-mupen64plus:0.0.1 .
    ```

2. Please be noticed that in order to enable multiple instances of the environment, the original docker-compose file is separated into two parts - base file (docker-compose.yml) and override files (e.g. instance1.yml). The following command gives an example of instantiating an environment:

    ```bash
    docker-compose -p agent1 -f docker-compose.yml -f instance1.yml up --build -d
    ```

    This will start the following 4 containers:
    - `xvfbsrv` runs XVFB
    - `vncsrv` runs a VNC server connected to the Xvfb container
    - `agent` runs the example python script
    - `emulator` runs the mupen64plus emulator

    Note:
    - `-p` flag is the name of this environment instance
    - Before creating a new instance, be sure to create a override file to modify the port numbers (see `instance1.yml` for more details).
    - Make sure that the `docker-compose down` command given below also matches the file name of your instance and file names.

3. Under the root of the repository, there is a Python 3 file `SocketWrapper.py`. This file contains the wrapper for our RL training. We can first create a virtual environment for our project by:

    ```bash
    python -m venv RL_env
    ```

    Activate the environment:
    ```bash
    source RL_env/bin/activate
    ```

    Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```


4. Then you can use your favorite VNC client (e.g., [VNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)) to connect to `localhost` to watch the XVFB display in real-time. Note that running the VNC server and client can cause some performance overhead.

    For VSCode & TightVNC Users:
    - Forward the port 5901/5902 to the desired port on the local host.
    - Open TightVNC and connect to `localhost::desired_port_num`, e.g. `localhost::5901`.

5. To turn off the docker compose container (e.g. suppose we follow the naming criteria above `agent1` as the instance name and use `instance1.yml` for the override file), use the following command:

    ```bash
    docker-compose -p agent1 -f docker-compose.yml -f instance1.yml down
    ```

    Note:
    - To create another instance, you can create another tmux channel to run another with a different instance name and override file.

**Additional Notes:**

1. To view the status (output log) of a single compose, you can use the following command (suppose our instance name is `agent1`):

    ```bash
    docker-compose -p agent1 logs xvfbsrv
    docker-compose -p agent1 logs vncsrv
    docker-compose -p agent1 logs emulator
    docker-compose -p agent1 logs agent
    ```

## Features

![image](https://hackmd.io/_uploads/ByQNjkiS1g.png)

- **Advanced Visual Feature Extraction**:
Integrates **Multi-Order Gated Aggregation Network (MogaNet)** to enhance visual feature extraction, surpassing traditional Rainbow DQN.

- **Efficient and High-Performing**:
Outperforms Rainbow DQN in complex 3D environments with only **500k training timesteps**, achieving state-of-the-art results.

- **Human-Level Competence**:
Proven to exceed human performance in competitive scenarios, establishing itself as a cutting-edge reinforcement learning solution.

## Launch a training run

Above The Rainbow: 
```python
python train_atr.py
```

Rainbow DQN:

```python
python train_rainbow.py
```

Proximal Policy Optimization (PPO)

```python
python train_ppo.py
```

Deep Q Learning (DQN)

```python
python train_dqn.py
```

## Test Your Agent

```python
python test.py --model-path=./checkpoint/your_trained_model
```

## Cite

If you find this code or paper useful, please use the following reference:

```
@misc{above-the-rainbow,
  author = {Jia-Hua Lee and Li-Yu Chen and Ting-Hsuan Huang and Chih-Yun Liu and Jui-Hsuan Chang},
  title = {Above The Rainbow: Enhancing Rainbow DQN Performance via Multi-Order Gated Aggregation Network},
  year = {2024},
  institution = {Department of Computer Science, National Tsing Hua University, Hsinchu, Taiwan},
  url = {https://github.com/LJH-coding/Above-The-Rainbow},
  note = {Accessed: 2024-12-25}
}
```

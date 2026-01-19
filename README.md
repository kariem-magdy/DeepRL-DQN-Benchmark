<div align="center">

# 🎮 DeepRL DQN Benchmark

### Deep Q-Network vs Double DQN:  A Comprehensive PyTorch Comparison

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-0081A5?style=for-the-badge&logo=openaigym&logoColor=white)](https://gymnasium.farama.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A research-grade implementation of DQN and DDQN algorithms for classic control tasks*

[🚀 Quick Start](#-quick-start) • [📊 Results](#-observed-results) • [📖 Documentation](#-overview) • [🤝 Contributing](#-contributing)

</div>

---

## 📖 Overview

This repository provides a **research-grade implementation** of DQN and DDQN algorithms for discrete and discretized-continuous action spaces. The project demonstrates the complete RL training pipeline with modular, well-documented code.

### ✨ Key Highlights

| Feature | Description |
|---------|-------------|
| 🔄 **Algorithm Comparison** | Side-by-side comparison of DQN vs DDQN performance |
| 🎯 **Multiple Environments** | Support for `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`, `Pendulum-v1` |
| 💾 **Experience Replay** | Configurable buffer sizes (50K-100K transitions) |
| 🎯 **Target Networks** | Periodic updates for training stability |
| 🔍 **ε-Greedy Exploration** | Exponential decay with customizable parameters |
| 📊 **W&B Integration** | Optional Weights & Biases experiment tracking |
| 🎬 **Video Recording** | Automated recording of evaluation episodes |

---

## 🧠 Problem Statement

Reinforcement learning agents must learn optimal policies through trial-and-error interactions with an environment. Standard Q-learning suffers from **overestimation bias** when using function approximation. 

### Challenges Addressed

| Challenge | Description | Solution |
|-----------|-------------|----------|
| ⚠️ **Value Overestimation** | DQN uses the same network for action selection and evaluation | DDQN decouples these operations |
| 📉 **Training Instability** | Correlated sequential experiences destabilize training | Experience replay + target networks |
| 🎯 **Sparse Rewards** | Environments like MountainCar provide minimal feedback | Reward shaping techniques |
| 🔄 **Continuous Actions** | Q-learning requires discrete actions | Action discretization |

> 💡 **DDQN Solution**: Decouples action selection (online network) from action evaluation (target network), resulting in more stable and often superior learning.

---

## 🔄 RL System Pipeline

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     1. Environment Initialization        │
              │    (Gymnasium:  CartPole, Acrobot, etc.)  │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     2. State Observation (s_t)           │
              │    (Position, velocity, angles, etc.)    │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     3. Action Selection (ε-greedy)       │
              │    Explore:  random | Exploit: argmax Q   │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     4. Execute Action in Environment     │
              │    Receive:  reward (r), next state (s')  │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     5. Store Transition in Replay Buffer │
              │    (s, a, r, s', done) → Memory          │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     6. Sample Mini-Batch from Buffer     │
              │    Random sampling breaks correlation    │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     7. Compute TD Target                 │
              │    DQN:   y = r + γ * max_a' Q_target(s') │
              │    DDQN: y = r + γ * Q_target(s', argmax │
              │                      Q_online(s'))       │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     8. Update Online Q-Network           │
              │    Minimize loss: L = (Q(s,a) - y)²      │
              │    Backpropagation + Adam optimizer      │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │     9. Periodic Target Network Sync      │
              │    θ_target ← θ_online (every N steps)   │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │    10. Decay Exploration Rate (ε)        │
              │    ε = max(ε_min, ε * decay_rate)        │
              └──────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌───────────────────┐
                         │  Episode Complete │
                         │  Loop to Step 2   │
                         └───────────────────┘
                                    │
                                    ▼
              ┌──────────────────────────────────────────┐
              │    11. Evaluation & Video Recording      │
              │    Deterministic policy (ε = ε_min)      │
              └──────────────────────────────────────────┘
```

<details>
<summary>📚 <b>Click to expand:  Step-by-Step Explanation</b></summary>

#### Step 1: Environment Initialization
- **Purpose**: Create the simulation environment and extract state/action space specifications
- **Input**: Environment name (e.g., `CartPole-v1`, `MountainCar-v0`)
- **Output**: Environment object, state dimension, action dimension
- **Implementation**: Uses Gymnasium's `gym.make()` API

#### Step 2: State Observation
- **Purpose**:  Capture the current environment state as input to the Q-network

| Environment | State Dimensions | Components |
|-------------|------------------|------------|
| CartPole-v1 | 4 | Position, velocity, pole angle, angular velocity |
| Acrobot-v1 | 6 | cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2 |
| MountainCar-v0 | 2 | Position, velocity (normalized) |
| Pendulum-v1 | 3 | cos(θ), sin(θ), angular velocity |

#### Step 3: Action Selection (ε-Greedy Policy)
```python
if random() < ε:
    return random_action()  # Explore
else:
    return argmax(Q_network(state))  # Exploit
```

#### Step 4: Execute Action & Receive Feedback
- Interact with environment to observe consequences of actions
- **Reward Shaping** (MountainCar):
```python
reward = r_env + γ * α * pos_next - α * pos_cur + β * |velocity|
if goal_reached:
    reward += 100
```

#### Step 5: Experience Replay Memory
```python
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
```

#### Step 6-11: Training Loop
- Mini-batch sampling, TD target computation, network updates, and evaluation

</details>

---

## ⚔️ DQN vs DDQN Comparison

<table>
<tr>
<th>Aspect</th>
<th>🔵 DQN</th>
<th>🟢 DDQN</th>
</tr>
<tr>
<td><b>Target Computation</b></td>
<td><code>max Q_target(s', a')</code></td>
<td><code>Q_target(s', argmax Q_online(s', a'))</code></td>
</tr>
<tr>
<td><b>Overestimation</b></td>
<td>❌ High (same network selects & evaluates)</td>
<td>✅ Reduced (decoupled selection/evaluation)</td>
</tr>
<tr>
<td><b>Stability</b></td>
<td>⚠️ Moderate</td>
<td>✅ Improved</td>
</tr>
<tr>
<td><b>Sample Efficiency</b></td>
<td>Good</td>
<td>Often better on complex tasks</td>
</tr>
</table>

### 🧮 The Overestimation Problem

DQN's max operator introduces a positive bias: 
```
E[max(Q₁, Q₂, ..., Qₙ)] ≥ max(E[Q₁], E[Q₂], ..., E[Qₙ])
```

**DDQN's Solution:**
1. **Action Selection**: Use online network → `a* = argmax Q_online(s')`
2. **Action Evaluation**: Use target network → `Q_target(s', a*)`

---

## 🧪 Supported Environments

| Environment | Action Space | State Space | Goal | Max Steps |
|-------------|: ------------:|:-----------:|------|:---------:|
| 🎢 CartPole-v1 | Discrete(2) | Box(4) | Balance pole | 500 |
| 🤸 Acrobot-v1 | Discrete(3) | Box(6) | Swing up | 500 |
| 🏔️ MountainCar-v0 | Discrete(3) | Box(2) | Reach flag | 200 |
| 🔄 Pendulum-v1 | Box(1) → Discretized | Box(3) | Stay upright | 200 |

---

## 📊 Observed Results

| Environment | Agent | Mean Reward | Std Dev | Status |
|-------------|: -----:|:-----------:|:-------:|: ------:|
| CartPole-v1 | DQN | ~370 | ~50 | ✅ Solved |
| CartPole-v1 | DDQN | ~195 | ~10 | ✅ Solved |
| Acrobot-v1 | DQN | ~-110 | ~30 | ✅ Good |
| Acrobot-v1 | DDQN | ~-90 | ~20 | ✅ Better |
| Pendulum-v1 | DQN | ~-125 | ~100 | ⚠️ Variable |
| Pendulum-v1 | DDQN | ~-130 | ~100 | ⚠️ Variable |

> ⚠️ **Note**: MountainCar requires extended training (1000+ episodes) and reward shaping for success.

---

## 🚀 Quick Start

### Prerequisites

- 🐍 Python 3.10+ (3.11 recommended)
- 🎮 CUDA-capable GPU (optional, for faster training)

### 💻 Windows Installation

```batch
::  1) Clone the repository
git clone https://github.com/kariem-magdy/DeepRL-DQN-Benchmark.git
cd DeepRL-DQN-Benchmark

:: 2) Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 3) Install dependencies
pip install --upgrade pip
pip install torch numpy matplotlib gymnasium gymnasium[classic_control] wandb jupyter tqdm moviepy pygame

:: 4) Launch Jupyter Notebook
jupyter notebook
```

### 🐧 Linux/macOS Installation

```bash
# 1) Clone the repository
git clone https://github.com/kariem-magdy/DeepRL-DQN-Benchmark.git
cd DeepRL-DQN-Benchmark

# 2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install torch numpy matplotlib gymnasium "gymnasium[classic_control]" wandb jupyter tqdm moviepy pygame

# 4) Launch Jupyter Notebook
jupyter notebook
```

### 📊 Weights & Biases Setup (Optional)

```bash
wandb login
# Enter your API key when prompted
```

---

## 📝 Usage Examples

### Training Both Agents

```python
# Environments to train on
envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]

for env_name in envs:
    # Train DQN
    dqn_agent, dqn_rewards, dqn_meta = train_agent(env_name, "DQN", episodes=100)
    
    # Train DDQN
    ddqn_agent, ddqn_rewards, ddqn_meta = train_agent(env_name, "DDQN", episodes=100)
```

### Custom Configuration

```python
config = {
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "lr": 1e-3,
    "memory_size": 100000,
    "batch_size": 64
}

agent = DDQNAgent(state_size=4, action_size=2, config=config)
```

### Evaluating a Trained Agent

```python
# Load and evaluate
agent.load("models/CartPole-v1_DDQN.pth")
eval_rewards, videos = evaluate_and_record(
    agent,
    env_name="CartPole-v1",
    actions_list=[0, 1],
    agent_type="DDQN",
    episodes=10
)
print(f"Mean evaluation reward: {np.mean(eval_rewards):.2f}")
```

---

## 📁 Project Structure

```
DeepRL-DQN-Benchmark/
│
├── 📓 final_dqn_ddqn_record_last3. ipynb  # Main experiments notebook
│   ├── QNetwork class                    # Neural network architecture
│   ├── ReplayBuffer class                # Experience replay implementation
│   ├── DQNAgent class                    # DQN algorithm
│   ├── DDQNAgent class                   # DDQN algorithm
│   ├── train_agent()                     # Training loop
│   └── evaluate_and_record()             # Evaluation with video
│
├── 📓 updatedWithMountainCar. ipynb       # MountainCar-focused experiments
│
├── 📄 Assignment 2.pdf                   # Lab handout and references
│
├── 📂 models/                            # Saved model weights (generated)
│   ├── CartPole-v1_DQN.pth
│   ├── CartPole-v1_DDQN.pth
│   └── ... 
│
├── 📂 videos/                            # Evaluation recordings (generated)
│   └── {env_name}/{agent_type}/*. mp4
│
└── 📄 README.md                          # This file
```

---

## ⚙️ Hyperparameters

| Parameter | Default | MountainCar | Description |
|-----------|: -------:|:-----------:|-------------|
| `gamma` | 0.99 | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.01 | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | 0.9995 | Decay multiplier per step |
| `learning_rate` | 1e-3 | 5e-4 | Adam optimizer learning rate |
| `batch_size` | 64 | 64 | Training batch size |
| `memory_size` | 50,000 | 100,000 | Replay buffer capacity |
| `target_update` | Per episode | Every 500-1000 steps | Target network sync frequency |

---

## ⚠️ Limitations & Known Issues

### Current Limitations

| Limitation | Details |
|------------|---------|
| 🏔️ **MountainCar** | Requires reward shaping + extended training (1000+ episodes) |
| 🔄 **Continuous Actions** | Pendulum uses discretized actions, limiting precision |
| 🎛️ **Hyperparameter Sensitivity** | Different environments require tuned settings |
| 📊 **No Prioritized Replay** | Uniform sampling may be sample-inefficient |

### Known Issues

- ⚠️ Video recording may fail if `moviepy` or `pygame` are not properly installed
- ℹ️ W&B integration is optional; code handles its absence gracefully

---

## 🔮 Future Improvements

- [ ] 📊 **Prioritized Experience Replay (PER)**
- [ ] 🧠 **Dueling DQN Architecture**
- [ ] 🔊 **Noisy Networks** for exploration
- [ ] 🌈 **Rainbow DQN** combination
- [ ] 📋 Add proper `requirements.txt`
- [ ] 🔄 Implement soft target updates (Polyak averaging)
- [ ] 📈 Add TensorBoard logging
- [ ] ⚡ Multi-environment parallel training
- [ ] 🎯 Add A2C, PPO for comparison

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/improvement`)
3. 📝 Make changes with clear documentation
4. 🧪 Test on at least one environment (CartPole recommended)
5. 🚀 Submit a pull request

> 💬 For major changes, please open an issue first to discuss the approach.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

| Resource | Link |
|----------|------|
| 📘 PyTorch RL Tutorial | [pytorch.org](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) |
| 🎮 Gymnasium Documentation | [gymnasium. farama.org](https://gymnasium.farama.org) |
| 📊 Weights & Biases Guides | [docs.wandb.ai](https://docs.wandb.ai/guides/track/) |
| 📄 DQN Paper (Nature 2015) | [nature.com](https://www.nature.com/articles/nature14236) |
| 📄 DDQN Paper (AAAI 2016) | [arxiv.org](https://arxiv.org/abs/1509.06461) |

---

<div align="center">

**Made with ❤️ for the Deep RL community**

⭐ Star this repo if you find it helpful!

</div>

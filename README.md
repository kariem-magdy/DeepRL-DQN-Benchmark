# DQN vs DDQN: Deep Q-Learning Comparison in PyTorch

A comprehensive reinforcement learning project implementing and comparing Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) algorithms on classic control tasks using Gymnasium environments.

---

## Overview

This repository provides a research-grade implementation of DQN and DDQN algorithms for discrete and discretized-continuous action spaces. The project demonstrates the complete RL training pipeline—from environment interaction to policy evaluation—with support for multiple Gymnasium environments, experiment tracking via Weights & Biases, and video recording of trained agent behavior.

**Key Highlights:**
- Side-by-side comparison of DQN vs DDQN performance
- Support for discrete (`CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`) and continuous (`Pendulum-v1`) action spaces
- Experience replay with configurable buffer sizes
- Target network updates for training stability
- Epsilon-greedy exploration with exponential decay
- Optional Weights & Biases integration for experiment tracking
- Automated video recording of evaluation episodes

---

## Problem Statement

Reinforcement learning agents must learn optimal policies through trial-and-error interactions with an environment. Standard Q-learning suffers from **overestimation bias** when using function approximation (neural networks), which can lead to suboptimal policies and unstable training. 

This project addresses the following challenges:
1. **Value Overestimation**: DQN uses the same network for action selection and evaluation, causing systematic overestimation of Q-values
2. **Training Instability**:  Correlated sequential experiences and moving targets destabilize neural network training
3. **Sparse Rewards**: Environments like MountainCar provide minimal feedback, making exploration difficult
4. **Continuous Action Spaces**: Q-learning requires discrete actions, necessitating action discretization for continuous control tasks

DDQN mitigates overestimation by decoupling action selection (online network) from action evaluation (target network), resulting in more stable and often superior learning. 

---

## RL System Pipeline / End-to-End Workflow

### High-Level Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘
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
              │    Explore: random | Exploit: argmax Q   │
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
              ┌───────────────────────���──────────────────┐
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
              └───────────────────────��──────────────────┘
```

### Step-by-Step Explanation

#### Step 1: Environment Initialization
- **Purpose**: Create the simulation environment and extract state/action space specifications
- **Input**: Environment name (e.g., `CartPole-v1`, `MountainCar-v0`)
- **Output**: Environment object, state dimension, action dimension
- **Implementation Details**:
  - Uses Gymnasium's `gym.make()` API
  - For continuous action spaces (Pendulum), actions are discretized into 5-15 bins
  - Random seeds are set for reproducibility
- **Why Necessary**: Provides the interface for agent-environment interaction and defines the problem structure

#### Step 2: State Observation
- **Purpose**:  Capture the current environment state as input to the Q-network
- **Input**: Environment's internal state
- **Output**:  Observation vector (numpy array)
- **State Representations**:
  | Environment | State Dimensions | Components |
  |-------------|------------------|------------|
  | CartPole-v1 | 4 | Position, velocity, pole angle, angular velocity |
  | Acrobot-v1 | 6 | cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2 |
  | MountainCar-v0 | 2 | Position, velocity (normalized) |
  | Pendulum-v1 | 3 | cos(θ), sin(θ), angular velocity |
- **Why Necessary**: The state is the neural network's input; accurate representation enables learning of the value function

#### Step 3: Action Selection (ε-Greedy Policy)
- **Purpose**: Balance exploration of new actions vs exploitation of learned knowledge
- **Input**: Current state, exploration rate ε
- **Output**: Action index
- **Algorithm**:
  ```
  if random() < ε:
      return random_action()  # Explore
  else: 
      return argmax(Q_network(state))  # Exploit
  ```
- **Hyperparameters**:
  - `epsilon_start`: 1.0 (100% exploration initially)
  - `epsilon_min`: 0.01 (1% exploration minimum)
  - `epsilon_decay`: 0.995-0.9995 (environment-dependent)
- **Why Necessary**: Pure exploitation prevents discovery of better actions; pure exploration prevents convergence to optimal policy

#### Step 4: Execute Action & Receive Feedback
- **Purpose**:  Interact with environment to observe consequences of actions
- **Input**: Selected action
- **Output**: Reward signal, next state, termination flag
- **Reward Shaping** (MountainCar only):
  ```python
  reward = r_env + γ * α * pos_next - α * pos_cur + β * |velocity|
  if goal_reached:  reward += 100
  ```
- **Why Necessary**: The reward signal provides the learning signal; shaped rewards can accelerate learning in sparse-reward environments

#### Step 5: Experience Replay Memory
- **Purpose**: Store transitions for later training, breaking temporal correlations
- **Input**:  Transition tuple (s, a, r, s', done)
- **Output**: Updated replay buffer
- **Implementation**:
  ```python
  class ReplayBuffer:
      def __init__(self, capacity=50000-100000):
          self.buffer = deque(maxlen=capacity)
      def push(self, state, action, reward, next_state, done):
          self.buffer.append((state, action, reward, next_state, done))
  ```
- **Why Necessary**: 
  - Breaks correlation between consecutive samples (i.i.d. assumption for SGD)
  - Enables reuse of experiences (sample efficiency)
  - Smooths learning over many past behaviors

#### Step 6: Mini-Batch Sampling
- **Purpose**: Draw random samples for network training
- **Input**: Replay buffer, batch size
- **Output**:  Batch of transitions (states, actions, rewards, next_states, dones)
- **Batch Size**:  64 (default)
- **Why Necessary**:  Random sampling from memory provides uncorrelated training data essential for stable neural network updates

#### Step 7: Compute TD Target (Core DQN vs DDQN Difference)
- **Purpose**: Calculate the target value for the Bellman equation
- **Input**:  Batch of transitions, online network, target network
- **Output**: Target Q-values

**DQN Target Computation**:
```python
next_q_values = target_network(next_states).max(1)[0]
targets = rewards + gamma * next_q_values * (1 - dones)
```
- Uses target network for both action selection AND evaluation
- Prone to overestimation bias

**DDQN Target Computation**:
```python
next_actions = online_network(next_states).argmax(1)  # Select action
next_q_values = target_network(next_states).gather(1, next_actions)  # Evaluate
targets = rewards + gamma * next_q_values * (1 - dones)
```
- Uses online network for action selection
- Uses target network for action evaluation
- Decoupling reduces overestimation bias

- **Why Necessary**: The TD target provides the "ground truth" for supervised learning; proper target computation is critical for convergence

#### Step 8: Neural Network Update
- **Purpose**: Update Q-network weights to minimize prediction error
- **Input**: Predicted Q-values, target Q-values
- **Output**: Updated network weights
- **Network Architecture**:
  ```
  Input(state_dim) → FC(128, ReLU) → FC(128, ReLU) → FC(action_dim)
  ```
- **Loss Function**:  MSE Loss or Smooth L1 (Huber) Loss
- **Optimizer**: Adam with learning rate 1e-3 to 5e-4
- **Gradient Clipping**: Max norm 10. 0 (prevents exploding gradients)
- **Why Necessary**: This is where learning occurs; the network adjusts to better predict action values

#### Step 9: Target Network Update
- **Purpose**: Provide stable targets for TD learning
- **Input**: Online network weights
- **Output**: Updated target network weights
- **Update Strategy**:  Hard update (full weight copy) every N steps or per episode
- **Implementation**:
  ```python
  target_network.load_state_dict(q_network.state_dict())
  ```
- **Why Necessary**: Without a frozen target, both prediction and target change simultaneously, causing oscillation and divergence

#### Step 10: Exploration Decay
- **Purpose**: Gradually shift from exploration to exploitation
- **Input**:  Current epsilon, decay rate
- **Output**: New epsilon value
- **Decay Schedule**: Exponential decay after each training step
  ```python
  epsilon = max(epsilon_min, epsilon * epsilon_decay)
  ```
- **Why Necessary**: Early exploration finds good actions; later exploitation refines the policy

#### Step 11: Evaluation & Recording
- **Purpose**: Assess learned policy performance without exploration noise
- **Input**: Trained agent, evaluation episodes
- **Output**: Episode rewards, video recordings
- **Settings**:  `epsilon = epsilon_min` (near-deterministic)
- **Why Necessary**: Provides unbiased assessment of learned policy quality

---

## Key Features & Innovations

### DQN vs DDQN Comparison

| Aspect | DQN | DDQN |
|--------|-----|------|
| Target Computation | max Q_target(s', a') | Q_target(s', argmax Q_online(s', a')) |
| Overestimation | High (same network selects & evaluates) | Reduced (decoupled selection/evaluation) |
| Stability | Moderate | Improved |
| Sample Efficiency | Good | Often better on complex tasks |

### Addressing the Overestimation Problem

DQN's max operator introduces a positive bias: 
```
E[max(Q₁, Q₂, ..., Qₙ)] ≥ max(E[Q₁], E[Q₂], ..., E[Qₙ])
```

DDQN's solution:
1. **Action Selection**: Use online network → `a* = argmax Q_online(s')`
2. **Action Evaluation**: Use target network → `Q_target(s', a*)`

This decoupling means noise in online network's action selection doesn't inflate target values.

### Experience Replay Benefits

1. **Breaks Temporal Correlation**: Consecutive experiences are highly correlated; random sampling provides i.i.d. data
2. **Data Efficiency**: Each experience can be reused multiple times
3. **Stabilizes Learning**: Smooths out abrupt changes from individual experiences

### Target Network Importance

Without target networks: 
- Both Q(s,a) and target y change with each update
- Updates can be self-reinforcing, causing value explosion
- Policy oscillates between suboptimal behaviors

With target networks:
- Target y remains stable for N steps
- Provides fixed optimization objective
- Allows gradual policy improvement

---

## Models & RL Techniques

### Neural Network Architecture

```
QNetwork(
  (fc1): Linear(state_dim → 128)
  (relu1): ReLU
  (fc2): Linear(128 → 128)
  (relu2): ReLU
  (fc3): Linear(128 → action_dim)
)
```

**Design Choices**:
- Two hidden layers provide sufficient representational capacity for classic control
- 128 units per layer balances expressiveness and training speed
- ReLU activation prevents vanishing gradients
- No batch normalization (not needed for low-dimensional inputs)

### Hyperparameters

| Parameter | Default | MountainCar | Description |
|-----------|---------|-------------|-------------|
| `gamma` | 0.99 | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.01 | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | 0.9995 | Decay multiplier per step |
| `learning_rate` | 1e-3 | 5e-4 | Adam optimizer learning rate |
| `batch_size` | 64 | 64 | Training batch size |
| `memory_size` | 50,000-100,000 | 100,000 | Replay buffer capacity |
| `target_update` | Per episode | Every 500-1000 steps | Target network sync frequency |

---

## Environment & Simulation Setup

### Supported Environments

| Environment | Action Space | State Space | Goal | Max Steps |
|-------------|--------------|-------------|------|-----------|
| CartPole-v1 | Discrete(2) | Box(4) | Balance pole | 500 |
| Acrobot-v1 | Discrete(3) | Box(6) | Swing up | 500 |
| MountainCar-v0 | Discrete(3) | Box(2) | Reach flag | 200 |
| Pendulum-v1 | Box(1) → Discretized(5-15) | Box(3) | Stay upright | 200 |

### Action Space Handling

**Discrete Environments** (CartPole, Acrobot, MountainCar):
- Direct action indexing:  `action = actions_list[action_idx]`

**Continuous Environments** (Pendulum):
- Discretization via linear spacing: 
  ```python
  actions_list = np.linspace(action_low, action_high, resolution)
  # Example: [-2.0, -1.0, 0.0, 1.0, 2.0] for resolution=5
  ```

### State Normalization

For MountainCar, states are normalized to [0, 1]: 
```python
state_normalized = (state - obs_low) / (obs_high - obs_low)
```
This improves neural network learning by centering inputs. 

---

## Training & Evaluation Details

### Training Procedure

1. **Warmup Phase** (MountainCar:  3000 steps; Others: 2000 steps):
   - Collect random experiences to populate replay buffer
   - No network updates during warmup

2. **Training Loop**:
   - Episodes: 100-150 (simple tasks) or 1000 (MountainCar)
   - Update network after each environment step
   - Update target network periodically (per episode or every N steps)

3. **Checkpointing**:
   - Save best model (highest episode reward)
   - Save final model after training completes

### Evaluation Protocol

- **Episodes**: 100 deterministic evaluation runs
- **Metrics**: Mean reward, standard deviation, episode duration
- **Video Recording**: Last 3 episodes captured as MP4

### Observed Results

| Environment | Agent | Mean Reward | Std Dev |
|-------------|-------|-------------|---------|
| CartPole-v1 | DQN | ~370 | ~50 |
| CartPole-v1 | DDQN | ~195 | ~10 |
| Acrobot-v1 | DQN | ~-110 | ~30 |
| Acrobot-v1 | DDQN | ~-90 | ~20 |
| Pendulum-v1 | DQN | ~-125 | ~100 |
| Pendulum-v1 | DDQN | ~-130 | ~100 |

*Note: MountainCar requires extended training (1000+ episodes) and reward shaping for success.*

---

## Installation & Setup

### Prerequisites

- Python 3.10+ (3.11 recommended)
- CUDA-capable GPU (optional, for faster training)

### Quick Start (Windows)

```bat
::  1) Clone the repository
git clone https://github.com/kariem-magdy/DDQN-vs-DQN-pytorch.git
cd DDQN-vs-DQN-pytorch

:: 2) Create and activate virtual environment
python -m venv . venv
. venv\Scripts\activate

:: 3) Install dependencies
pip install --upgrade pip
pip install torch numpy matplotlib gymnasium gymnasium[classic_control] wandb jupyter tqdm moviepy pygame

:: 4) Launch Jupyter Notebook
jupyter notebook
```

### Quick Start (Linux/macOS)

```bash
# 1) Clone the repository
git clone https://github.com/kariem-magdy/DDQN-vs-DQN-pytorch.git
cd DDQN-vs-DQN-pytorch

# 2) Create and activate virtual environment
python3 -m venv .venv
source . venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install torch numpy matplotlib gymnasium "gymnasium[classic_control]" wandb jupyter tqdm moviepy pygame

# 4) Launch Jupyter Notebook
jupyter notebook
```

### Weights & Biases Setup (Optional)

```bash
wandb login
# Enter your API key when prompted
```

---

## Usage Examples

### Training Both Agents on All Environments

Open `final_dqn_ddqn_record_last3.ipynb` and run all cells sequentially: 

```python
# Environments to train on
envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]

for env_name in envs:
    # Train DQN
    dqn_agent, dqn_rewards, dqn_meta = train_agent(env_name, "DQN", episodes=100)
    
    # Train DDQN
    ddqn_agent, ddqn_rewards, ddqn_meta = train_agent(env_name, "DDQN", episodes=100)
```

### Custom Training Configuration

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
agent. load("models/CartPole-v1_DDQN. pth")
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

## Example Agent Behavior / Results

### Training Curves

The notebooks generate training reward plots comparing DQN and DDQN learning dynamics:

- **CartPole**:  Both agents typically solve the task within 30-50 episodes
- **Acrobot**:  Convergence around 50-80 episodes
- **MountainCar**: Requires reward shaping and 500+ episodes
- **Pendulum**: High variance due to discretized continuous actions

### Video Recordings

Evaluation videos are saved to: 
```
videos/
├── CartPole-v1/
│   ├── DQN/
│   │   └── rl-video-episode-*. mp4
│   └── DDQN/
│       └── rl-video-episode-*.mp4
├── Acrobot-v1/
│   └── ... 
└── ... 
```

---

## Project Structure

```
DDQN-vs-DQN-pytorch/
│
├── final_dqn_ddqn_record_last3.ipynb  # Main experiments notebook
│   ├── QNetwork class                 # Neural network architecture
│   ├── ReplayBuffer class             # Experience replay implementation
│   ├── DQNAgent class                 # DQN algorithm
│   ├── DDQNAgent class                # DDQN algorithm (inherits DQNAgent)
│   ├── train_agent()                  # Training loop
│   └── evaluate_and_record()          # Evaluation with video
│
├── updatedWithMountainCar.ipynb       # MountainCar-focused experiments
│   ├── Reward shaping implementation
│   ├── State normalization
│   ├── Extended warmup
│   └── Stability tests (100 episodes)
│
├── Assignment 2. pdf                   # Lab handout and references
│
├── models/                            # Saved model weights (generated)
│   ├── CartPole-v1_DQN.pth
│   ├── CartPole-v1_DDQN.pth
│   └── ... 
│
├── videos/                            # Evaluation recordings (generated)
│   └── {env_name}/{agent_type}/*. mp4
│
└── README.md                          # This file
```

---

## Limitations & Challenges

### Current Limitations

1. **MountainCar Performance**: Standard DQN/DDQN struggle with MountainCar's sparse rewards without: 
   - Reward shaping (position/velocity bonuses)
   - Extended training (1000+ episodes)
   - State normalization

2. **Continuous Action Discretization**: Pendulum uses discretized actions, which: 
   - Limits action precision
   - May miss optimal continuous policies
   - Higher resolution increases action space complexity

3. **Hyperparameter Sensitivity**: Different environments require tuned settings:
   - Epsilon decay rates vary significantly
   - Learning rates need adjustment per task

4. **No Prioritized Replay**:  Uniform sampling may be sample-inefficient for sparse reward tasks

### Known Issues

- Video recording may fail if `moviepy` or `pygame` are not properly installed
- WandB integration is optional; code handles its absence gracefully

---

## Future Improvements

### Algorithmic Enhancements

1. **Prioritized Experience Replay (PER)**
   - Sample important transitions more frequently
   - Particularly beneficial for sparse rewards

2. **Dueling DQN Architecture**
   - Separate state-value and advantage streams
   - `Q(s,a) = V(s) + A(s,a) - mean(A)`

3. **Noisy Networks**
   - Replace epsilon-greedy with parametric noise
   - More efficient exploration

4. **Rainbow DQN**
   - Combine:  DDQN + PER + Dueling + Noisy + N-step + Distributional

### Implementation Improvements

- Add proper `requirements.txt` with pinned versions
- Implement soft target updates (Polyak averaging)
- Add TensorBoard logging alternative to WandB
- Support multi-environment parallel training
- Add policy gradient methods (A2C, PPO) for comparison

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make changes in the Jupyter notebooks with clear documentation
4. Test on at least one environment (CartPole recommended for speed)
5. Submit a pull request with a description of changes

For major changes, please open an issue first to discuss the approach.

---

## License

This project is intended for educational purposes.  If you plan to publish or redistribute: 

1. Add a proper license file (MIT recommended)
2. Cite original DQN and DDQN papers: 
   - Mnih et al. (2015) "Human-level control through deep reinforcement learning"
   - Van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"

---

## References

- [PyTorch RL Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning. html)
- [Gymnasium Documentation](https://gymnasium.farama.org)
- [Weights & Biases Guides](https://docs.wandb.ai/guides/track/)
- [DQN Paper (Nature 2015)](https://www.nature.com/articles/nature14236)
- [DDQN Paper (AAAI 2016)](https://arxiv.org/abs/1509.06461)

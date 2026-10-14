# RL Project - Usage Guide

Complete guide to training, testing, and analyzing your Modular DQN Agent.

---

## 📁 File Structure

```
rl_project/
├── config.py              # All hyperparameters (edit this!)
├── train.py              # Main training script
├── test.py               # Testing/evaluation script
├── utils.py              # Plotting and analysis utilities
├── requirements.txt      # Package dependencies
│
├── environments/
│   ├── __init__.py
│   ├── world.py          # World environment class
│   └── helper_functions.py
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py     # ReplayBuffer
│   ├── networks.py       # Network class
│   ├── modular_dqn_agent.py
│   ├── dqn_agent.py
│   └── arbitrators/
│       ├── __init__.py
│       └── ceo_arbitrator.py
│
└── saved_agents/         # Trained agents saved here
    └── agent_gmQ_20240115_143022/
        ├── module_0.pt
        ├── module_1.pt
        ├── arbitrator.pt
        └── metadata.pt
```

---

## 🚀 Quick Start

### 1. Edit Configuration

```bash
# Edit config.py with your desired hyperparameters
nano config.py
```

Key parameters to adjust:

```python
# Training
NUM_FRAMES = 30000           # Total training steps
BATCH_SIZE = 512             # Mini-batch size
LEARNING_RATE = 1e-3         # Learning rate

# Agent
DECISION_PROCESS = 'gmQ'     # 'gmQ', 'CEO', 'variance', etc.
HIDDEN_SIZE = 512            # Network size

# Environment
ENV_SIZE = 10                # Grid size
NUM_STATS = 4                # Number of needs
SET_POINT = 5                # Homeostasis target
```

### 2. Train Agent

```bash
# Run training
python train.py

# Or in background
nohup python train.py > training.log &
```

**What happens:**
- Creates environment with config parameters
- Creates agent with N modules (one per stat)
- Trains for NUM_FRAMES steps
- Saves trained agent to `./saved_agents/agent_<name>_<timestamp>/`
- Prints progress and plots every 100 steps

**Output:**
```
🖥️  Using device: cuda
🌍 Creating environment...
   Environment created: 10×10 grid with 4 stats
🤖 Creating agent...
   Agent created with 4 modules
   Decision process: gmQ
📈 Starting training for 30,000 frames...
```

### 3. Test Trained Agent

```bash
# Test the latest trained agent
python test.py

# Or test a specific agent
python test.py ./saved_agents/agent_gmQ_20240115_143022/
```

**What happens:**
- Loads the trained agent
- Runs it for N episodes without learning
- Calculates performance metrics
- Shows detailed step-by-step info (optional)

**Output:**
```
📊 EVALUATION RESULTS
========================================================
Reward Statistics:
  Mean:                 -0.0234
  Std:                   0.0456
  Individual:           ['-0.0234', '-0.0189', '-0.0312']

Deviation Statistics:
  Mean (cumulative):   123.45
  Mean (max):            2.34
========================================================
```

---

## 📊 Analyzing Results

### Using Utils

```python
from agents import Modular_DQN_Agent
from utils import *

# Load your trained agent (from train.py)
agent = ...  # Your trained agent

# Plot training curves
plot_training_curves(agent, show=True)

# Plot value maps (if stored)
plot_value_maps(agent, show=True)

# Plot visited locations
plot_heatmap(agent, show=True)

# Analyze performance
analyze_agent_performance(agent, window=100)

# Save summary
save_agent_summary(agent, "./analysis/")
```

### Plot Types

**1. Training Curves** - 6-subplot figure showing:
- Rewards over time
- Modular rewards per step
- Internal stats over time
- Reward distribution histogram
- Stats distribution histogram
- Summary statistics

**2. Value Maps** - Heatmaps of learned value functions:
- One per module
- Shows which locations each module values
- Multiple snapshots during training

**3. Heatmap** - Where did the agent go?
- Shows all visited locations
- Darker = visited more often

### Analysis Metrics

```python
from utils import calculate_homeostasis_score, calculate_stability

# How well does agent maintain stats at set point?
score = calculate_homeostasis_score(agent.stats, SET_POINT)
# Lower is better (0 = perfect homeostasis)

# How stable are the stats? (low variance = stable)
stability = calculate_stability(agent.stats)
# Lower is better (0 = perfectly stable)
```

---

## 🔧 Common Tasks

### Change Decision Process

```python
# In config.py
DECISION_PROCESS = 'CEO'  # Options:
                          # 'gmQ' - sum all modules
                          # 'CEO' - learned arbitrator
                          # 'variance' - most confident
                          # 'voting' - democratic
                          # 'max_abs' - most passionate
                          # 'stat_weighted' - by urgency
                          # 'allQ' - absolute best
```

Then retrain:
```bash
python train.py
```

### Run Multiple Experiments

```python
# In config.py, use preset experiments:

# Option 1: Manual loop
experiments = ['gmQ', 'CEO', 'stat_weighted', 'variance']
for exp in experiments:
    DECISION_PROCESS = exp
    # train...

# Option 2: Use built-in presets
from config import load_experiment

load_experiment('ceo_arbitrator')  # Load preset
# Now DECISION_PROCESS is 'CEO' with optimal params
python train.py
```

### Visualize One Episode

```bash
# In test.py, change display to True:
results_display = evaluate_agent(
    agent,
    timesteps=50,
    display=True,  # ← Change to True
    num_episodes=1
)
```

Output shows each step:
```
Step   1
  Action:          right
  Reward:           0.0234
  Modular rewards: [0.05 -0.01 -0.02 0.01]
  Stats:           [0.29 0.48 0.75 0.95]
  Location:        [5 6]
  Dead:            False
```

---

## 📈 Comparing Agents

```python
from test import load_agent, evaluate_agent

# Load multiple agents
agent1, _ = load_agent('./saved_agents/agent_gmQ_...')
agent2, _ = load_agent('./saved_agents/agent_CEO_...')

# Test both
results1 = evaluate_agent(agent1, timesteps=100, num_episodes=3)
results2 = evaluate_agent(agent2, timesteps=100, num_episodes=3)

# Compare
print(f"gmQ mean reward:  {results1['mean_reward']:.4f}")
print(f"CEO mean reward:  {results2['mean_reward']:.4f}")

# Plot comparison
import matplotlib.pyplot as plt
plt.plot(results1['episode_rewards'], label='gmQ')
plt.plot(results2['episode_rewards'], label='CEO')
plt.legend()
plt.show()
```

---

## ⚙️ Advanced Configuration

### Environment Presets

```python
# Different difficulty levels

# Easy: Stationary environment, simple resources
STATIONARY = True
MODES = 3
RADIUS = 2

# Medium: Slow rotation
STATIONARY = False
ROTATION_SPEED = 0.1
MODES = 2

# Hard: Fast rotation, sparse resources
STATIONARY = False
ROTATION_SPEED = 3.14159  # Full rotation per episode
MODES = 1
RESOURCE_PERCENTILE = 95
```

### Reward Function Types

```python
REWARD_TYPE = 'HRRL'        # Homeostatic RL (expert choice)
# Other options:
# 'sq_dev' - squared deviation
# 'lin_dev' - linear deviation
# 'well_being' - binary (good/bad)
# 'min_sq' - minimize worst stat
```

### Blind Ablation Study

```python
# Test if agent actually uses vision or just memorizes

BLIND = True   # Mask out visual input
               # Agent only sees internal stats
               # If performance drops, agent was using vision
```

---

## 🐛 Troubleshooting

### Training is Slow

**Problem:** Training takes forever

**Solution:**
```python
# Reduce network size
HIDDEN_SIZE = 256  # Was 512

# Reduce update frequency
updates_per_step = 1  # Was 10 in agent code

# Reduce batch size
BATCH_SIZE = 256  # Was 512
```

### Agent Gets Stuck

**Problem:** Agent keeps going to same location

**Solution:**
```python
# Increase exploration
EPSILON_DECAY = 5e-5  # Explore longer
MIN_EPSILON = 0.05    # Keep some exploration

# Try different decision process
DECISION_PROCESS = 'stat_weighted'  # Prioritize needs
```

### High Variance in Results

**Problem:** Results differ a lot between runs

**Solution:**
```python
# More stable training
GAMMA = 0.99   # Plan further ahead
BATCH_SIZE = 1024  # Larger batches (if GPU allows)
LEARNING_RATE = 5e-4  # Smaller steps
```

### Out of Memory

**Problem:** CUDA out of memory

**Solution:**
```python
# Reduce memory
BATCH_SIZE = 256  # Was 512
MEMORY_SIZE = 50000  # Was 100000
HIDDEN_SIZE = 256  # Was 512

# Use CPU instead
# In train.py, set: device = torch.device("cpu")
```

---

## 📝 Logging and Saving

### Save Results Automatically

```python
# In train.py, after training:
from utils import save_agent_summary, plot_training_curves

# Save summary
save_agent_summary(agent, "./results/")

# Save plots
plot_training_curves(agent, save_dir="./results/", show=False)
plot_heatmap(agent, save_dir="./results/", show=False)
```

### Custom Logging

```python
# Add to train.py to log metrics every N steps

if frame_idx % 1000 == 0:
    avg_reward = np.mean(agent.rewards[-100:])
    avg_stat = np.mean(agent.stats[-100:])
    print(f"Step {frame_idx}: reward={avg_reward:.4f}, stat={avg_stat:.4f}")
    
    # Save to file
    with open('training_log.txt', 'a') as f:
        f.write(f"{frame_idx},{avg_reward},{avg_stat}\n")
```

---

## 🎯 Typical Workflow

```bash
# 1. Configure experiment
nano config.py

# 2. Train agent
python train.py
# Outputs: ./saved_agents/agent_gmQ_20240115_143022/

# 3. Test agent
python test.py

# 4. Analyze results
python -c "
from agents import Modular_DQN_Agent
from utils import plot_training_curves, analyze_agent_performance
# Load and analyze agent
"

# 5. Compare with other decision processes
# Edit config.py, change DECISION_PROCESS
nano config.py  # DECISION_PROCESS = 'CEO'
python train.py
python test.py

# 6. Generate report
python -c "
from test import load_agent, evaluate_agent
from utils import plot_training_curves
# Create comparison figures and save
"
```

---

## 📚 Example: Run Experiment Suite

```bash
# Run all decision processes and compare

for process in gmQ CEO variance voting stat_weighted; do
    sed -i "s/DECISION_PROCESS = .*/DECISION_PROCESS = '$process'/" config.py
    python train.py
done

# Then analyze results:
python test_all_agents.py  # (custom script to compare all)
```

---

## 💡 Tips & Tricks

1. **Use config presets for quick experiments:**
   ```python
   from config import load_experiment
   load_experiment('ceo_arbitrator')
   ```

2. **Monitor training in real-time:**
   ```bash
   tail -f training.log
   # Or use tensorboard (future addition)
   ```

3. **Save best agent automatically:**
   ```python
   if mean_reward > best_reward:
       save_agent(agent)
   ```

4. **Parallelize experiments:**
   ```bash
   python train.py & python train.py &  # 2 experiments
   ```

5. **Use GPU efficiently:**
   ```python
   # In train.py
   torch.cuda.empty_cache()  # Free memory
   agent.modules[0].to(device)  # Move to GPU
   ```

---

**For questions or issues, check the docstrings in each file!**

# Files Created - Summary

## 📋 Overview

Four main script files have been created for your RL project. Here's what each does:

---

## 1. 📝 `config.py` - Configuration File

**Purpose:** Central location for ALL hyperparameters

**What it contains:**
- Training parameters (learning rates, batch size, num_frames)
- Agent parameters (decision process, blind, skip_unpreferred)
- Environment parameters (grid size, num stats, reward type)
- Experiment presets (quick switching between setups)

**How to use:**
```python
# Edit this file to change experiments
# Then run: python train.py
# Train.py automatically imports all settings from here

DECISION_PROCESS = 'gmQ'  # Change this
NUM_FRAMES = 30000        # Change this
LEARNING_RATE = 1e-3      # Change this
```

**Why it's useful:**
- ✅ One place to manage all parameters
- ✅ Easy to run multiple experiments
- ✅ Don't have to edit train.py repeatedly
- ✅ Easy to compare different settings

**Size:** ~200 lines

---

## 2. 🚀 `train.py` - Main Training Script

**Purpose:** Run training from start to finish

**What it does:**
1. Loads all parameters from config.py
2. Creates the environment (World)
3. Creates the agent (Modular_DQN_Agent)
4. Runs training for NUM_FRAMES steps
5. Saves trained agent to disk
6. Prints summary statistics

**How to use:**
```bash
# Basic training
python train.py

# Background training
nohup python train.py > training.log &

# Monitor progress
tail -f training.log
```

**What you'll see:**
```
✅ Agent saved to: ./saved_agents/agent_gmQ_20240115_143022/
   Module 0: module_0.pt
   Module 1: module_1.pt
   Arbitrator: arbitrator.pt
   Metadata: metadata.pt
```

**Functions:**
- `create_environment()` - Make the World
- `create_agent(env)` - Make the Modular_DQN_Agent
- `save_agent(agent)` - Save trained weights
- `print_config()` - Display all settings
- `main()` - Main training loop

**Size:** ~350 lines

---

## 3. 🧪 `test.py` - Testing & Evaluation Script

**Purpose:** Load and evaluate trained agents

**What it does:**
1. Loads a saved agent from checkpoint
2. Runs it in the environment (no learning)
3. Calculates performance metrics
4. Optionally shows step-by-step behavior
5. Compares performance metrics

**How to use:**
```bash
# Test the most recent agent
python test.py

# Test a specific agent
python test.py ./saved_agents/agent_gmQ_20240115_143022/

# Test with detailed output
# (edit test.py, set display=True)
```

**Output example:**
```
EVALUATION RESULTS
========================================================
Reward Statistics:
  Mean:                 -0.0234
  Std:                   0.0456

Deviation Statistics:
  Mean (cumulative):   123.45
  Mean (max):            2.34
========================================================
```

**Functions:**
- `load_agent(agent_dir)` - Load saved agent
- `evaluate_agent(agent, timesteps, display)` - Run agent
- `print_results(results)` - Format output
- `find_latest_agent()` - Auto-find newest agent
- `main()` - Main testing loop

**Size:** ~360 lines

---

## 4. 📊 `utils.py` - Plotting & Analysis Utilities

**Purpose:** Visualize and analyze training results

**What it contains:**

### Plotting Functions:
- `plot_training_curves(agent)` - 6-subplot figure with rewards, stats, distributions
- `plot_value_maps(agent)` - Heatmaps of what each module learned
- `plot_heatmap(agent)` - Where did agent visit?

### Analysis Functions:
- `calculate_homeostasis_score(stats)` - How well maintained is homeostasis?
- `calculate_stability(stats)` - How stable are the stats?
- `analyze_agent_performance(agent)` - Print detailed metrics

### Logging Functions:
- `save_agent_summary(agent)` - Save text summary to file

**How to use:**
```python
from utils import plot_training_curves, analyze_agent_performance

# Inside your code (or in a Jupyter notebook):
plot_training_curves(agent, show=True)  # Display plot
analyze_agent_performance(agent)         # Print analysis
```

**Typical output:**
```
📊 AGENT PERFORMANCE ANALYSIS
========================================================
💰 REWARD ANALYSIS:
   Mean:                 -0.0234
   Std:                   0.0456
   Last 100 steps:       -0.0189

📈 STATS ANALYSIS:
   Homeostasis Score:     0.4523 (lower is better)
   Stability Score:       0.1234 (lower is better)
```

**Size:** ~450 lines

---

## 5. 📖 `USAGE_GUIDE.md` - Complete Usage Guide

**Purpose:** Documentation on how to use all the scripts

**What it contains:**
- Quick start guide (5 minutes to first training)
- Detailed explanations of each script
- Common tasks and how to do them
- Advanced configuration options
- Troubleshooting guide
- Typical workflow example
- Tips and tricks

**Size:** ~400 lines

---

## 🗂️ File Dependency Tree

```
train.py
  ├─ imports config.py          ← All hyperparameters
  ├─ imports World              ← Environment
  ├─ imports Modular_DQN_Agent  ← Agent
  └─ calls save_agent()         → Saves to ./saved_agents/

test.py
  ├─ imports config.py          ← Needed for environment
  ├─ imports World
  ├─ imports Modular_DQN_Agent
  └─ loads agents from ./saved_agents/

utils.py
  └─ imports matplotlib        ← For plotting
     Standalone - call functions from train.py or test.py
```

---

## 🚀 Quick Start

```bash
# 1. Edit configuration
nano config.py
# Change DECISION_PROCESS, NUM_FRAMES, etc.

# 2. Train agent
python train.py
# Automatically uses config.py parameters

# 3. Test agent
python test.py
# Loads most recent saved agent

# 4. Analyze results (in Python or Jupyter)
from agents import Modular_DQN_Agent
from utils import plot_training_curves, analyze_agent_performance

# (Manually load your agent and analyze)
plot_training_curves(agent, show=True)
analyze_agent_performance(agent)
```

---

## 📊 Typical File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| config.py | ~200 | Settings |
| train.py | ~350 | Training |
| test.py | ~360 | Testing |
| utils.py | ~450 | Analysis |
| **Total** | **~1360** | **Complete system** |

---

## ✅ What You Can Do Now

✅ **Train agents:** `python train.py`
✅ **Test agents:** `python test.py`
✅ **Change experiments:** Edit `config.py`
✅ **Run multiple experiments:** Change decision process in config
✅ **Analyze results:** Use `utils.py` functions
✅ **Compare agents:** Test different checkpoints
✅ **Visualize training:** Plot rewards, stats, heatmaps
✅ **Save results:** Automatic saving to `./saved_agents/`

---

## 🎯 Next Steps

1. **Look at config.py** - Understand available hyperparameters
2. **Run first training** - `python train.py`
3. **Test trained agent** - `python test.py`
4. **Analyze results** - Use utils.py functions
5. **Run experiments** - Change config.py, retrain, compare

**See USAGE_GUIDE.md for detailed instructions!**


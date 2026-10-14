# Modular DQN Agent - Complete Training System

A professional-grade reinforcement learning training system for the **Modular DQN Agent** in a grid-world environment.

## 🎯 What This Is

Complete training, testing, and analysis infrastructure for:
- **Modular Deep Q-Network (DQN)** agents
- **Grid-world environment** with multiple internal stats
- **Multiple decision processes** (gmQ, CEO, variance, voting, etc.)
- **Professional-grade analysis tools**

## 📦 What You Get

```
├── config.py              # Edit this! All hyperparameters
├── train.py              # python train.py → trains agent
├── test.py               # python test.py  → evaluates agent
├── utils.py              # Analysis & plotting functions
├── USAGE_GUIDE.md        # Complete documentation
├── FILES_CREATED.md      # File reference guide
└── README.md             # This file
```

## 🚀 Quick Start (3 Commands)

```bash
# 1. Train an agent
python train.py

# 2. Test the agent
python test.py

# 3. Analyze results (in Python)
python -c "from utils import plot_training_curves; plot_training_curves(agent)"
```

## 📝 Configuration

All hyperparameters in one place:

```python
# config.py
NUM_FRAMES = 30000              # Training steps
DECISION_PROCESS = 'gmQ'        # Decision method
LEARNING_RATE = 1e-3            # LR for modules
GAMMA = 0.5                     # Discount factor
NUM_STATS = 4                   # Number of needs
SET_POINT = 5                   # Homeostasis target
```

**To change experiments:** Edit `config.py`, then run `python train.py`

## 📊 Files Explained

### config.py (5.3 KB)
Central configuration file. Contains:
- Training hyperparameters (learning rate, batch size, frames)
- Agent hyperparameters (decision process, network size)
- Environment parameters (grid size, stats, rewards)
- Experiment presets for quick switching

**Use this for:** Changing experiments between runs

### train.py (7.0 KB)
Main training script. Does:
1. Loads parameters from config.py
2. Creates World environment
3. Creates Modular_DQN_Agent
4. Trains for NUM_FRAMES steps
5. Saves agent to `./saved_agents/`
6. Prints summary statistics

**Run with:** `python train.py`

### test.py (8.9 KB)
Testing and evaluation script. Does:
1. Loads saved agents
2. Runs in environment (no learning)
3. Calculates performance metrics
4. Shows step-by-step behavior (optional)
5. Compares results

**Run with:** `python test.py`

### utils.py (13 KB)
Analysis and plotting utilities. Includes:
- `plot_training_curves()` - 6-subplot figure
- `plot_value_maps()` - Module learned values
- `plot_heatmap()` - Agent locations visited
- `calculate_homeostasis_score()` - Quality metric
- `calculate_stability()` - Variance metric
- `analyze_agent_performance()` - Detailed analysis
- `save_agent_summary()` - Text report

**Use in:** Python/Jupyter notebooks

### USAGE_GUIDE.md (11 KB)
Comprehensive documentation:
- Quick start guide
- Detailed file explanations
- Common tasks & how-to
- Advanced configuration
- Troubleshooting
- Tips & tricks
- Workflow examples

**Start here:** If you're new to the system

### FILES_CREATED.md (6.8 KB)
File reference guide:
- File-by-file summary
- Dependency tree
- Feature list
- Learning path

**Use for:** Quick reference

## ✨ Key Features

### Easy Configuration
- Single `config.py` file to edit
- 45+ parameters manageable in one place
- Preset experiments for quick switching
- No need to edit other files

### Full Training Pipeline
- `train.py` handles everything
- Automatic checkpoint saving
- Progress monitoring with plots
- Summary statistics

### Rich Analysis Tools
- 6-subplot training curves
- Value function heatmaps
- Location heatmaps
- Homeostasis metrics
- Stability analysis
- Performance reports

### Professional Quality
- Comprehensive docstrings
- Clear error messages
- Progress indicators
- Detailed configuration
- Full documentation

## 📊 Typical Workflow

```bash
# 1. Edit configuration
nano config.py
# Change: DECISION_PROCESS = 'CEO'

# 2. Train agent
python train.py
# Outputs: ./saved_agents/agent_CEO_<timestamp>/

# 3. Test agent
python test.py
# Shows: Performance metrics

# 4. Compare with other method
nano config.py
# Change: DECISION_PROCESS = 'variance'
python train.py
python test.py
# Compare results!
```

## 🎯 Decision Processes

Available decision methods to choose from:

| Method | Description | When to use |
|--------|-------------|------------|
| `gmQ` | Sum all Q-values | Balanced approach |
| `CEO` | Learned arbitrator | Optimal decisions |
| `variance` | Most confident module | Trust expertise |
| `voting` | Democratic weighting | Fair consensus |
| `stat_weighted` | By urgency | Prioritize needs |
| `max_abs` | Most passionate | Strong commitment |
| `allQ` | Absolute best | Greedy selection |

## 📈 Example Output

### Training
```
🖥️  Using device: cuda
🌍 Creating environment...
🤖 Creating agent...
📈 Starting training for 30,000 frames...
✅ Training completed successfully!
💾 Agent saved to: ./saved_agents/agent_gmQ_20240115_143022/
```

### Testing
```
📊 EVALUATION RESULTS
========================================================
Reward Statistics:
  Mean:                 -0.0234
  Std:                   0.0456
Deviation Statistics:
  Mean (cumulative):   123.45
  Mean (max):            2.34
========================================================
```

## 🔧 Customization

### Change Training Length
```python
# config.py
NUM_FRAMES = 50000  # Was 30000
```

### Change Network Size
```python
# config.py
HIDDEN_SIZE = 256  # Was 512 (faster, smaller)
```

### Change Learning Rate
```python
# config.py
LEARNING_RATE = 5e-4  # More stable learning
```

### Add Custom Analysis
```python
# In Python/Jupyter
from utils import analyze_agent_performance
analyze_agent_performance(agent, window=100)
```

## 📚 Learning Path

1. **Read** USAGE_GUIDE.md (comprehensive)
2. **Edit** config.py (understand parameters)
3. **Run** `python train.py` (see training)
4. **Run** `python test.py` (see results)
5. **Analyze** with utils.py (understand performance)
6. **Experiment** with different settings

## ⚠️ Troubleshooting

### Training is slow?
```python
# config.py
HIDDEN_SIZE = 256      # Smaller network
BATCH_SIZE = 256       # Smaller batches
```

### Out of memory?
```python
# config.py
BATCH_SIZE = 256       # Reduce batch size
MEMORY_SIZE = 50000    # Reduce replay buffer
```

### High variance in results?
```python
# config.py
GAMMA = 0.99           # Plan further ahead
LEARNING_RATE = 5e-4   # Smaller steps
```

See USAGE_GUIDE.md for more troubleshooting tips.

## 📞 Getting Help

- **How to use files?** → Read USAGE_GUIDE.md
- **File references?** → Check FILES_CREATED.md
- **Code docstrings?** → `python -c "import train; help(train)"`
- **Specific errors?** → Check error message + USAGE_GUIDE.md

## ✅ What's Included

- ✅ Complete training pipeline
- ✅ Evaluation and testing tools
- ✅ Analysis and visualization
- ✅ Comprehensive documentation
- ✅ Example configurations
- ✅ Error handling
- ✅ Progress monitoring
- ✅ Automatic checkpointing

## 🎁 Bonus Features

- Automatic timestamped checkpoints
- Metadata tracking
- Experiment presets
- Multi-agent comparison
- Homeostasis metrics
- Stability analysis
- Value function visualization
- Comprehensive logging

## 📄 File Sizes

| File | Size | Purpose |
|------|------|---------|
| config.py | 5.3 KB | Settings |
| train.py | 7.0 KB | Training |
| test.py | 8.9 KB | Testing |
| utils.py | 13 KB | Analysis |
| USAGE_GUIDE.md | 11 KB | Documentation |
| FILES_CREATED.md | 6.8 KB | Reference |
| **Total** | **~52 KB** | **Complete system** |

## 🎓 Example: Full Experiment

```bash
# Train gmQ agent
python train.py  # Uses DECISION_PROCESS='gmQ' from config

# Test it
python test.py
# Results: ./saved_agents/agent_gmQ_<time>/

# Change to CEO
sed -i "s/DECISION_PROCESS = .*/DECISION_PROCESS = 'CEO'/" config.py

# Train CEO agent
python train.py
# Results: ./saved_agents/agent_CEO_<time>/

# Test it
python test.py

# Analyze both (in Python)
python << 'ANALYSIS'
from test import load_agent, evaluate_agent
from utils import plot_training_curves

agent1, _ = load_agent('./saved_agents/agent_gmQ_...')
agent2, _ = load_agent('./saved_agents/agent_CEO_...')

plot_training_curves(agent1)
plot_training_curves(agent2)

print(f"gmQ results: {agent1...}")
print(f"CEO results: {agent2...}")
ANALYSIS
```

## 🚀 Next Steps

1. Read USAGE_GUIDE.md
2. Edit config.py
3. Run `python train.py`
4. Run `python test.py`
5. Use utils.py for analysis
6. Run more experiments!

---

**Created:** Complete RL training system for Modular DQN agents

**Start with:** `python train.py`

**Good luck with your research!** 🎉

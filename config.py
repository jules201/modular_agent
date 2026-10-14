"""
Configuration file for the Modular DQN Agent training.

All hyperparameters are defined here for easy experimentation.
Change values here instead of editing train.py or agent code.
"""

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

NUM_FRAMES = 30000             # Total training steps change to 3000 if gpu
BATCH_SIZE = 512                # Mini-batch size for training
MEMORY_SIZE = 100000            # Replay buffer capacity
TARGET_UPDATE = 200             # Update target network every N steps
LEARNING_RATE = 1e-3            # Learning rate for module networks
ARBITRATOR_LR = 1e-3            # Learning rate for arbitrator
GAMMA = 0.5                     # Discount factor (0=immediate reward, 1=future)
EPSILON_DECAY = 1e-3            # How fast to reduce exploration 1e-3 - 1000 annealing steps
MAX_EPSILON = 1.0               # Start with full exploration
MIN_EPSILON = 0.01              # Minimum exploration rate
PLOTTING_INTERVAL = 10000       # Plot every N frames

# ============================================================================
# AGENT HYPERPARAMETERS
# ============================================================================

HIDDEN_SIZE = 512               # Hidden layer size in networks
DECISION_PROCESS = 'gmQ'        # How to pick modules: 'gmQ', 'CEO', 'variance', 
                                # 'voting', 'max_abs', 'allQ', 'stat_weighted'
BLIND = True                    # Use attention mask (ablation study)
SKIP_UNPREFERRED = False        # Only train modules that wanted the action
SWITCH_EPISODE = 1500000        # Frame to trigger environment change (usually > NUM_FRAMES)

# ============================================================================
# ENVIRONMENT HYPERPARAMETERS
# ============================================================================

# Grid & Vision
ENV_SIZE = 10                  # Grid size (10x10)
FIELD_OF_VIEW = 1              # How far agent can see (1 = 3x3 view)
BOUNDS = 3                      # Resource distribution bounds

# Resources
MODES = 1                       # Number of resource patches per stat
RADIUS = 3.5                    # Distance of resource patches from center
VARIANCE = 1                    # Gaussian variance of resources
CIRCLE = True                   # Use circular arrangement (True) or random (False)

# Training
STATIONARY = False              # Does world change? (False = rotates)
POISSON_SHUFFLE = False         # Random resource reshuffling
ROTATION_SPEED = 3.14159        # How fast world rotates (pi radians)
EP_LENGTH = 50000                 # Steps per episode
RESOURCE_PERCENTILE = 95        # Threshold for resource effect

# Stats & Rewards
NUM_STATS = 4                   # Number of internal needs (hunger, thirst, etc.)
STAT_TYPES = [0, 0, 0, 0]      # Type per stat: 0=deplete, 1=capped
STAT_DECREASE_PER_STEP = [0.01, 0.01, 0.01, 0.01]  # Depletion rate per stat
STAT_INCREASE_MULTIPLIER = 1    # How much resources increase stats
INITIAL_STATS = [0.5, 0.5]     # Initial stat range [min, max]
SET_POINT = 5                   # Target stat value (homeostasis goal)

# Reward Function
REWARD_TYPE = 'HRRL'           # 'HRRL', 'sq_dev', 'lin_dev', 'well_being', 'min_sq'
MODULE_REWARD_TYPE = 'HRRL'    # Separate rewards per module
PQ = [4, 2]                    # Exponents for HRRL reward function
REWARD_CLIP = 500              # Clip rewards to ±this value
REWARD_SCALING = 1             # Scale total reward
MOD_REWARD_SCALING = 1         # Scale per-module reward
MOD_REWARD_BIAS = 0            # Bias for per-module reward

# ============================================================================
# EXPERIMENT VARIANTS
# ============================================================================

# You can define different experiment presets here:

EXPERIMENTS = {
    'default': {
        'LEARNING_RATE': 1e-3,
        'GAMMA': 0.5,
        'EPSILON_DECAY': 1e-4,
        'DECISION_PROCESS': 'gmQ',
        'BLIND': True,
    },
    
    'fast_learning': {
        'LEARNING_RATE': 5e-3,
        'GAMMA': 0.99,
        'EPSILON_DECAY': 5e-5,
        'DECISION_PROCESS': 'gmQ',
        'BLIND': False,
    },
    
    'ceo_arbitrator': {
        'LEARNING_RATE': 1e-3,
        'GAMMA': 0.5,
        'EPSILON_DECAY': 1e-4,
        'DECISION_PROCESS': 'CEO',
        'BLIND': False,
    },
    
    'stat_weighted': {
        'LEARNING_RATE': 1e-3,
        'GAMMA': 0.5,
        'EPSILON_DECAY': 1e-4,
        'DECISION_PROCESS': 'stat_weighted',
        'BLIND': True,
    },
}

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def load_experiment(experiment_name='default'):
    """
    Load an experiment preset and update global config.
    
    Usage:
        from config import load_experiment
        load_experiment('ceo_arbitrator')
    """
    global LEARNING_RATE, GAMMA, EPSILON_DECAY, DECISION_PROCESS, BLIND
    
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    params = EXPERIMENTS[experiment_name]
    for key, value in params.items():
        globals()[key] = value
    
    print(f"✅ Loaded experiment: {experiment_name}")

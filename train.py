"""
Training script for Modular DQN Agent in the Resource World environment.

This script:
1. Loads hyperparameters from config.py
2. Creates the environment
3. Creates the agent
4. Trains the agent
5. Saves the trained agent
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from utils import plot_training_curves

# Import configuration
from config import (
    NUM_FRAMES,
    BATCH_SIZE,
    MEMORY_SIZE,
    TARGET_UPDATE,
    LEARNING_RATE,
    ARBITRATOR_LR,
    GAMMA,
    EPSILON_DECAY,
    MAX_EPSILON,
    MIN_EPSILON,
    HIDDEN_SIZE,
    DECISION_PROCESS,
    BLIND,
    SKIP_UNPREFERRED,
    SWITCH_EPISODE,
    PLOTTING_INTERVAL,
    # Environment params
    ENV_SIZE,
    FIELD_OF_VIEW,
    NUM_STATS,
    STAT_TYPES,
    BOUNDS,
    MODES,
    RADIUS,
    VARIANCE,
    CIRCLE,
    STATIONARY,
    #ROTATION_SPEED,
    EP_LENGTH,
    RESOURCE_PERCENTILE,
    REWARD_TYPE,
    MODULE_REWARD_TYPE,
    STAT_DECREASE_PER_STEP,
    STAT_INCREASE_MULTIPLIER,
    INITIAL_STATS,
    SET_POINT,
    PQ,
    REWARD_CLIP,
    REWARD_SCALING,
    MOD_REWARD_SCALING,
    MOD_REWARD_BIAS,
)

# Import environment and agent
from environments import World
from agents import Modular_DQN_Agent


def create_environment():
    """Create and return the World environment with configured parameters."""
    env = World(
        n_actions=4,
        size=ENV_SIZE,
        bounds=BOUNDS,
        modes=MODES,
        field_of_view=FIELD_OF_VIEW,
        stat_types=STAT_TYPES,
        variance=VARIANCE,
        radius=RADIUS,
        offset=0.5,
        circle=CIRCLE,
        stationary=STATIONARY,
        poisson_shuffle=False,
        #rotation_speed=ROTATION_SPEED,
        ep_length=EP_LENGTH,
        resource_percentile=RESOURCE_PERCENTILE,
        reward_type=REWARD_TYPE,
        module_reward_type=MODULE_REWARD_TYPE,
        stat_decrease_per_step=STAT_DECREASE_PER_STEP,
        stat_increase_multiplier=STAT_INCREASE_MULTIPLIER,
        initial_stats=INITIAL_STATS,
        set_point=SET_POINT,
        pq=PQ,
        reward_clip=REWARD_CLIP,
        reward_scaling=REWARD_SCALING,
        mod_reward_scaling=MOD_REWARD_SCALING,
        mod_reward_bias=MOD_REWARD_BIAS,
    )
    return env


def create_agent(env):
    """Create and return the Modular DQN Agent with configured parameters."""
    agent = Modular_DQN_Agent(
        env=env,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        target_update=TARGET_UPDATE,
        epsilon_decay=EPSILON_DECAY,
        max_epsilon=MAX_EPSILON,
        min_epsilon=MIN_EPSILON,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        arbitrator_lr=ARBITRATOR_LR,
        hidden_size=HIDDEN_SIZE,
        decision_process=DECISION_PROCESS,
        blind=BLIND,
        skip_unpreferred=SKIP_UNPREFERRED,
        switch_episode=SWITCH_EPISODE,
    )
    return agent


def save_agent(agent, save_dir="./saved_agents"):
    """Save trained agent modules and metadata."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_dir = os.path.join(save_dir, f"agent_{DECISION_PROCESS}_{timestamp}")
    os.makedirs(agent_dir, exist_ok=True)
    
    # Save each module
    for i, module in enumerate(agent.modules):
        torch.save(
            module.state_dict(),
            os.path.join(agent_dir, f"module_{i}.pt")
        )
    
    # Save arbitrator
    torch.save(
        agent.arbitrator.state_dict(),
        os.path.join(agent_dir, "arbitrator.pt")
    )
    
    # Save statistics and metadata
    metadata = {
        'num_frames': NUM_FRAMES,
        'decision_process': DECISION_PROCESS,
        'num_modules': agent.numModules,
        'action_dim': agent.action_dim,
        'hidden_size': HIDDEN_SIZE,
        'stats': agent.stats,
        'rewards': agent.rewards,
        'mod_rewards': agent.mod_rewards,
    }
    
    torch.save(metadata, os.path.join(agent_dir, "metadata.pt"))
    
    print(f"\n✅ Agent saved to: {agent_dir}")
    return agent_dir


def print_config():
    """Print all configuration parameters."""
    print("\n" + "="*60)
    print("🔧 TRAINING CONFIGURATION")
    print("="*60)
    
    print("\n📊 TRAINING PARAMS:")
    print(f"  Frames:           {NUM_FRAMES:,}")
    print(f"  Batch Size:       {BATCH_SIZE}")
    print(f"  Memory Size:      {MEMORY_SIZE:,}")
    print(f"  Learning Rate:    {LEARNING_RATE}")
    print(f"  Gamma:            {GAMMA}")
    print(f"  Epsilon Decay:    {EPSILON_DECAY}")
    
    print("\n🌍 ENVIRONMENT PARAMS:")
    print(f"  Grid Size:        {ENV_SIZE}×{ENV_SIZE}")
    print(f"  Field of View:    {FIELD_OF_VIEW}")
    print(f"  Num Stats:        {NUM_STATS}")
    print(f"  Episode Length:   {EP_LENGTH}")
    print(f"  Set Point:        {SET_POINT}")
    
    print("\n🤖 AGENT PARAMS:")
    print(f"  Decision Process: {DECISION_PROCESS}")
    print(f"  Hidden Size:      {HIDDEN_SIZE}")
    print(f"  Blind:            {BLIND}")
    print(f"  Skip Unpreferred: {SKIP_UNPREFERRED}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main training function."""
    
    # Print configuration
    print_config()
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Create environment
    print("\n🌍 Creating environment...")
    env = create_environment()
    print(f"   Environment created: {ENV_SIZE}×{ENV_SIZE} grid with {NUM_STATS} stats")
    
    # Create agent
    print("\n🤖 Creating agent...")
    agent = create_agent(env)
    print(f"   Agent created with {agent.numModules} modules")
    print(f"   Decision process: {DECISION_PROCESS}")
    
    # Configure agent settings
    agent.plotting = True
    agent.plotting_interval = PLOTTING_INTERVAL
    agent.store_value_maps = False
    agent.updates_per_step = 1
    
    # Train agent
    print(f"\n📈 Starting training for {NUM_FRAMES:,} frames...")
    print("-" * 60)
    
    try:
        agent.train(num_frames=NUM_FRAMES)
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
    
    # Save agent
    print("\n💾 Saving trained agent...")
    agent_path = save_agent(agent)
    
    # Print final statistics
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"Total frames trained:        {NUM_FRAMES:,}")
    print(f"Final reward mean:           {np.mean(agent.rewards[-100:]):.4f}")
    print(f"Cumulative deviation:        {agent.cumulative_deviation:.4f}")
    print(f"Maximum deviation:           {agent.maximum_deviation:.4f}")
    print(f"Final stats:                 {np.round(agent.env.stats, 2)}")
    print(f"Agent saved to:              {agent_path}")
    print("="*60 + "\n")

    plot_training_curves(agent, save_dir="./results/", show=True)
if __name__ == "__main__":
    main()

"""
Testing script for evaluating trained Modular DQN Agents.

This script:
1. Loads a saved agent from a checkpoint
2. Runs it in the environment without learning
3. Evaluates performance metrics
4. Optionally displays agent behavior
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Import environment and agent
from environments import World
from agents import Modular_DQN_Agent
from config import *


def load_agent(agent_dir):
    """
    Load a saved agent from a directory.
    
    Args:
        agent_dir (str): Path to directory containing saved agent files
        
    Returns:
        agent (Modular_DQN_Agent): Loaded agent
        metadata (dict): Agent metadata
    """
    
    if not os.path.exists(agent_dir):
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")
    
    print(f"\n📂 Loading agent from: {agent_dir}")
    
    # Load metadata
    metadata_path = os.path.join(agent_dir, "metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)
        print(f"✅ Metadata loaded")
        print(f"   Decision Process: {metadata.get('decision_process', 'unknown')}")
        print(f"   Num Modules:      {metadata.get('num_modules', 'unknown')}")
    else:
        metadata = {}
        print(f"⚠️  No metadata found")
    
    # Create environment
    env = World(
        n_actions=4,
        size=ENV_SIZE,
        bounds=BOUNDS,
        modes=MODES,
        field_of_view=FIELD_OF_VIEW,
        stat_types=STAT_TYPES,
        variance=VARIANCE,
        radius=RADIUS,
        circle=CIRCLE,
        stationary=STATIONARY,
        ep_length=EP_LENGTH,
        resource_percentile=RESOURCE_PERCENTILE,
        reward_type=REWARD_TYPE,
        module_reward_type=MODULE_REWARD_TYPE,
        stat_decrease_per_step=STAT_DECREASE_PER_STEP,
        stat_increase_multiplier=STAT_INCREASE_MULTIPLIER,
        initial_stats=INITIAL_STATS,
        set_point=SET_POINT,
        pq=PQ,
    )
    
    # Create agent
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
        decision_process=metadata.get('decision_process', DECISION_PROCESS),
        blind=BLIND,
        skip_unpreferred=SKIP_UNPREFERRED,
    )
    
    # Load module weights
    for i in range(agent.numModules):
        module_path = os.path.join(agent_dir, f"module_{i}.pt")
        if os.path.exists(module_path):
            agent.modules[i].load_state_dict(torch.load(module_path))
            agent.targets[i].load_state_dict(torch.load(module_path))
        else:
            print(f"⚠️  Module {i} weights not found")
    
    # Load arbitrator weights
    arb_path = os.path.join(agent_dir, "arbitrator.pt")
    if os.path.exists(arb_path):
        agent.arbitrator.load_state_dict(torch.load(arb_path))
    
    print(f"✅ Agent loaded successfully")
    return agent, metadata


def evaluate_agent(agent, timesteps=100, display=False, num_episodes=3):
    """
    Evaluate agent performance over multiple episodes.
    
    Args:
        agent (Modular_DQN_Agent): Agent to evaluate
        timesteps (int): Steps per episode
        display (bool): Show detailed info during evaluation
        num_episodes (int): Number of episodes to run
        
    Returns:
        results (dict): Evaluation metrics
    """
    
    print(f"\n🧪 EVALUATING AGENT")
    print(f"   Episodes:        {num_episodes}")
    print(f"   Steps per ep:    {timesteps}")
    print(f"   Display:         {display}")
    print("-" * 60)
    
    agent.is_test = True
    
    episode_rewards = []
    episode_deviations = []
    episode_max_deviations = []
    
    for episode in range(num_episodes):
        print(f"\n📊 Episode {episode + 1}/{num_episodes}")
        
        score = 0
        cumulative_deviation = 0
        maximum_deviation = 0
        t = 0
        
        state = agent.env.reset().unsqueeze(0)
        done = False
        
        while t < timesteps and not done:
            t += 1
            
            action = agent.select_action(state, t)
            next_state, reward, done, info = agent.step(action)
            
            if display:
                action_names = ['down', 'up', 'right', 'left']
                print(f"\n  Step {t:3d}")
                print(f"    Action:          {action_names[action]}")
                print(f"    Reward:          {reward:7.4f}")
                print(f"    Modular rewards: {np.round(info, 3)}")
                print(f"    Stats:           {np.round(agent.env.stats, 2)}")
                print(f"    Location:        {agent.env.loc}")
                print(f"    Dead:            {agent.env.dead}")
            
            state = next_state.unsqueeze(0)
            score += reward
            
            # Calculate deviation metrics
            cumulative_deviation += sum([abs(stat - SET_POINT) for stat in agent.env.stats])
            for stat in agent.env.stats:
                if abs(stat - SET_POINT) > maximum_deviation:
                    maximum_deviation = abs(stat - SET_POINT)
        
        episode_rewards.append(score)
        episode_deviations.append(cumulative_deviation)
        episode_max_deviations.append(maximum_deviation)
        
        print(f"\n  ✅ Episode complete")
        print(f"     Total reward:       {score:.4f}")
        print(f"     Cumulative dev:     {cumulative_deviation:.4f}")
        print(f"     Max deviation:      {maximum_deviation:.4f}")
        print(f"     Final stats:        {np.round(agent.env.stats, 2)}")
    
    # Aggregate results
    results = {
        'episode_rewards': episode_rewards,
        'episode_deviations': episode_deviations,
        'episode_max_deviations': episode_max_deviations,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_deviation': np.mean(episode_deviations),
        'mean_max_deviation': np.mean(episode_max_deviations),
        'num_episodes': num_episodes,
    }
    
    return results


def print_results(results):
    """Print evaluation results in a nice format."""
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nReward Statistics:")
    print(f"  Mean:            {results['mean_reward']:8.4f}")
    print(f"  Std:             {results['std_reward']:8.4f}")
    print(f"  Individual:      {[f'{r:.4f}' for r in results['episode_rewards']]}")
    
    print(f"\nDeviation Statistics:")
    print(f"  Mean (cumulative): {results['mean_deviation']:8.4f}")
    print(f"  Mean (max):        {results['mean_max_deviation']:8.4f}")
    
    print("\n" + "="*60 + "\n")


def find_latest_agent(save_dir="./saved_agents"):
    """Find the most recently saved agent."""
    
    if not os.path.exists(save_dir):
        print(f"❌ Save directory not found: {save_dir}")
        return None
    
    # Find all agent directories
    agent_dirs = []
    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        if os.path.isdir(item_path) and item.startswith("agent_"):
            agent_dirs.append((item_path, item))
    
    if not agent_dirs:
        print(f"❌ No agents found in {save_dir}")
        return None
    
    # Sort by name (timestamp in name)
    agent_dirs.sort(key=lambda x: x[1], reverse=True)
    
    latest_path = agent_dirs[0][0]
    latest_name = agent_dirs[0][1]
    
    print(f"✅ Found {len(agent_dirs)} agent(s)")
    print(f"   Latest: {latest_name}")
    
    return latest_path


def main():
    """Main testing function."""
    
    print("\n" + "="*60)
    print("🤖 MODULAR DQN AGENT - TESTING")
    print("="*60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        agent_dir = sys.argv[1]
    else:
        # Try to find latest agent
        agent_dir = find_latest_agent()
        if not agent_dir:
            print("\n💡 Usage: python test.py <agent_directory>")
            return
    
    # Load agent
    try:
        agent, metadata = load_agent(agent_dir)
    except Exception as e:
        print(f"\n❌ Failed to load agent: {e}")
        return
    
    # Run evaluation
    try:
        results = evaluate_agent(
            agent,
            timesteps=100,
            display=False,  # Set to True to see detailed step info
            num_episodes=3
        )
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        raise
    
    # Print results
    print_results(results)
    
    # Optional: Visualize one episode with display
    print("\n🎬 Running one episode with display...")
    print("-" * 60)
    results_display = evaluate_agent(
        agent,
        timesteps=50,
        display=True,  # Show every step
        num_episodes=1
    )


if __name__ == "__main__":
    main()

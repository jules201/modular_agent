"""
Analyze a trained agent by loading it and running evaluation episodes.

Usage:
    python analyze.py
    > Enter agent path: ./saved_agents/agent_stat_weighted_20261007_140843/
"""

from test import load_agent, evaluate_agent
from utils import save_agent_summary, plot_training_curves
import numpy as np


if __name__ == "__main__":
    agent_path = input("Enter agent path: ").strip()
    
    # Load agent
    print(f"\n📂 Loading agent from: {agent_path}")
    agent, metadata = load_agent(agent_path)
    
    # Evaluate agent
    print("\n🧪 EVALUATING AGENT")
    results = evaluate_agent(agent, timesteps=100, display=False, num_episodes=3)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Mean Reward:        {results['mean_reward']:8.4f}")
    print(f"Reward Std Dev:     {results['std_reward']:8.4f}")
    print(f"Mean Deviation:     {results['mean_deviation']:8.4f}")
    print(f"Mean Max Deviation: {results['mean_max_deviation']:8.4f}")
    print(f"Episodes:           {results['num_episodes']}")
    print("="*60 + "\n")
    
    # Print individual episode rewards
    print("Individual Episode Rewards:")
    for i, reward in enumerate(results['episode_rewards'], 1):
        print(f"  Episode {i}: {reward:8.4f}")
    print()


    
    # Save summary with evaluation results
    save_agent_summary(agent, "./results/", eval_results=results)
    print("\n✅ Analysis complete!")

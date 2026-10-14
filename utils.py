"""
Utility functions for plotting, logging, and analysis of training results.

Functions include:
- Plotting training metrics
- Logging information
- Analyzing agent behavior
- Video rendering
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
import torch


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_curves(agent, save_dir=None, show=True):
    """
    Plot training curves: rewards, stats, and epsilon.
    
    Args:
        agent: Trained agent object OR path to saved agent (str)
        save_dir: Directory to save plot (optional)
        show: Whether to display the plot
    
    Returns:
        fig: Matplotlib figure object
    """
    
    # Load agent if path provided
    if isinstance(agent, str):
        print(f"Loading agent from: {agent}")
        agent = torch.load(agent, weights_only=False)
    
    # Check if agent has training history
    if not hasattr(agent, 'rewards') or len(agent.rewards) == 0:
        print("⚠️  Agent has no training history (rewards list is empty).")
        print("   This typically happens when loading a saved agent without retraining.")
        print("   To plot training curves, you need to:")
        print("     1. Train the agent and save after training, OR")
        print("     2. Use analyze.py to evaluate and save evaluation results (not training curves)")
        return None
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Rewards over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(agent.rewards, alpha=0.7, linewidth=0.5)
    ax1.plot(np.convolve(agent.rewards, np.ones(100)/100, mode='valid'), 
             color='red', linewidth=2, label='100-step avg')
    ax1.set_title('Total Reward per Step', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Modular rewards over time
    ax2 = fig.add_subplot(gs[0, 1])
    mod_rewards = np.array(agent.mod_rewards)
    for i in range(min(mod_rewards.shape[1], 4)):  # Plot first 4 modules
        ax2.plot(mod_rewards[:, i], alpha=0.6, linewidth=0.5, label=f'Module {i+1}')
    ax2.set_title('Modular Rewards per Step', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Stats over time
    ax3 = fig.add_subplot(gs[0, 2])
    stats = np.array(agent.stats)
    for i in range(stats.shape[1]):
        ax3.plot(stats[:, i], alpha=0.7, linewidth=1, label=f'Stat {i+1}')
    ax3.axhline(y=agent.env.set_point, color='red', linestyle='--', 
                linewidth=2, label='Set Point')
    ax3.set_title('Internal Stats over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Stat Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(agent.rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(np.mean(agent.rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(agent.rewards):.3f}')
    ax4.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Stats distribution
    ax5 = fig.add_subplot(gs[1, 1])
    stats_flat = stats.flatten()
    ax5.hist(stats_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(np.mean(stats_flat), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(stats_flat):.3f}')
    ax5.axvline(agent.env.set_point, color='orange', linestyle='--', 
                linewidth=2, label=f'Set Point: {agent.env.set_point}')
    ax5.set_title('Stats Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Stat Value')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = f"""
    TRAINING SUMMARY
    ═══════════════════════════════════
    
    Reward Statistics:
      Mean:           {np.mean(agent.rewards):.4f}
      Std:            {np.std(agent.rewards):.4f}
      Min:            {np.min(agent.rewards):.4f}
      Max:            {np.max(agent.rewards):.4f}
    
    Stats Statistics:
      Mean:           {np.mean(stats_flat):.4f}
      Std:            {np.std(stats_flat):.4f}
      Set Point:      {agent.env.set_point:.4f}
      
    Final State:
      Stats:          {np.round(agent.env.stats, 2)}
      Cumulative Dev: {agent.cumulative_deviation:.4f}
      Max Dev:        {agent.maximum_deviation:.4f}
    
    Decision Process: {agent.decision_process}
    Total Steps:      {len(agent.rewards):,}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"training_curves_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {filename}")
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig


def plot_value_maps(agent, save_dir=None, show=True):
    """
    Plot value maps for each module (if stored during training).
    
    Args:
        agent: Agent with value_map_history
        save_dir: Directory to save plot
        show: Whether to display plot
    """
    
    if not agent.value_map_history:
        print("⚠️  No value maps stored. Set agent.store_value_maps=True during training.")
        return
    
    num_maps = len(agent.value_map_history)
    num_modules = agent.value_map_history[0].shape[0]
    
    # Plot a few key maps
    indices = [0, num_maps//4, num_maps//2, 3*num_maps//4, -1]
    indices = [i for i in indices if 0 <= i < num_maps]
    
    fig, axes = plt.subplots(len(indices), num_modules, 
                             figsize=(num_modules*3, len(indices)*3))
    
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    for row, map_idx in enumerate(indices):
        v_map = agent.value_map_history[map_idx]
        for col in range(num_modules):
            ax = axes[row, col]
            im = ax.imshow(v_map[col], cmap='viridis')
            ax.set_title(f'Module {col+1} @ Step {map_idx}')
            plt.colorbar(im, ax=ax)
    
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"value_maps_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {filename}")
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig


def plot_heatmap(agent, save_dir=None, show=True):
    """
    Plot heatmap of agent locations visited during training.
    
    Args:
        agent: Agent with location_history
        save_dir: Directory to save plot
        show: Whether to display plot
    """
    
    if not hasattr(agent.env, 'heat_map') or agent.env.heat_map.sum() == 0:
        print("⚠️  No location history. Run training first.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    heat_map = agent.env.heat_map
    im = ax.imshow(heat_map, cmap='hot', interpolation='nearest')
    ax.set_title('Agent Location Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Count')
    
    # Add grid
    ax.grid(True, alpha=0.3, color='white')
    
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"heatmap_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {filename}")
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_homeostasis_score(stats, set_point, window=100):
    """
    Calculate how well agent maintains homeostasis (stats near set point).
    
    Args:
        stats: Array of stats over time (N_steps, N_stats)
        set_point: Target value for stats
        window: Window size for moving average
        
    Returns:
        score: Mean absolute deviation from set point
    """
    
    stats = np.array(stats)
    deviation = np.abs(stats - set_point)
    
    # Smooth with moving average
    if window > 1:
        cumsum = np.cumsum(np.insert(deviation.flatten(), 0, 0))
        moving_avg = (cumsum[window:] - cumsum[:-window]) / window
        return np.mean(moving_avg)
    else:
        return np.mean(deviation)


def calculate_stability(stats, window=100):
    """
    Calculate stability (low variance) of stats.
    
    Args:
        stats: Array of stats over time
        window: Window size for variance calculation
        
    Returns:
        stability: Lower is more stable
    """
    
    stats = np.array(stats)
    
    # Calculate rolling variance
    if window > 1:
        variances = []
        for i in range(len(stats) - window):
            variances.append(np.var(stats[i:i+window]))
        return np.mean(variances)
    else:
        return np.var(stats)


def analyze_agent_performance(agent, window=100):
    """
    Analyze and print agent performance metrics.
    
    Args:
        agent: Trained agent
        window: Window for moving averages
    """
    
    print("\n" + "="*60)
    print("📊 AGENT PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Reward analysis
    rewards = np.array(agent.rewards)
    print(f"\n💰 REWARD ANALYSIS:")
    print(f"   Total:           {np.sum(rewards):10.2f}")
    print(f"   Mean:            {np.mean(rewards):10.4f}")
    print(f"   Std:             {np.std(rewards):10.4f}")
    print(f"   Min:             {np.min(rewards):10.4f}")
    print(f"   Max:             {np.max(rewards):10.4f}")
    
    # Last window average
    if len(rewards) >= window:
        last_window_mean = np.mean(rewards[-window:])
        print(f"   Last {window} steps: {last_window_mean:10.4f}")
    
    # Stats analysis
    stats = np.array(agent.stats)
    print(f"\n📈 STATS ANALYSIS:")
    print(f"   Set Point:       {agent.env.set_point:10.4f}")
    print(f"   Mean:            {np.mean(stats):10.4f}")
    print(f"   Std:             {np.std(stats):10.4f}")
    
    # Homeostasis score
    homeostasis = calculate_homeostasis_score(stats, agent.env.set_point, window)
    print(f"   Homeostasis Score: {homeostasis:10.4f} (lower is better)")
    
    # Stability
    stability = calculate_stability(stats, window)
    print(f"   Stability Score: {stability:10.4f} (lower is better)")
    
    # Final stats
    print(f"\n🎯 FINAL STATE:")
    print(f"   Stats:           {np.round(agent.env.stats, 2)}")
    print(f"   Dead:            {agent.env.dead}")
    print(f"   Location:        {agent.env.loc}")
    
    print("\n" + "="*60 + "\n")


# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def save_agent_summary(agent, save_dir, eval_results=None):
    """
    Save a text summary of agent performance.
    
    Args:
        agent: Agent object
        save_dir: Directory to save summary
        eval_results: Dictionary from evaluate_agent() with 'mean_reward', 'mean_deviation', etc.
                     If None, uses agent.rewards (for agents with training history)
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"summary_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AGENT SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Timestamp:        {timestamp}\n")
        f.write(f"Decision Process: {agent.decision_process}\n")
        f.write(f"Num Modules:      {agent.numModules}\n")
        
        # Check if this is a loaded agent or trained agent
        if eval_results is not None:
            # Use evaluation results (for loaded agents)
            f.write("\nEVALUATION RESULTS:\n")
            f.write(f"  Mean Reward:        {eval_results['mean_reward']:8.4f}\n")
            f.write(f"  Reward Std Dev:     {eval_results.get('std_reward', 'N/A'):>8}\n")
            f.write(f"  Mean Deviation:     {eval_results['mean_deviation']:8.4f}\n")
            f.write(f"  Mean Max Deviation: {eval_results.get('mean_max_deviation', 'N/A'):>8}\n")
            f.write(f"  Num Episodes:       {eval_results.get('num_episodes', 'N/A')}\n\n")
            
            f.write("INDIVIDUAL EPISODE REWARDS:\n")
            if 'episode_rewards' in eval_results:
                for i, reward in enumerate(eval_results['episode_rewards'], 1):
                    f.write(f"  Episode {i}: {reward:8.4f}\n")
        
        elif hasattr(agent, 'rewards'):
            # Use training history (for agents with .rewards attribute)
            f.write(f"Total Steps:      {len(agent.rewards):,}\n\n")
            
            f.write("REWARD STATISTICS:\n")
            f.write(f"  Mean:           {np.mean(agent.rewards):.4f}\n")
            f.write(f"  Std:            {np.std(agent.rewards):.4f}\n")
            f.write(f"  Min:            {np.min(agent.rewards):.4f}\n")
            f.write(f"  Max:            {np.max(agent.rewards):.4f}\n\n")
            
            f.write("STATS STATISTICS:\n")
            stats = np.array(agent.stats)
            f.write(f"  Mean:           {np.mean(stats):.4f}\n")
            f.write(f"  Std:            {np.std(stats):.4f}\n")
            f.write(f"  Final:          {np.round(agent.env.stats, 2)}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Cumulative Dev: {agent.cumulative_deviation:.4f}\n")
            f.write(f"  Max Deviation:  {agent.maximum_deviation:.4f}\n")
            f.write(f"  Dead:           {agent.env.dead}\n")
        
        else:
            f.write("\n⚠️  No evaluation results or training history available.\n")
    
    print(f"✅ Summary saved to: {filename}")


if __name__ == "__main__":
    print("📊 Utils module loaded successfully")
    print("Functions available:")
    print("  - plot_training_curves()")
    print("  - plot_value_maps()")
    print("  - plot_heatmap()")
    print("  - calculate_homeostasis_score()")
    print("  - calculate_stability()")
    print("  - analyze_agent_performance()")
    print("  - save_agent_summary()")

from train import train_agent
from utils import plot_training_curves

agent = train_agent(num_frames=30000)  # Keep in memory
plot_training_curves(agent, save_dir="./results/", show=True)  # Plot!
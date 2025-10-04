"""
Evaluation Script for Trained RL Agent
Compares against baselines and reports comprehensive metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import torch

from microgrid_env import MicrogridEMSEnv
from train_ppo import PPOAgent
from env_config import STEPS_PER_EPISODE, BATTERIES, HOURS_PER_STEP

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class BaselineController:
    """Baseline controllers for comparison"""
    
    @staticmethod
    def rule_based_tou(obs, env):
        """
        Time-of-Use rule-based controller
        - Charge batteries during off-peak hours (cheap prices)
        - Discharge during peak hours (expensive prices)
        - Simple EV charging based on deadline urgency
        """
        # Extract features from observation
        # This is a simplified version - adjust indices based on actual observation
        hour = obs[0] * 24  # Normalized hour
        price_current = obs[50] * 100  # Adjust index
        battery_socs = [obs[30], obs[36]]  # Adjust indices
        
        # Battery control based on TOU
        battery_actions = []
        for i, soc in enumerate(battery_socs):
            if hour < 6:  # Off-peak: charge if below 80%
                if soc < 0.8:
                    battery_actions.append(0.6)  # Charge at 60% capacity
                else:
                    battery_actions.append(0.0)
            elif 17 <= hour < 21:  # Peak: discharge if above 30%
                if soc > 0.3:
                    battery_actions.append(-0.8)  # Discharge at 80% capacity
                else:
                    battery_actions.append(0.0)
            else:  # Mid-peak: maintain
                battery_actions.append(0.0)
        
        # Grid: let it balance naturally
        grid_action = 0.0
        
        # EV charging: charge at moderate rate during off-peak, slower during peak
        if hour < 6:
            ev_action = 0.8
        elif 17 <= hour < 21:
            ev_action = 0.3
        else:
            ev_action = 0.6
        
        # Renewable curtailment: none
        curtailment_action = -1.0  # No curtailment (maps to 0)
        
        return np.array(battery_actions + [grid_action, ev_action, curtailment_action])
    
    @staticmethod
    def greedy(obs, env):
        """
        Greedy controller: minimize immediate cost
        - Use batteries and renewable to minimize grid import
        - Charge batteries when renewable excess
        - Discharge when load exceeds renewable
        """
        # This would need actual implementation with access to env internals
        # For now, simple heuristic
        battery_actions = [0.0, 0.0]
        grid_action = 0.0
        ev_action = 0.5
        curtailment_action = -1.0
        
        return np.array(battery_actions + [grid_action, ev_action, curtailment_action])
    
    @staticmethod
    def random(obs, env):
        """Random controller for baseline"""
        return np.random.uniform(-1, 1, size=env.action_space.shape[0])


def evaluate_agent(
    env: MicrogridEMSEnv,
    agent,
    num_episodes: int = 10,
    controller_type: str = 'rl',
    render: bool = False
) -> Dict:
    """
    Evaluate agent on multiple episodes
    
    Args:
        env: Environment
        agent: Agent or baseline controller
        num_episodes: Number of episodes to evaluate
        controller_type: 'rl', 'rule_based_tou', 'greedy', 'random'
        render: Whether to render episodes
    
    Returns:
        Dictionary of metrics
    """
    episode_metrics = {
        'returns': [],
        'costs': [],
        'emissions': [],
        'unmet_demand_events': [],
        'unmet_demand_energy': [],
        'safety_overrides': [],
        'battery_cycles': [[] for _ in BATTERIES],
        'battery_throughput': [[] for _ in BATTERIES],
        'battery_dod_avg': [[] for _ in BATTERIES],
        'grid_energy_import': [],
        'grid_energy_export': [],
        'peak_import': [],
        'peak_export': []
    }
    
    trajectories = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_return = 0
        
        trajectory = {
            'timesteps': [],
            'observations': [],
            'actions': [],
            'rewards': [],
            'battery_socs': [[] for _ in BATTERIES],
            'battery_soh': [[] for _ in BATTERIES],
            'grid_power': [],
            'renewable_power': [],
            'load_power': [],
            'ev_charging_power': [],
            'price': [],
            'explanations': []
        }
        
        while not done:
            # Select action
            if controller_type == 'rl':
                action = agent.select_action(obs, deterministic=True)
            elif controller_type == 'rule_based_tou':
                action = BaselineController.rule_based_tou(obs, env)
            elif controller_type == 'greedy':
                action = BaselineController.greedy(obs, env)
            elif controller_type == 'random':
                action = BaselineController.random(obs, env)
            else:
                raise ValueError(f"Unknown controller type: {controller_type}")
            
            # Step
            next_obs, reward, done, info = env.step(action)
            
            # Record trajectory
            trajectory['timesteps'].append(env.current_step)
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action.copy())
            trajectory['rewards'].append(reward)
            for i in range(len(BATTERIES)):
                trajectory['battery_socs'][i].append(info['battery_socs'][i])
                trajectory['battery_soh'][i].append(info['battery_soh'][i])
            trajectory['explanations'].append(info.get('explanation', ''))
            
            episode_return += reward
            obs = next_obs
            
            if render:
                env.render(mode='text')
        
        # Record episode metrics
        episode_metrics['returns'].append(episode_return)
        episode_metrics['costs'].append(info['episode_metrics']['total_cost'])
        episode_metrics['emissions'].append(info['episode_metrics']['total_emissions'])
        episode_metrics['unmet_demand_events'].append(info['episode_metrics']['unmet_demand_events'])
        episode_metrics['unmet_demand_energy'].append(info['episode_metrics']['unmet_demand_energy_kwh'])
        episode_metrics['safety_overrides'].append(info['episode_metrics']['safety_overrides'])
        episode_metrics['grid_energy_import'].append(info['episode_metrics']['grid_energy_imported_kwh'])
        episode_metrics['grid_energy_export'].append(info['episode_metrics']['grid_energy_exported_kwh'])
        episode_metrics['peak_import'].append(info['episode_metrics']['peak_import_kw'])
        episode_metrics['peak_export'].append(info['episode_metrics']['peak_export_kw'])
        
        trajectories.append(trajectory)
        
        print(f"Episode {ep+1}/{num_episodes} - Return: {episode_return:.2f}, "
              f"Cost: ${info['episode_metrics']['total_cost']:.2f}, "
              f"Unmet Demand: {info['episode_metrics']['unmet_demand_events']}")
    
    # Compute summary statistics
    summary = {
        'mean_return': np.mean(episode_metrics['returns']),
        'std_return': np.std(episode_metrics['returns']),
        'mean_cost': np.mean(episode_metrics['costs']),
        'std_cost': np.std(episode_metrics['costs']),
        'mean_emissions': np.mean(episode_metrics['emissions']),
        'std_emissions': np.std(episode_metrics['emissions']),
        'total_unmet_demand_events': np.sum(episode_metrics['unmet_demand_events']),
        'mean_unmet_demand_energy': np.mean(episode_metrics['unmet_demand_energy']),
        'mean_safety_overrides': np.mean(episode_metrics['safety_overrides']),
        'mean_grid_import': np.mean(episode_metrics['grid_energy_import']),
        'mean_grid_export': np.mean(episode_metrics['grid_energy_export']),
        'mean_peak_import': np.mean(episode_metrics['peak_import']),
        'mean_peak_export': np.mean(episode_metrics['peak_export']),
    }
    
    return summary, episode_metrics, trajectories


def compare_controllers(
    env: MicrogridEMSEnv,
    rl_agent: PPOAgent,
    num_episodes: int = 10
) -> pd.DataFrame:
    """Compare RL agent against all baselines"""
    
    print("="*60)
    print("Evaluating Controllers")
    print("="*60)
    
    results = {}
    
    # Evaluate RL agent
    print("\n1. RL Agent (PPO)")
    rl_summary, _, _ = evaluate_agent(env, rl_agent, num_episodes, 'rl')
    results['RL (PPO)'] = rl_summary
    
    # Evaluate rule-based TOU
    print("\n2. Rule-Based TOU")
    tou_summary, _, _ = evaluate_agent(env, None, num_episodes, 'rule_based_tou')
    results['Rule-Based TOU'] = tou_summary
    
    # Evaluate greedy
    print("\n3. Greedy Controller")
    greedy_summary, _, _ = evaluate_agent(env, None, num_episodes, 'greedy')
    results['Greedy'] = greedy_summary
    
    # Evaluate random
    print("\n4. Random Controller")
    random_summary, _, _ = evaluate_agent(env, None, num_episodes, 'random')
    results['Random'] = random_summary
    
    # Create comparison dataframe
    df = pd.DataFrame(results).T
    
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    print(df.to_string())
    print("="*60)
    
    return df


def plot_evaluation_results(
    trajectories: List[Dict],
    save_path: str = "evaluation_plots.png"
):
    """Plot detailed evaluation results"""
    
    # Use first trajectory for detailed plots
    traj = trajectories[0]
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    
    hours = np.array(traj['timesteps']) * HOURS_PER_STEP
    
    # Battery SoC
    for i, bat in enumerate(BATTERIES):
        axes[0].plot(hours, traj['battery_socs'][i], label=f'{bat.name} SoC', linewidth=2)
        axes[0].axhline(bat.soc_max, color='r', linestyle='--', alpha=0.3, label='_nolegend_')
        axes[0].axhline(bat.soc_min, color='r', linestyle='--', alpha=0.3, label='_nolegend_')
    axes[0].set_ylabel('SoC')
    axes[0].set_title('Battery State of Charge')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Battery SoH
    for i, bat in enumerate(BATTERIES):
        axes[1].plot(hours, traj['battery_soh'][i], label=f'{bat.name} SoH', linewidth=2)
    axes[1].set_ylabel('SoH')
    axes[1].set_title('Battery State of Health')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.9, 1.01])
    
    # Actions (Battery)
    battery_actions = np.array(traj['actions'])[:, :len(BATTERIES)]
    for i, bat in enumerate(BATTERIES):
        axes[2].plot(hours, battery_actions[:, i], label=f'{bat.name}', linewidth=2)
    axes[2].axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    axes[2].set_ylabel('Normalized Action')
    axes[2].set_title('Battery Actions (Positive=Charge, Negative=Discharge)')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-1.1, 1.1])
    
    # Rewards
    axes[3].plot(hours, traj['rewards'], color='green', linewidth=2)
    axes[3].axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    axes[3].set_ylabel('Reward')
    axes[3].set_title('Step Rewards')
    axes[3].grid(True, alpha=0.3)
    
    # Cumulative reward
    cumulative_reward = np.cumsum(traj['rewards'])
    axes[4].plot(hours, cumulative_reward, color='blue', linewidth=2)
    axes[4].set_ylabel('Cumulative Reward')
    axes[4].set_xlabel('Hour')
    axes[4].set_title('Cumulative Reward Over Episode')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved evaluation plots to {save_path}")
    plt.close()


def plot_comparison_bars(comparison_df: pd.DataFrame, save_path: str = "comparison_bars.png"):
    """Plot bar chart comparison of controllers"""
    
    metrics_to_plot = ['mean_cost', 'mean_emissions', 'mean_safety_overrides', 'total_unmet_demand_events']
    metric_labels = ['Cost ($)', 'Emissions (kg CO2)', 'Safety Overrides', 'Unmet Demand Events']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        if metric in comparison_df.columns:
            comparison_df[metric].plot(kind='bar', ax=axes[i], color=['green', 'blue', 'orange', 'red'])
            axes[i].set_title(label, fontsize=12, fontweight='bold')
            axes[i].set_ylabel(label)
            axes[i].set_xlabel('Controller')
            axes[i].grid(True, alpha=0.3, axis='y')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison bars to {save_path}")
    plt.close()


def main():
    """Main evaluation script"""
    
    # Load data
    print("Loading data...")
    pv_profile = pd.read_csv('data/pv_profile_processed.csv')
    wt_profile = pd.read_csv('data/wt_profile_processed.csv')
    load_profile = pd.read_csv('data/load_profile_processed.csv')
    price_profile = pd.read_csv('data/price_profile_processed.csv')
    
    # Create environment
    print("Creating environment...")
    env = MicrogridEMSEnv(
        pv_profile=pv_profile,
        wt_profile=wt_profile,
        load_profile=load_profile,
        price_profile=price_profile,
        enable_evs=True,
        enable_degradation=True,
        enable_emissions=True,
        forecast_noise_std=0.05,  # Lower noise for evaluation
        random_seed=123
    )
    
    # Load trained agent
    print("Loading trained agent...")
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # Try to load best model
    model_path = "models/best_model.pt"
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("WARNING: No trained model found. Using random initialization.")
    
    # Evaluate and compare
    num_eval_episodes = 20
    comparison_df = compare_controllers(env, agent, num_episodes=num_eval_episodes)
    
    # Save comparison results
    os.makedirs('evaluation', exist_ok=True)
    comparison_df.to_csv('evaluation/controller_comparison.csv')
    print("\nSaved comparison to evaluation/controller_comparison.csv")
    
    # Plot comparison
    plot_comparison_bars(comparison_df, 'evaluation/comparison_bars.png')
    
    # Detailed evaluation of RL agent
    print("\nDetailed evaluation of RL agent...")
    summary, metrics, trajectories = evaluate_agent(env, agent, num_episodes=5, controller_type='rl')
    
    # Plot detailed trajectory
    plot_evaluation_results(trajectories, 'evaluation/rl_trajectory.png')
    
    # Save metrics
    pd.DataFrame(metrics).to_csv('evaluation/rl_detailed_metrics.csv', index=False)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("Results saved to evaluation/ directory")
    print("="*60)


if __name__ == "__main__":
    main()

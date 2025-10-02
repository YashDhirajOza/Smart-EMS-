"""
Quick test script for RL training with reduced parameters
"""

from train_ems_rl import load_and_prepare_data, train_rl_agent, evaluate_agent
import os

def quick_test():
    """Run a quick test of the RL training pipeline"""
    print("=== Quick RL Training Test ===")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    data_df = load_and_prepare_data('processed_ems_data.csv')
    
    # Quick training with fewer timesteps
    print("Training PPO agent (quick test)...")
    model, eval_env = train_rl_agent(
        data_df, 
        algorithm='PPO', 
        total_timesteps=5000,  # Much smaller for testing
        model_name='ems_test'
    )
    
    # Quick evaluation
    print("Evaluating agent...")
    results = evaluate_agent(model, eval_env, n_episodes=5)
    
    print(f"\n=== Quick Test Complete! ===")
    print(f"Average reward: {sum(results['rewards'])/len(results['rewards']):.2f}")
    print(f"Average cost: ${sum(results['costs'])/len(results['costs']):.2f}")
    print(f"Test successful! Ready for full training.")

if __name__ == "__main__":
    quick_test()
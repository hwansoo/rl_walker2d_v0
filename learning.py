from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from custom_walker2d import CustomEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import os
import numpy as np

# TODO: Modify if necessary
# Adjust based on available CPU cores (Windows CMD: wmic cpu get NumberOfLogicalProcessors)
N_ENVS = 4

class CurriculumCallback(CheckpointCallback):
    """Custom callback for curriculum learning progression"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_phase = 0  # 0: flat, 1: practice, 2: challenge
        self.phase_threshold = 2000000  # steps per phase
        
    def _on_step(self) -> bool:
        # Progress curriculum based on timesteps
        current_phase = min(2, self.num_timesteps // self.phase_threshold)
        if current_phase != self.curriculum_phase:
            self.curriculum_phase = current_phase
            phase_names = ["flat ground", "bump practice", "bump challenge"]
            print(f"Curriculum progression: Moving to {phase_names[current_phase]} phase")
        
        return super()._on_step()

def make_env(bump_practice = False, bump_challenge=False, use_original_obs=False):
    def _init():
        env = CustomEnvWrapper(render_mode=None, bump_practice=bump_practice, bump_challenge=bump_challenge, use_original_obs=use_original_obs)
        return env
    return _init

# TODO: Modify if necessary
# Enhanced network architecture for obstacle navigation
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 128, 128, 64], vf=[256, 128, 128, 64])],  # Larger networks for complex terrain
    log_std_init=-0.5,  # Less conservative exploration
    activation_fn='tanh'  # Better for continuous control
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bump_practice", action="store_true", help="Enable bump practice mode")
parser.add_argument("--bump_challenge", action="store_true", help="Enable bump challenge mode")
parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning progression")
parser.add_argument("--transfer_learning", action="store_true", help="Start from pretrained model")
parser.add_argument("--enhanced_obs", action="store_true", help="Use enhanced observation space (only when training from scratch)")
parser.add_argument("--pretrained_path", type=str, default="./checkpoints/pretrained_models/walker_model_1960000_steps.zip", help="Path to pretrained model")
args = parser.parse_args()

if __name__ == "__main__":
    # Check for incompatible arguments
    if args.transfer_learning and args.enhanced_obs:
        print("Warning: --enhanced_obs is ignored when using --transfer_learning (using original observation space for compatibility)")
    
    # Determine whether to use enhanced observations (only for training from scratch)
    use_enhanced_obs = args.enhanced_obs and not args.transfer_learning
    
    # Environment setup - default to curriculum mode if curriculum flag is set
    if args.curriculum:
        # Start with flat ground for curriculum learning
        env = SubprocVecEnv([make_env(bump_practice=False, bump_challenge=False, use_original_obs=not use_enhanced_obs) for _ in range(N_ENVS)])
        folder_name = "curriculum_learning"
    else:
        env = SubprocVecEnv([make_env(bump_practice=args.bump_practice, bump_challenge=args.bump_challenge, use_original_obs=not use_enhanced_obs) for _ in range(N_ENVS)])
        if args.bump_practice:
            folder_name = "bump_practice"
        elif args.bump_challenge:
            folder_name = "bump_challenge"
        else:
            folder_name = "walker_model"
    
    env = VecMonitor(env)
    save_path = f'./checkpoints/{folder_name}/'

    # Setup callbacks
    if args.curriculum:
        curriculum_callback = CurriculumCallback(
            save_freq=50000,
            save_path=save_path,
            name_prefix="curriculum_walker"
        )
        callbacks = [curriculum_callback]
    else:
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,  # Save more frequently for bump challenges
            save_path=save_path,
            name_prefix="walker_model"
        )
        callbacks = [checkpoint_callback]
    
    # TODO: Modify if necessary
    # Optimized hyperparameters for obstacle navigation
    learning_rate = 3e-4 if (args.bump_practice or args.bump_challenge or args.curriculum) else 1e-4
    
    # Model initialization
    if args.transfer_learning and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained model from {args.pretrained_path}")
        print("Note: Using original observation space for transfer learning compatibility")
        
        # Create environment with original observation space for transfer learning
        env.close()
        env = SubprocVecEnv([make_env(bump_practice=args.bump_practice, bump_challenge=args.bump_challenge, use_original_obs=True) for _ in range(N_ENVS)])
        env = VecMonitor(env)
        
        # Load the pretrained model
        model = PPO.load(args.pretrained_path, env=env, verbose=1, tensorboard_log="./logs/")
        
        # Update hyperparameters for fine-tuning
        model.learning_rate = learning_rate * 0.5  # Lower learning rate for fine-tuning
        print("Loaded pretrained model successfully - training with original observation space + enhanced rewards")
        
    else:
        print("Training from scratch with optimized hyperparameters")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log="./logs/", 
            policy_kwargs=policy_kwargs, 
            device="cpu",
            learning_rate=learning_rate,
            batch_size=256,  # Larger batch size for stability
            n_steps=4096,    # More steps per update for better sample efficiency
            gamma=0.995,     # Slightly higher discount for long-term planning
            gae_lambda=0.95, # GAE parameter
            clip_range=0.2,  # PPO clipping parameter
            ent_coef=0.01,   # Entropy coefficient for exploration
            vf_coef=0.5      # Value function coefficient
        )
    
    # Training
    total_timesteps = 15000000 if (args.bump_practice or args.bump_challenge or args.curriculum) else 5000000
    print(f"Starting training for {total_timesteps} timesteps")
    
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    # Save final model
    final_model_name = f"ppo_{folder_name}_final"
    model.save(final_model_name)
    print(f"Training completed! Final model saved as {final_model_name}")
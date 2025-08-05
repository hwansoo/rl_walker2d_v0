# [2025 Samsung DS] RL Project Practice 1: Controller Training

This repository contains the skeleton code for **Practice 1** of the Samsung DS RL Project.  
It is based on the [Walker2d-v5](https://github.com/Farama-Foundation/Gymnasium) environment from OpenAI Gymnasium, and demonstrates training using PPO from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).

## Installation

We have tested this code on Ubuntu 22.04 and Windows.

```bash
# Create a virtual environment
cd {project_folder}
python -m venv venv

.\venv\Scripts\activate # Windows
source ./venv/bin/activate # Linux

# Install dependencies
pip install "stable_baselines3[extra]>=2.0.0a9"
pip install "gymnasium[mujoco]"
```

## Training

Train a controller using Deep Reinforcement Learning (PPO).  
The trained model will be saved in the `checkpoints/` folder as a `.zip` file.

```bash
# Optional arguments:
# --bump_practice    Train with 2 bumps
# --bump_challenge   Train with 8 bumps (for the team challenge)
python learning.py (--bump_practice) (--bump_challenge)
```

### Hyperparameters

You can modify hyperparameters for PPO in `learning.py` under `policy_kwargs` (Refer to the Stable-Baselines3 PPO documentation (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) for available options).

## Rendering

Simulate the trained model.

```bash
# Arguments:
# --model           Path to the trained model
# --bump_practice   Simulate on 2 bumps terrain
# --bump_challenge  Simulate on 8 bumps terrain (team challenge)
# --record          Press 'r' to start recording, and press 'r' again to stop.
#                   Video will be saved as record_{model_name}.mp4
# --test            Test mode - run a few steps without rendering
python render.py --model {model_path} (--bump_practice) (--bump_challenge) (--record) (--test)

# Example usage:
python render.py --model ./checkpoints/bump_challenge/walker_model_200000_steps.zip --bump_challenge
python render.py --model ./checkpoints/bump_challenge/walker_model_200000_steps.zip --bump_challenge --test
```

**Note**: The render script automatically detects whether your model uses 18D (transfer learning) or 30D (enhanced) observation space and configures the environment accordingly.

## Logging

Training logs are saved in the `logs/` directory. You can visualize them with TensorBoard:

```bash
tensorboard --logdir=logs
```

## Environment Details

### Bump Terrain Information

![Simulation with bumps](./asset/obstacle.png)

The environment includes several bump obstacles defined in the XML. 
Their positions and sizes are as follows:

`custom_walker2d_bumps_practice.xml`

| Bump        | Position         | Size              |
|-------------|------------------|-------------------|
| Bump #1     | (6.0, 0.0, 0.0)  | (0.3, 2.0, 0.2)   |
| Bump #2     | (10.0, 0.0, 0.0) | (0.6, 2.0, 0.45)  |


`custom_walker2d_bumps.xml` (team challenge)

| Bump        | Position         | Size              |
|-------------|------------------|-------------------|
| Bump #1     | (2.0, 0.0, 0.0)  | (0.5, 2.0, 0.05)  |
| Bump #2     | (4.0, 0.0, 0.0)  | (0.3, 2.0, 0.15)  |
| Bump #3     | (6.0, 0.0, 0.0)  | (0.2, 2.0, 0.2)   |
| Bump #4     | (8.0, 0.0, 0.0)  | (0.4, 2.0, 0.35)  |
| Bump #5     | (12.0, 0.0, 0.0) | (0.5, 2.0, 0.15)  |
| Bump #6     | (13.0, 0.0, 0.0) | (0.4, 2.0, 0.35)  |
| Bump #7     | (15.0, 0.0, 0.0) | (0.4, 2.0, 0.5)   |
| Bump #8     | (16.0, 0.0, 0.0) | (0.5, 2.0, 0.2)   |
| Bump #9     | (18.0, 0.0, 0.0) | (0.3, 2.0, 0.4)   |
| Bump #10    | (18.8, 0.0, 0.0) | (0.3, 2.0, 0.9)   |
| Bump #11    | (21.0, 0.0, 0.0) | (0.5, 2.0, 0.5)   |
| Bump #12    | (22.0, 0.0, 0.0) | (0.4, 2.0, 1.0)   |
| Bump #13    | (26.0, 0.0, 0.0) | (0.5, 2.0, 0.45)  |
| Bump #14    | (27.0, 0.0, 0.0) | (0.4, 2.0, 0.25)  |
| Bump #15    | (28.0, 0.0, 0.0) | (0.4, 2.0, 0.7)   |
| Bump #16    | (30.0, 0.0, 0.0) | (0.4, 2.0, 0.25)  |
| Bump #17    | (31.0, 0.0, 0.0) | (0.5, 2.0, 0.75)  |
| Bump #18    | (32.0, 0.0, 0.0) | (0.4, 2.0, 1.2)   |
| Bump #19    | (34.0, 0.0, 0.0) | (0.2, 2.0, 0.5)   |
| Bump #20    | (34.5, 0.0, 0.0) | (0.15, 2.0, 0.9)  |
| Bump #21    | (35.0, 0.0, 0.0) | (0.2, 2.0, 1.3)   |


### Enhanced Bump Challenge Features

This implementation includes advanced features for optimal bump challenge performance:

#### Enhanced Observation Space
- **Terrain Scanning**: Ray-casting to detect upcoming obstacles at 5 different distances
- **Contact Detection**: Foot contact sensors for better balance control
- **Stability Features**: Torso stability and height measurements for better navigation

#### Adaptive Reward Function
- **Forward Progress**: Primary reward for moving toward the goal
- **Obstacle Navigation**: Bonus rewards for successfully clearing obstacles
- **Stability**: Rewards for maintaining upright posture and proper height
- **Energy Efficiency**: Slight penalties for excessive joint torques

#### Intelligent Termination
- **Recovery Time**: Allows walker to recover from stumbles instead of immediate termination
- **Height-based**: More lenient fall detection for obstacle navigation
- **Stability-based**: Only terminates on critical falls or extreme tilting

#### Training Strategies
- **Transfer Learning**: Start from pretrained flat-ground model
- **Curriculum Learning**: Progressive difficulty from flat → practice → challenge
- **Optimized Hyperparameters**: Enhanced network architecture and training parameters

### Advanced Training Options

```bash
# Transfer learning from pretrained model (uses original observation space + enhanced rewards)
python learning.py --bump_challenge --transfer_learning

# Training from scratch with enhanced observations (30D observation space)
python learning.py --bump_challenge --enhanced_obs

# Curriculum learning (progressive difficulty)
python learning.py --curriculum --transfer_learning

# Practice mode with transfer learning
python learning.py --bump_practice --transfer_learning

# Use custom pretrained model path
python learning.py --bump_challenge --transfer_learning --pretrained_path ./path/to/model.zip
```

**Important Note**: Transfer learning maintains the original 18D observation space for compatibility with pretrained models, but still benefits from enhanced rewards and termination logic. For the full enhanced observation space (30D with terrain scanning), use `--enhanced_obs` when training from scratch.

### Training Script
For easy training setup, use the provided script:
```bash
./train_bump_challenge.sh
```

### Customization

You can customize the Observation, Reward, and Termination Condition in `custom_walker2d.py`.  The TODO sections have been implemented with advanced features, but can be further modified.

Note: Do not modify the XML file path or environment parameters in `__init__`. Test will be based on the default environment setup.


## Reference

- https://github.com/snumrl/2025_SNU_HumanMotion_HW.git


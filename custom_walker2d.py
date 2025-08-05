import numpy as np
import gymnasium as gym
import os

# The observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
# | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
# | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
# | 0   | x-coordinate of the torso                          | -Inf | Inf | rootz                            | slide | position (m)             |
# | 1   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | rootz                            | slide | position (m)             |
# | 2   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
# | 3   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
# | 4   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
# | 5   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
# | 6   | angle of the left thigh joint                      | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
# | 7   | angle of the left leg joint                        | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
# | 8   | angle of the left foot joint                       | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
# | 9   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
# | 10  | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
# | 11  | angular velocity of the angle of the torso         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
# | 12  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
# | 13  | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
# | 14  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
# | 15  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
# | 16  | angular velocity of the leg hinge                  | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
# | 17  | angular velocity of the foot hinge                 | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |

# The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
# | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
# |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
# | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
# | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
# | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
# | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
# | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
# | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False, use_original_obs=False):
        if bump_challenge:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10,
                healthy_z_range=(0.3, 10.0))  # More lenient lower bound
        elif bump_practice:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10,
                healthy_z_range=(0.3, 10.0))  # More lenient lower bound
        else:
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10)
        
        super().__init__(env)

        self.bump_practice = bump_practice
        self.bump_challenge = bump_challenge
        self.use_original_obs = use_original_obs  # Set this before reset
        
        # Recovery tracking for intelligent termination
        self.low_height_counter = 0
        self.recovery_threshold = 10  # frames to allow for recovery
        
        ## change observation space according to the new observation
        obs, _ = self.reset()
        
        # Use original observation space if specified (for transfer learning compatibility)
        if self.use_original_obs:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.low_height_counter = 0  # Reset recovery counter
        custom_obs = self.custom_observation(obs)
        return custom_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs, reward)
        custom_terminated = self.custom_terminated(terminated, obs)
        custom_truncated = self.custom_truncated(truncated)
        return custom_obs, custom_reward, custom_terminated, custom_truncated, info

    def custom_terminated(self, terminated, obs):
        if not (self.bump_practice or self.bump_challenge):
            return terminated
            
        torso_x, torso_z = obs[0], obs[1]
        torso_angle = obs[2]
        
        # More intelligent termination for bump environments
        
        # 1. Critical fall detection (very low height)
        if torso_z < 0.2:
            return True
            
        # 2. Extreme tilting (completely fallen over)
        if abs(torso_angle) > 1.5:  # ~86 degrees
            return True
            
        # 3. Recovery-based height termination
        if torso_z < 0.6:  # Below safe walking height
            self.low_height_counter += 1
            if self.low_height_counter >= self.recovery_threshold:
                return True  # Failed to recover after threshold frames
        else:
            self.low_height_counter = 0  # Reset counter when height is good
            
        # 4. Backward movement termination (optional, less strict)
        if torso_x < -2.0:  # Moved too far backward
            return True
            
        # Otherwise, allow the walker to continue and potentially recover
        return False
    
    def custom_truncated(self, truncated):
        # TODO: Implement your own truncation condition
        return truncated

    def custom_observation(self, obs):
        # For transfer learning compatibility, check if we need to maintain original observation space
        if hasattr(self, 'use_original_obs') and self.use_original_obs:
            return obs
            
        enhanced_obs = list(obs)
        
        if self.bump_practice or self.bump_challenge:
            torso_x, torso_z = obs[0], obs[1]
            torso_angle = obs[2]
            
            # Terrain scanning: ray-cast forward to detect upcoming obstacles
            scan_distances = [0.5, 1.0, 1.5, 2.0, 2.5]
            terrain_heights = []
            
            for distance in scan_distances:
                scan_x = torso_x + distance
                # Estimate terrain height based on known bump positions
                height = self._estimate_terrain_height(scan_x)
                terrain_heights.append(height)
            
            # Contact detection (approximated from physics)
            left_foot_contact = 1.0 if torso_z < 1.3 else 0.0
            right_foot_contact = 1.0 if torso_z < 1.3 else 0.0
            
            # Balance and stability features
            torso_stability = abs(torso_angle)  # angular deviation from upright
            height_stability = max(0, torso_z - 0.8)  # height above minimum safe level
            
            # Velocity components for better prediction
            vel_x, vel_z = obs[9], obs[10]
            angular_vel = obs[11]
            
            # Add enhanced features to observation
            enhanced_obs.extend([
                *terrain_heights,  # 5 terrain height readings
                left_foot_contact, right_foot_contact,  # 2 contact sensors
                torso_stability, height_stability,  # 2 stability measures
                vel_x, vel_z, angular_vel  # 3 velocity components (already in obs but emphasized)
            ])
        
        return np.array(enhanced_obs, dtype=np.float64)
    
    def _estimate_terrain_height(self, x_pos):
        """Estimate terrain height at given x position based on known bump locations"""
        if not (self.bump_practice or self.bump_challenge):
            return 0.0
            
        if self.bump_practice:
            # Practice bumps at x=6 (h=0.2) and x=10 (h=0.45)
            if 5.7 <= x_pos <= 6.3:
                return 0.2
            elif 9.4 <= x_pos <= 10.6:
                return 0.45
            return 0.0
            
        elif self.bump_challenge:
            # Challenge bumps - approximate height based on position
            bump_data = [
                (2, 0.05), (4, 0.15), (6, 0.2), (8, 0.35),  # First 4 bumps
                (12, 0.15), (13, 0.35), (15, 0.5), (16, 0.2), (18, 0.4), (18.8, 0.9), (21, 0.5), (22, 1.0),  # Next 8
                (26, 0.45), (27, 0.25), (28, 0.7), (30, 0.25), (31, 0.75), (32, 1.2), (34, 0.5), (34.5, 0.9), (35, 1.3)  # Final 9
            ]
            
            for bump_x, bump_height in bump_data:
                # Check if x_pos is within bump range (approximate width based on size)
                if abs(x_pos - bump_x) <= 0.5:  # Most bumps have width around 0.3-0.6
                    return bump_height
            return 0.0
        
        return 0.0

    def custom_reward(self, obs, original_reward):
        if not (self.bump_practice or self.bump_challenge):
            return original_reward
            
        torso_x, torso_z = obs[0], obs[1]
        torso_angle = obs[2]
        vel_x, vel_z = obs[9], obs[10]
        angular_vel = obs[11]
        
        reward = 0.0
        
        # 1. Forward progress reward (main objective)
        progress_reward = vel_x * 2.0  # Encourage forward movement
        if vel_x > 0:
            progress_reward += 0.5  # Bonus for positive velocity
        reward += progress_reward
        
        # 2. Stability rewards
        # Upright posture bonus
        upright_bonus = max(0, 1.0 - abs(torso_angle))  # Reward for staying upright
        reward += upright_bonus * 0.5
        
        # Height maintenance
        height_reward = 0.0
        if torso_z > 1.0:  # Normal walking height
            height_reward = min(1.0, torso_z - 1.0)
        elif torso_z < 0.8:  # Penalize being too low
            height_reward = -(0.8 - torso_z) * 2.0
        reward += height_reward
        
        # 3. Obstacle navigation bonuses
        current_terrain_height = self._estimate_terrain_height(torso_x)
        if current_terrain_height > 0:  # On or near an obstacle
            # Bonus for successfully navigating obstacles
            if torso_z > current_terrain_height + 0.5:  # Successfully over obstacle
                obstacle_bonus = current_terrain_height * 3.0  # Higher obstacles = more reward
                reward += obstacle_bonus
            
            # Extra stability reward when near obstacles
            if abs(torso_angle) < 0.3:  # Stable while on obstacle
                reward += 1.0
        
        # 4. Energy efficiency (slight penalty for excessive actions)
        # This will be calculated from actions in the step function if needed
        
        # 5. Survival bonus - encourage longer episodes
        survival_bonus = 0.1
        reward += survival_bonus
        
        # 6. Penalties for dangerous states
        # Excessive tilting
        if abs(torso_angle) > 0.8:
            reward -= 2.0
        
        # Falling (low height)
        if torso_z < 0.5:
            reward -= 5.0
        
        # Excessive angular velocity (spinning)
        if abs(angular_vel) > 3.0:
            reward -= 1.0
        
        # 7. Backward movement penalty
        if vel_x < -0.5:
            reward -= abs(vel_x) * 2.0
        
        return reward

## Test Rendering
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs = env.reset()

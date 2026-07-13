from pettingzoo.utils import ParallelEnv
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import functools
from scipy.constants import speed_of_light 


""" According to the article: Experiment Setup

UAV number = 3
User number = 40
L = 1000 m
H = 100 m
Mission time T = 100 slot
Each slot duration = 1 s
20% user wk = 5, 30% user wk = 2, 50% user wk = 1

"""


"""
The environment is designed to simulate a UAV emergency communication scenario.



"""



class UAVEmergencyEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "uav_emergency_v0"}

    def __init__(self, render_mode = None, num_uavs=3, num_users=40):
        # Initialize the environment with the number of UAVs and users

        self.num_users = num_users
        self.num_uavs = num_uavs
        self.possible_agents = [f"uav_{i}" for i in range(self.num_uavs)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode
        
        self.max_observed_user = 10 # Each UAV can connect to 10 users
        
        self.L = 1000  # Size of the environment (L x L)
        self.H = 100   # UAV altitude (fixed)
        self.d_min = 50  # Minimum distance between UAVs to avoid collision
        self.E_max = 30  # Maximum energy capacity of UAVs (Kj)
        self.E_min = 5  # Minimum energy threshold for UAVs (Kj)
        
         
        self.uav_position = np.zeros((self.num_uavs, 2))  # UAV positions (x, y)
        self.uav_energy = np.ones(self.num_uavs) * self.E_max  # UAV energy levels (initially full)
        self.user_positions = np.zeros((self.num_users, 2))  # User positions (x, y)
        self.user_priority = np.zeros(self.num_users)  # User urgency weights (to be defined)

        self.move_step = 20.0
        self.V_max = 20  # Maximum UAV speed (m/s)

        self.alpha_env = 9.61 # Alpha env: enviroment-depent parameter
        self.beta_env = 0.16 # Beta env: enviroment-depent parameter
        self.fc = 2e9 # Carrier frequency (hz)
        self.eta_LoS = 1 # excessive path-loss components associated with LoS propagation
        self.eta_NLoS = 20 # excessive path-loss components associated with NLoS propagation

        
        # 3. Global State Space
        global_state_dim = (self.num_uavs * 2 +  # UAV positions (x, y) H is fixed and energy
                            self.num_uavs * 1 +  # UAV energy levels
                            self.num_users * 2 +  # User positions (x, y)
                            self.num_users * 1 +   # User urgency weight
                            self.num_users * 1)  # large-scale channel gain for each user

        # s[t] 
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(global_state_dim,), dtype=np.float32) 


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        This method defines the observation space for each UAV agent in the environment. 
        
        Args: 
            agent: The agent for which the observation space is being defined.
        
        Returns:
            The observation space for the specified agent.
        """
        
        self_dim = 2 + 1                            # Self: position_norm (2) + energy_norm (1)
        
        other_uavs_dim = (self.num_uavs - 1) * 4    # Other UAV: position (2) + distance (1) + energy (1) = 4
        
        users_dim = self.max_observed_user * 4      # User: distance_3D (1) + position (2) + weight (1) = 4
        
        self.obs_dim = self_dim + other_uavs_dim + users_dim 

        return Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        This method defines the action space for each UAV agent in the environment.
        Args:
            agent: The agent for which the action space is being defined.
        Returns:
            The action space for the specified agent.
        """

        # 9 discrete actions: 8 directions + hover
        # 4 service groups: 0, 1, 2, 3
        # 3 power levels: 0.25, 0.5, 1.0
        return MultiDiscrete([9, 4, 3])

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        if seed is not None:
            np.random.seed(seed)
        
        # Initialize the environment state 
        # User positions are randomly distributed in the environment
        self.user_positions = np.random.uniform(0, self.L, size=(self.num_users, 2)) 
        user_wk5 = int(0.20 * self.num_users)                                                 # Number of users with urgency weight 5
        user_wk2 = int(0.30 * self.num_users)                                                 # Number of users with urgency weight 2
        user_wk1 = self.num_users - user_wk5 - user_wk2                                       # Number of users with urgency weight 1

        weights = ([5.0] * user_wk5) + ([2.0] * user_wk2) + ([1.0] * user_wk1)
        np.random.shuffle(weights)  # Shuffle the weights to randomize user urgency
        self.user_priority = np.array(weights)

        # Initialize UAV positions randomly within the environment
        # for i in range(self.num_uavs):
        #     valid_position = False
        #     while not valid_position:
        #         position = np.random.uniform(0, self.L, size=2) 
        #         valid_position = True
        #         for j in range(i):
        #             if np.linalg.norm(position - self.uav_position[j]) < self.d_min:
        #                 valid_position = False
        #                 break
        #         if valid_position:    
        #             self.uav_position[i] = position
        for i in range(self.num_uavs):
            while True:
                position = np.random.uniform(0, self.L, size=2)
                if all(np.linalg.norm(position - self.uav_position[j]) >= self.d_min for j in range(i)):
                    self.uav_position[i] = position
                    break
        # Initialize UAV energy levels
        self.uav_energy = np.ones(self.num_uavs) * self.E_max


        # Return local observations for each UAV
        observations = {agent: self._get_local_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    


    def _get_local_obs(self, agent_id):
        """
        This method constructs the local observation for a specific UAV agent.
        Args:
            agent_id: The ID of the UAV agent for which the local observation is being constructed.
        Returns:
            local_obs: The local observation for the specified UAV agent.
        """
        """
        This method constructs the local observation for a specific UAV agent.
        Args:
            agent_id: The ID of the UAV agent for which the local observation is being constructed.
        Returns:
            position_norm: The normalized position of the UAV agent.    shape (2,)
            energy_norm: The normalized energy level of the UAV agent.  shape (1,)

            other_uav_position: The normalized positions of other UAVs.  shape (2*(num_uavs-1),)
            distance_to_other_uavs: The distances to other UAVs.  shape (num_uavs-1,)
            other_uav_energy: The normalized energy levels of other UAVs.  shape (num_uavs-1,)

            close_user_distance_3D: The distances to the closest observed users.  shape (max_observed_user,2)
            close_user_position: The normalized positions of the closest observed users.  shape (max_observed_user,2)
            close_user_weight: The normalized urgency weights of the closest observed users.  shape (max_observed_user,1)

        """
        agent_idx = self.agents.index(agent_id)
        MAX_WEIGHT = 5.0
        D_MAX = np.sqrt(2 * (self.L ** 2) + (self.H ** 2))  # Maximum possible distance in the environment
        # Local Observation

        # Normalize UAV position [0,1]
        self_position_norm = self.uav_position[agent_idx] / self.L  

        # Normalize UAV energy [0,1]
        self_energy_norm = np.array([self.uav_energy[agent_idx] / self.E_max])

        # User positions (x, y)
        self_user_positions = self.user_positions / self.L  

        # Normalize other UAV positions [0,1]
        other_uav_position = []
        for i in range(self.num_uavs):
            if i != agent_idx:
                other_uav_position.append(self.uav_position[i] / self.L)       
        other_uav_position = np.concatenate(other_uav_position) if other_uav_position else np.array([])

        # Distance to other UAVs
        distance_to_other_uavs = np.linalg.norm(self.uav_position[agent_idx] - other_uav_position.reshape(-1, 2), axis=1) / D_MAX if other_uav_position.size > 0 else np.array([])

        # Other UAV energy levels
        other_uav_energy = []
        for j in range(self.num_uavs):
            if j != agent_idx:
                other_uav_energy.append(self.uav_energy[j] / self.E_max)  # Normalize energy
        other_uav_energy = np.array(other_uav_energy) if other_uav_energy else np.array([])


        
        distances = np.linalg.norm(self_position_norm - self_user_positions, axis=1) 

        sorted_indices = np.argsort(distances)
        close_indicies = sorted_indices[:self.max_observed_user]
        
        close_user_distance = distances[close_indicies] # Shape [max_observed_user]
        close_user_distance_3D = np.hypot(close_user_distance, self.H) / D_MAX  # Normalize 3D distance [0,1]
        close_user_weight = self.user_priority[close_indicies] / MAX_WEIGHT  # Normalize user urgency weight [0,1]
        close_user_position = self_user_positions[close_indicies].flatten()  # User positions [0,1]



        # Large-scale channel gain (average of observed users)

        close_user_channel_gain = self._compute_channel_gain(agent_idx, close_indicies)

        # To be continued
                # print("Local Observation for", agent_id)
        # print("  UAV Position (normalized):", self_position_norm)
        # print("  UAV Energy (normalized):", self_energy_norm)
        # print("  Other UAV Positions (normalized):", other_uav_position)
        # print("  Distances to Other UAVs:", distance_to_other_uavs)
        # print("  Other UAV Energy Levels (normalized):", other_uav_energy)
        # print("  Closest User Distances (3D):", close_user_distance_3D)
        # print("  Closest User Positions (normalized):", close_user_position)
        # print("  Closest User Weights (normalized):", close_user_weight)    
        print("  Close user channel gain:" ,close_user_channel_gain)

        # print("self_position_norm shape:", self_position_norm.shape)
        # print("self_energy_norm shape:", self_energy_norm.shape)
        # print("other_uav_position shape:", other_uav_position.shape)
        # print("distance_to_other_uavs shape:", distance_to_other_uavs.shape)
        # print("other_uav_energy shape:", other_uav_energy.shape)
        # print("close_user_distance_3D shape:", close_user_distance_3D.shape)
        # print("close_user_position shape:", close_user_position.shape)
        # print("close_user_weight shape:", close_user_weight.shape)
        print("close_user_channel_gain:", close_user_channel_gain.shape)
        # Trả về mảng numpy chứa thông tin của uav đó
        local_obs = np.concatenate([
            self_position_norm,
            self_energy_norm,
            other_uav_position,
            other_uav_energy,
            distance_to_other_uavs,
            close_user_distance_3D,
            close_user_weight,
            close_user_position,
            close_user_channel_gain
        ])
        return local_obs
    
    def _compute_channel_gain(self,agent_idx,user_indicies):
        
        horizontal_distance = np.linalg.norm(self.uav_position[agent_idx] - self.user_positions[user_indicies],axis = 1) 
        link_distance = np.hypot(horizontal_distance, self.H) # Formula 9

        elevation_angle = 180/np.pi * np.arcsin(self.H / link_distance) # Formula 10

        pLoS = 1 / (1 + self.alpha_env * np.exp(-self.beta_env*(elevation_angle - self.alpha_env ))) # Formula 11
        pNLoS = 1 - pLoS
        
        PLoss = 20 * np.log10 ( 4 * np.pi * self.fc * link_distance / speed_of_light )  + pLoS * self.eta_LoS + pNLoS * self.eta_NLoS #Formula 12

        channel_gain = np.power(10,-PLoss/10)
        return channel_gain



            
        
    def _move_uav(self, agent_idx, move_action):
        """
        Cập nhật vị trí UAV dựa trên action di chuyển (0-8).
        Trả về: đã di chuyển hay không (để tính năng lượng), vị trí mới đã clip biên
        """
        direction = self.direction_vectors[move_action]  # (2,)
        displacement = direction * self.move_step

        new_position = self.uav_position[agent_idx] + displacement

        # Giới hạn trong biên [0, L] x [0, L]
        new_position = np.clip(new_position, 0, self.L)

        is_hover = (move_action == 8)
        self.uav_position[agent_idx] = new_position

        return is_hover
    
    def _update_energy(self, agent_idx, is_hover, transmit_power_level):
        # Năng lượng di chuyển (bay tốn nhiều hơn hover)
        move_energy_cost = 0.05 if is_hover else 0.15  # Kj/slot, cần tune theo bài báo gốc
        
        # Năng lượng truyền dữ liệu - phụ thuộc action thứ 3 (mức công suất phát)
        transmit_energy_cost = 0.02 * (transmit_power_level + 1)  # ví dụ tuyến tính

        total_cost = move_energy_cost + transmit_energy_cost
        self.uav_energy[agent_idx] = max(0.0, self.uav_energy[agent_idx] - total_cost)

    def step(self, actions):
        # Execute actions for all UAVs simultaneously
        # Update energy, positions, calculate rewards (r[t])
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

        # Update UAV positions

        # angles = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180
        # self.direction_vectors = np.array([
        #     [np.cos(a), np.sin(a)] for a in angles
        # ])
        # self.direction_vectors = np.vstack([self.direction_vectors, [0.0, 0.0]])  # index 8 = hover
        # # shape: (9, 2)

        # is_hover = self._move_uav()    


        observations = {agent: self._get_local_obs(agent) for agent in self.agents}
        
        # Trả về phần thưởng chung hoặc riêng tùy định nghĩa (Thường trong CTDE là chung)
        rewards = {agent: 0.0 for agent in self.agents} 
        
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Nếu một agent bị loại bỏ (ví dụ: hết pin), hãy xóa nó khỏi self.agents
        
        return observations, rewards, terminations, truncations, infos



    def state(self):
        """
        Xây dựng s[t]: Phương thức cốt lõi cho CTDE trong PettingZoo.
        Trả về trạng thái toàn cục để nạp vào Centralized Critic.
        """

        
        # Trộn tất cả thông tin toàn cục thành một mảng 1D duy nhất
        return np.zeros(self.state_space.shape, dtype=np.float32)
    
if __name__ == "__main__":
    # This is a simple way to test your environment.
    # It will not be part of the PettingZoo test suite, but can be useful for debugging.
    env = UAVEmergencyEnv(render_mode="human")
    observations, infos = env.reset(seed=42)

    print("Initial Observations:", observations)
    print("Initial Infos:", infos)
    print("Observation space", env.observation_space("uav_0"))
    print("Action space", env.action_space("uav_0"))



    """print("Observations shape for each UAV:")
    for agent, obs in observations.items():
        print(f"  {agent}: {obs.shape}")

    print("Action space UAV 1:", env.action_space("uav_0"))

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)"""




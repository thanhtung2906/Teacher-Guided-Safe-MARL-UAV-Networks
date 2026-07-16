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
        self.D_MAX = np.sqrt(2 * (self.L ** 2) + (self.H ** 2)) # Maximum possible distance in (L x L) enviroment

        self.E_max = 30000  # Maximum energy capacity of UAVs (Joules)
        self.E_min = 5000  # Minimum energy threshold for UAVs (Joules)
        
        
        self.time_slot = 100 # Mission duration divided into T time slots
        self.delta_T = 1 # Each time slot duration
        
        
        self.uav_position = np.zeros((self.num_uavs, 2))  # UAV positions (x, y)
        self.uav_energy = np.ones(self.num_uavs) * self.E_max  # UAV energy levels (initially full)
        self.user_positions = np.zeros((self.num_users, 2))  # User positions (x, y)
        self.user_priority = np.zeros(self.num_users)  # User urgency weights (to be defined)

        self.V_step = 10.0
        self.V_max = 20.0  # Maximum UAV speed (m/s)
        self.power_levels = [0.25, 0.5, 1.0]
        self.B_m = 1e6 # Bandwidth per UAV (Hz)
        self.P_m = 0.5 # Maximum transmit power (W)
        self._renderer = None
        
        self.agent_name_to_idx = {agent: i for i, agent in enumerate(self.possible_agents)} # Mapping agent into index
        self.P0 = 79.86 # Blade profile power (W)
        self.Pi = 88.63 # Induced power (W)
        self.Utip = 120 # Rotor blade tip speed (m/s)
        self.v0 = 4.03 # Mean induced velocity (m/s)
        self.d0 = 0.6 # Fuselage drag ratio 
        self.rho = 1.225 # Air density (kg/m3)
        self.s = 0.05 # Rotor solidity
        self.A = 0.503 # Rotor disc area

        self.alpha_env = 9.61 # Alpha env: enviroment-depent parameter
        self.beta_env = 0.16 # Beta env: enviroment-depent parameter
        self.fc = 2e9 # Carrier frequency (Hz)
        self.eta_LoS = 1 # excessive path-loss components associated with LoS propagation
        self.eta_NLoS = 20 # excessive path-loss components associated with NLoS propagation
        self.noise_PSD = -174 # Noise PSD dBm/Hz

        # Reward coefficients - Formula (35)
        self.lambda_E = 30   # hệ số phạt năng lượng (cần scale vì E tính bằng J, R tính bằng bps)
        self.lambda_Q = 100    # hệ số phạt vi phạm QoS khẩn cấp
        self.lambda_C = 30    # hệ số phạt va chạm
        self.lambda_B = 30    # hệ số phạt pin cạn
        self._cumulative_reward = 0.0
        self.consumed_energy = np.zeros(num_uavs)
        # QoS threshold cho user khẩn cấp - dùng trong (31h), (32)
        self.R_min_emg = 0.1   # bps, giá trị cụ thể bạn cần chọn theo thực nghiệm

        # Đánh dấu user nào là "khẩn cấp" (K_emg ⊆ K)
        self.is_emergency = None  # sẽ set trong reset(), ví dụ user có weight=5

        self.angles = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180
        self.direction_vectors = np.array([[np.cos(a), np.sin(a)] for a in self.angles])
        self.direction_vectors = np.vstack([self.direction_vectors, [0.0, 0.0]])  # index 8 = hover, shape (9,2)

        self.a = np.zeros((self.num_uavs, self.num_users), dtype=np.int32)    # association indicator
        self.p = np.zeros((self.num_uavs, self.num_users), dtype=np.float32)  # transmit power
        self.b = np.zeros((self.num_uavs, self.num_users), dtype=np.float32)  # bandwidth
        
        gain_max , gain_min = self._channel_gain_bounds() # Gain max min according to D_max and H
        self.gain_max_db = 10 * np.log10(gain_max) # Convert into db
        self.gain_min_db = 10 * np.log10(gain_min)

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
        
        channel_gain_dim = self.max_observed_user * 1 # Channel gain: channel_gain (1)
        self.obs_dim = self_dim + other_uavs_dim + users_dim + channel_gain_dim



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
        """
        This function resets the enviroment to new random enviroment 
        Args:
            seed: random initialize seed 
        Returns:
            new enviroment
        """
        self.agents = self.possible_agents[:]

        if seed is not None:
            np.random.seed(seed)
        
        # Initialize the environment state 
        # User positions are randomly distributed in the environment
        self.user_positions = np.random.uniform(0, self.L, size=(self.num_users, 2)) \

        user_wk5 = int(0.20 * self.num_users)                                                 # Number of users with urgency weight 5
        user_wk2 = int(0.30 * self.num_users)                                                 # Number of users with urgency weight 2
        user_wk1 = self.num_users - user_wk5 - user_wk2                                       # Number of users with urgency weight 1
        self._cumulative_reward = 0.0
        weights = ([5.0] * user_wk5) + ([2.0] * user_wk2) + ([1.0] * user_wk1)
        np.random.shuffle(weights)  # Shuffle the weights to randomize user urgency
        self.user_priority = np.array(weights)
        self.is_emergency = (self.user_priority == 5.0)
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

        self.a = np.zeros((self.num_uavs, self.num_users), dtype=np.int32)
        self.p = np.zeros((self.num_uavs, self.num_users), dtype=np.float32)
        self.b = np.zeros((self.num_uavs, self.num_users), dtype=np.float32)

        self.current_step = 0

        # Return local observations for each UAV
        observations = {agent: self._get_local_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def _channel_gain_bounds(self):
        
        def gain_compute(horizontal_distance):
            link_distance = np.hypot(horizontal_distance, self.H) # Formula 9

            elevation_angle = 180/np.pi * np.arcsin(self.H / link_distance) # Formula 10

            pLoS = 1 / (1 + self.alpha_env * np.exp(-self.beta_env*(elevation_angle - self.alpha_env ))) # Formula 11
            pNLoS = 1 - pLoS
            
            PLoss = 20 * np.log10 ( 4 * np.pi * self.fc * link_distance / speed_of_light )  + pLoS * self.eta_LoS + pNLoS * self.eta_NLoS #Formula 12
            channel_gain = np.power(10,-PLoss/10)
            return channel_gain
        gain_max = gain_compute(0.0)
        gain_min = gain_compute(self.D_MAX)
        
        return gain_max, gain_min




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
        agent_idx = self.agent_name_to_idx[agent_id]
        MAX_WEIGHT = 5.0
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
        distance_to_other_uavs = np.linalg.norm(self.uav_position[agent_idx] - other_uav_position.reshape(-1, 2), axis=1) / self.D_MAX if other_uav_position.size > 0 else np.array([])

        # Other UAV energy levels
        other_uav_energy = []
        for j in range(self.num_uavs):
            if j != agent_idx:
                other_uav_energy.append(self.uav_energy[j] / self.E_max)  # Normalize energy
        other_uav_energy = np.array(other_uav_energy) if other_uav_energy else np.array([])


        
        distances = np.linalg.norm(self.uav_position[agent_idx]- self.user_positions, axis=1) 

        sorted_indices = np.argsort(distances)
        close_indicies = sorted_indices[:self.max_observed_user]
        
        close_user_distance = distances[close_indicies] # Shape [max_observed_user]
        close_user_distance_3D = np.hypot(close_user_distance, self.H) / self.D_MAX  # Normalize 3D distance [0,1]
        
        close_user_weight = self.user_priority[close_indicies] / MAX_WEIGHT  # Normalize user urgency weight [0,1]
        close_user_position = self_user_positions[close_indicies].flatten()  # User positions [0,1]



        # Large-scale channel gain (average of observed users)

        close_user_c_g = self._compute_channel_gain(agent_idx, close_indicies)
        close_user_c_g_db = 10 * np.log10(close_user_c_g)
        close_user_c_g_norm = (close_user_c_g_db - self.gain_min_db) / (self.gain_max_db - self.gain_min_db + 1e-9) # Min -  max normalization

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
        #print("  Close user channel gain:" ,close_user_channel_gain)

        # print("self_position_norm shape:", self_position_norm.shape)
        # print("self_energy_norm shape:", self_energy_norm.shape)
        # print("other_uav_position shape:", other_uav_position.shape)
        # print("distance_to_other_uavs shape:", distance_to_other_uavs.shape)
        # print("other_uav_energy shape:", other_uav_energy.shape)
        # print("close_user_distance_3D shape:", close_user_distance_3D.shape)
        # print("close_user_position shape:", close_user_position.shape)
        # print("close_user_weight shape:", close_user_weight.shape)
        #print("close_user_channel_gain:", close_user_channel_gain.shape)
        #print("UAV Energy", self.uav_energy)
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
            close_user_c_g_norm
        ])
        return local_obs.astype(np.float32)
    

    def _compute_channel_gain(self,agent_idx,user_indicies):
        
        horizontal_distance = np.linalg.norm(self.uav_position[agent_idx] - self.user_positions[user_indicies],axis = 1) 
        link_distance = np.hypot(horizontal_distance, self.H) # Formula 9

        elevation_angle = 180/np.pi * np.arcsin(self.H / link_distance) # Formula 10

        pLoS = 1 / (1 + self.alpha_env * np.exp(-self.beta_env*(elevation_angle - self.alpha_env ))) # Formula 11
        pNLoS = 1 - pLoS
        
        PLoss = 20 * np.log10 ( 4 * np.pi * self.fc * link_distance / speed_of_light )  + pLoS * self.eta_LoS + pNLoS * self.eta_NLoS #Formula 12

        channel_gain = np.power(10,-PLoss/10)
        
        return channel_gain
    
    def _get_nearest_user_indices(self, agent_idx):
        """
        Trả về chỉ số của K user gần nhất mà UAV agent_idx quan sát được.
        Tách riêng từ _get_local_obs để dùng chung với service-group building.
        """
        uav_pos_norm = self.uav_position[agent_idx] / self.L
        user_pos_norm = self.user_positions / self.L
        distances = np.linalg.norm(uav_pos_norm - user_pos_norm, axis=1)
        sorted_indices = np.argsort(distances)
        return sorted_indices[:self.max_observed_user]


    def _compute_user_score(self, agent_idx, candidate_indices):
        """
        Composite score = effective priority weight x channel quality (normalized).
        Dùng để rank user trước khi chia vào service groups.
        """
        weights = self.user_priority[candidate_indices]
        gains = self._compute_channel_gain(agent_idx, candidate_indices)

        gains_db = 10 * np.log10(gains + 1e-20)
        g_min, g_max = gains_db.min(), gains_db.max()
        gains_norm = (gains_db - g_min) / (g_max - g_min + 1e-9) #Min max normalize
        weights_norm = weights / 5.0  # MAX_WEIGHT

        score = weights_norm * gains_norm
        return score


    def _build_service_groups(self, agent_idx, num_groups=4):
        """
        Chia K user gần nhất thành `num_groups` candidate group,
        theo kiểu nested/cumulative: group g = top (g+1)*group_size user tốt nhất.
        """
        candidate_indices = self._get_nearest_user_indices(agent_idx)
        scores = self._compute_user_score(agent_idx, candidate_indices)
        ranked = candidate_indices[np.argsort(-scores)]  # giảm dần theo score

        K = len(ranked)
        group_size = max(1, K // num_groups)
        groups = []
        for g in range(num_groups):
            n_users = min(K, (g + 1) * group_size)
            groups.append(ranked[:n_users])
        return groups

    def _resolve_association_and_resources(self, actions):
        """
        Deterministic control layer: map action -> association (a), power (p), bandwidth (b).
        Trả về 3 ma trận shape (num_uavs, num_users).
        Thực hiện công thức (14)-(19) của paper.
        """
        M, K = self.num_uavs, self.num_users
        num_srv_groups = 4   # khớp chiều thứ 2 của MultiDiscrete([9,4,3])

        a = np.zeros((M, K), dtype=np.int32)
        p = np.zeros((M, K), dtype=np.float32)
        b = np.zeros((M, K), dtype=np.float32)

        # ---- BƯỚC 1: mỗi UAV chọn service group thô, có thể xung đột ----
        raw_served = {}  # agent_idx -> array user indices được chọn (trước khi resolve xung đột)
        for agent in self.agents:
            m = self.agent_name_to_idx[agent]
            _, srv_action, pow_action = actions[agent]

            groups = self._build_service_groups(m, num_srv_groups)
            served_users = groups[srv_action]
            raw_served[m] = served_users

            for k in served_users:
                a[m, k] = 1

        # ---- BƯỚC 2: giải quyết xung đột - ràng buộc (14), mỗi user tối đa 1 UAV ----
        for k in range(K):
            serving_uavs = np.where(a[:, k] == 1)[0]
            if len(serving_uavs) > 1:
                # user được nhiều UAV chọn -> giữ UAV có channel gain tốt nhất
                gains = [self._compute_channel_gain(m, np.array([k]))[0] for m in serving_uavs]
                best_m = serving_uavs[np.argmax(gains)]
                for m in serving_uavs:
                    if m != best_m:
                        a[m, k] = 0

        # ---- BƯỚC 3: cập nhật lại danh sách user thực sự được phục vụ (sau resolve xung đột) ----
        # rồi phân bổ power/bandwidth - công thức (15)-(19)
        for agent in self.agents:
            m = self.agent_name_to_idx[agent]
            _, _, pow_action = actions[agent]

            served_users = np.where(a[m, :] == 1)[0]   # danh sách CUỐI CÙNG sau khi resolve xung đột
            K_m = len(served_users)

            if K_m == 0:
                continue  # p[m,:] và b[m,:] giữ nguyên 0, đúng yêu cầu paper

            # Power allocation - chia đều trong nhóm, tổng <= P_max (công thức 15, 17)
            power_levels = self.power_levels  # ví dụ [0.3, 0.6, 1.0], định nghĩa sẵn trong __init__
            total_power = power_levels[pow_action] * self.P_m
            power_per_user = total_power / K_m
            p[m, served_users] = power_per_user

            # Bandwidth allocation - equal sharing, công thức (19)
            bandwidth_per_user = self.B_m / K_m
            b[m, served_users] = bandwidth_per_user

        return a, p, b
    def _compute_noise_power(self, agent_idx, user_indicies, b):
        """
        Compute noise power from uav m to user k
        noise_power = b_m,k * noise_psd 
        """
        b_m_k = b[agent_idx, user_indicies]  # Hz
        noise_psd_linear = 10 ** (self.noise_PSD / 10) * 1e-3   # dBm -> W/Hz
        noise_power = noise_psd_linear * b_m_k                    # W
        return noise_power
    def _compute_SINR(self, p, b, agent_idx, user_indicies):
        """
        Compute SINR for user in `user_indicies` when being served by `agent_idx`.
        p: power matrix, shape (num_uavs, num_users) - p[j, l] = p_{j,l}[t]
        
        agent_idx: Serving UAV
        user_indicies: Users served by UAV agent_idx 

        Return: SINR array, shape (len(user_indicies),) 
        """
        noise_power = self._compute_noise_power(agent_idx, user_indicies, b)
        # --- Tử số: signal mong muốn từ chính UAV agent_idx ---
        g_mk = self._compute_channel_gain(agent_idx, user_indicies)      # g_{m,k}, shape (N,)
        p_mk = p[agent_idx, user_indicies]                                 # p_{m,k}, shape (N,)
        signal = p_mk * g_mk

        # --- Mẫu số: nhiễu từ các UAV khác (worst-case, theo Remark 1) ---
        interference = np.zeros(len(user_indicies))
        for j in range(self.num_uavs):
            if j == agent_idx:
                continue
            total_power_j = np.sum(p[j, :])              # sum_l p_{j,l}[t] - TOÀN BỘ công suất UAV j phát ra
            g_jk = self._compute_channel_gain(j, user_indicies)   # g_{j,k} - gain từ UAV j đến user k (đang xét)
            interference += total_power_j * g_jk

        sinr = signal / (interference + noise_power)   # sigma^2 = self.noise_power
        return sinr
    def _compute_achievable_rate(self, a, p, b):
        """
        Tính R_k[t] cho TẤT CẢ user, công thức (21).
        a, p, b: ma trận association/power/bandwidth đầy đủ, shape (num_uavs, num_users)
        Trả về: rates, shape (num_users,)
        """
        rates = np.zeros(self.num_users)

        for m in range(self.num_uavs):
            served_users = np.where(a[m, :] == 1)[0]     # user_indicies mà UAV m đang phục vụ
            if len(served_users) == 0:
                continue

            sinr = self._compute_SINR(p,b, m, served_users)      # gamma_{m,k}
            b_mk = b[m, served_users]                           # bandwidth tương ứng
            rates[served_users] = b_mk * np.log2(1 + sinr)      # a_{m,k}=1 đã đảm bảo qua served_users
        rates_mbps = rates / 1e6 
        return rates
    
    def _move_uav(self, agent_idx, mov_action):
        """
        Cập nhật vị trí UAV dựa trên action di chuyển (0-8).
        Trả về: speed thực tế (m/s) sau khi đã áp dụng ràng buộc biên (7).
        """
        is_hover = (mov_action == 8)

        direction_vec = self.direction_vectors[mov_action]   # (2,)
        displacement = direction_vec * self.V_step

        old_position = self.uav_position[agent_idx].copy()
        new_position = old_position + displacement
        new_position = np.clip(new_position, 0, self.L)      # ràng buộc (7) - KHÔNG nên bỏ

        self.uav_position[agent_idx] = new_position

        # Speed thực tế theo công thức (5), tính TỪ vị trí thực tế sau clip
        actual_displacement = np.linalg.norm(new_position - old_position)
        speed = actual_displacement / self.delta_T

        return speed
    
    def _transmit_energy(self,agent_idx,p): 
        """
        Formula 23
        E_tx = deltaT * sum_k p(m,k)t
        Args:
            agent_idx: agent index
            p: power matrix of all UAVs and all users
        Returns:
            E_tx: Total transmit energy of all user K transmit by UAV m
        """

        total_energy = np.sum(p[agent_idx,:])
        E_tx = total_energy * self.delta_T
        return E_tx

    def _propulsion_power(self, v):
        """
        Formula (24) - rotary-wing UAV propulsion power model.
        Args:
            v: UAV speed (m/s)
        Return:
            P_fly: Propulsion power (W)
        """
        # Số hạng 1: blade profile power 
        term1 = self.P0 * (1 + 3 * v**2 / self.Utip**2)

        # Số hạng 2: induced power 
        inner = np.sqrt(1 + v**4 / (4 * self.v0**4)) - v**2 / (2 * self.v0**2)
        term2 = self.Pi * np.sqrt(inner)

        # Số hạng 3: parasite/fuselage drag power 
        term3 = 0.5 * self.d0 * self.rho * self.s * self.A * v**3

        P_fly = term1 + term2 + term3
        return P_fly
    def _update_energy(self, a, p, speeds):
        """
        Cập nhật năng lượng cho TẤT CẢ UAV sau 1 slot.
        a, p: ma trận association/power đã tính 1 LẦN ở đầu step() (không tính lại ở đây)
        speeds: array shape (num_uavs,) - tốc độ v_m[t] của từng UAV, lấy từ _move_uav
        """
        for m in range(self.num_uavs):
            # Propulsion energy - công thức (24), (25)
            P_fly = self._propulsion_power(speeds[m])
            E_fly = P_fly * self.delta_T

            # Transmit energy - công thức (23)
            E_tx = self._transmit_energy(m, p)

            # Tổng năng lượng tiêu thụ slot này - công thức (22)
            E_total = E_fly + E_tx
            self.consumed_energy[m] = E_total
            # Cập nhật năng lượng còn lại - công thức (26)
            self.uav_energy[m] = max(0.0, self.uav_energy[m] - E_total)  # không cho âm
            #print("UAV Energy", self.uav_energy[m])

        return self.consumed_energy.copy() # Trả về năng lượng đã tiêu hao sử dụng
        
    def _compute_qos_violation(self, rates):
        """Formula (32): chỉ tính cho user khẩn cấp"""
        emg_rates = rates[self.is_emergency]
        psi_qos = np.maximum(self.R_min_emg - emg_rates, 0.0)
        return psi_qos  # shape (num_emg_users,)

    def _compute_collision_violation(self):
        """Formula (33): với mọi cặp UAV m != j"""
        psi_col = 0.0
        for m in range(self.num_uavs):
            for j in range(m + 1, self.num_uavs):   # chỉ tính mỗi cặp 1 lần, x2 sau
                dist = np.linalg.norm(self.uav_position[m] - self.uav_position[j])
                psi_col += 2 * max(self.d_min - dist, 0.0)  # x2 vì sum m!=j đếm cả (m,j) và (j,m)
        return psi_col

    def _compute_battery_violation(self):
        """Formula (34)"""
        psi_bat = np.maximum(self.E_min - self.uav_energy, 0.0)
        return psi_bat  # shape (num_uavs,)
    
    def _compute_reward(self, a, rates, E_consumed, psi_qos, psi_col, psi_bat):
        
        rates_mbps = rates / 1e6  # mbps to kbps

        U_R = np.sum(self.user_priority * rates_mbps)  
        U_E = np.sum(E_consumed) / 1e3 # j to Kj
 
        penalty_qos = self.lambda_Q * np.sum(psi_qos)
        penalty_col = self.lambda_C * psi_col
        penalty_bat = self.lambda_B * np.sum(psi_bat)

        r = U_R - self.lambda_E * U_E - penalty_qos - penalty_col - penalty_bat
        return r

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
        speeds = np.zeros(self.num_uavs)
        for agent in self.agents:
            agent_idx = self.agent_name_to_idx[agent]
            mov_action = actions[agent][0]
            speeds[agent_idx] = self._move_uav(agent_idx, mov_action)   
        a, p, b = self._resolve_association_and_resources(actions)   # gọi 1 LẦN duy nhất
        self.a, self.p, self.b = a, p, b
        E_consumed = self._update_energy(a, p, speeds)   # dùng lại a, p đã tính, không tính lại

        rates = self._compute_achievable_rate(a, p, b)

        psi_qos = self._compute_qos_violation(rates)             # (32)
        psi_col = self._compute_collision_violation()             # (33)
        psi_bat = self._compute_battery_violation( )                # (34)

        observations = {agent: self._get_local_obs(agent) for agent in self.agents}
        
        r_shared = self._compute_reward(a, rates, E_consumed, psi_qos, psi_col, psi_bat)  # (35)
        self._cumulative_reward += r_shared
        # Trả về phần thưởng chung hoặc riêng tùy định nghĩa (Thường trong CTDE là chung)
        rewards = {agent: r_shared for agent in self.agents} 
        self.current_step += 1
        terminations = {}
        truncations = {}
        for agent in self.agents:
            agent_idx = self.agent_name_to_idx[agent]
            terminations[agent] = bool(self.uav_energy[agent_idx] <= self.E_min)
            truncations[agent] = bool(self.current_step >= self.time_slot)

        infos = {agent: {
        "psi_qos_sum": float(np.sum(psi_qos)),
        "psi_col": float(psi_col),
        "psi_bat_sum": float(np.sum(psi_bat)),
        } for agent in self.agents}
        self.agents = [
            agent for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos



    def state(self):
        close_gains = np.array([
            self._compute_channel_gain(0, np.arange(self.num_users))
        ]).flatten()  # ví dụ đơn giản, bạn cần quyết định channel gain tính theo UAV nào / trung bình ra sao

        global_state = np.concatenate([
            self.uav_position.flatten() / self.L,      # (num_uavs*2,)
            self.uav_energy / self.E_max,               # (num_uavs,)
            self.user_positions.flatten() / self.L,     # (num_users*2,)
            self.user_priority / 5.0,                    # (num_users,)
            close_gains                                  # (num_users,) - cần định nghĩa lại cho hợp lý
        ])
        return global_state.astype(np.float32)
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from uav_render import UAVRenderer
            self._renderer = UAVRenderer(self.L, self.render_mode)
        return self._renderer.render_frame(
            self.uav_position, self.uav_energy, self.E_max, self.E_min,
            self.user_positions, self.user_priority, self.is_emergency,
            self.a, self.current_step, self._cumulative_reward, self.agents, self.num_uavs
    )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":

    # env = UAVEmergencyEnv(render_mode="human")
    # obs, infos = env.reset(seed=42)

    # print("Agents:", env.agents)
    # print("Number of agents:", len(env.agents))
    # print()

    # for agent, ob in obs.items():
    #     print(agent, ob.shape)

    # for agent in env.agents:
    #     obs = env._get_local_obs(agent)

    #     print(agent)
    #     print(obs.shape)
    #     print(env.observation_space(agent).shape)

    #     assert env.observation_space(agent).contains(obs.astype(np.float32))


    # for step in range(env.time_slot):
    #     if not env.agents:
    #         break
    #     actions = {
    #         agent: env.action_space(agent).sample()
    #         for agent in env.agents
    #     }

    #     obs, rewards, terminations, truncations, infos = env.step(actions)

    #     print(f"\n========== STEP {step} ==========")

    #     for agent in env.agents:
    #         print(f"\n{agent}")
    #         print("Action :", actions[agent])
    #         print("Reward :", rewards[agent])
    #         print("Obs :", obs[agent])
    #         print("Obs shape :", obs[agent].shape)
    #         print("Global Observation", env.state())
    env = UAVEmergencyEnv(render_mode="human")
    obs, infos = env.reset(seed=42)
    for _ in range(100):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        env.render()   # cửa sổ matplotlib phải update liên tục, không mở cửa sổ mới mỗi lần
    env.close()
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Import môi trường và mạng Actor của bạn
from uav_env import UAVEmergencyEnv
from train_mappo import Actor

def run_inference_episode(env_seed, model_path, mode="deterministic", device="cpu"):
    """
    Chạy 1 episode và trả về mảng phần thưởng tích lũy qua từng step.
    mode: "deterministic" (argmax), "stochastic_policy" (sample), hoặc "random" (hành động ngẫu nhiên)
    """
    env = UAVEmergencyEnv(render_mode=None) # Tắt đồ họa để chạy test cho nhanh
    obs, infos = env.reset(seed=env_seed)
    
    # Khởi tạo và load model nếu không phải là random baseline
    actor = None
    if mode != "random":
        obs_dim = env.observation_space("uav_0").shape[0]
        actor = Actor(obs_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()

    cumulative_rewards = []
    current_cumulative = 0.0

    for step in range(env.time_slot):
        if not env.agents:
            # Nếu tất cả UAV đã chết, phần thưởng các bước sau giữ nguyên
            cumulative_rewards.append(current_cumulative)
            continue
            
        actions_dict = {}
        
        with torch.no_grad():
            for agent in env.agents:
                if mode == "random":
                    # Lấy hành động hoàn toàn ngẫu nhiên từ không gian hành động
                    actions_dict[agent] = env.action_space(agent).sample()
                else:
                    obs_array = obs[agent]
                    obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
                    d_move, d_srv, d_pow = actor.forward(obs_tensor)
                    
                    if mode == "deterministic":
                        a_move = torch.argmax(d_move.logits, dim=-1).item()
                        a_srv = torch.argmax(d_srv.logits, dim=-1).item()
                        a_pow = torch.argmax(d_pow.logits, dim=-1).item()
                    elif mode == "stochastic_policy":
                        a_move = d_move.sample().item()
                        a_srv = d_srv.sample().item()
                        a_pow = d_pow.sample().item()
                        
                    actions_dict[agent] = np.array([a_move, a_srv, a_pow])
        
        next_obs, rewards, terms, truncs, infos = env.step(actions_dict)
        
        # Lấy phần thưởng của step này (Tính trung bình của các agent còn sống)
        step_reward = np.mean([rewards[a] for a in env.agents]) if env.agents else 0.0
        current_cumulative += step_reward
        cumulative_rewards.append(current_cumulative)
        
        obs = next_obs
        
    env.close()
    return cumulative_rewards

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    TEST_SEEDS = [42, 100, 2024] 
    
    print("Đang đánh giá: MAPPO (Reward Shaping)...")
    shaping_results = [run_inference_episode(seed, "mappo_checkpoint_rewardshaping.pt", "deterministic", DEVICE) for seed in TEST_SEEDS]
    shaping_mean = np.mean(shaping_results, axis=0)

    print("Đang đánh giá: MAPPO (Raw)...")
    raw_results = [run_inference_episode(seed, "mappo_checkpoint_raw.pt", "deterministic", DEVICE) for seed in TEST_SEEDS]
    raw_mean = np.mean(raw_results, axis=0)

    print("Đang đánh giá: Random Baseline...")
    random_results = [run_inference_episode(seed, None, "random", DEVICE) for seed in TEST_SEEDS]
    random_mean = np.mean(random_results, axis=0)

    print("Đang tạo biểu đồ...")
    plt.figure(figsize=(10, 6))
    
    steps = np.arange(len(shaping_mean))
    
    plt.plot(steps, shaping_mean, label="MAPPO (Proposed PBRS)", color="blue", linewidth=2.5)
    plt.plot(steps, raw_mean, label="MAPPO (Baseline)", color="orange", linewidth=2.5, linestyle="--")
    plt.plot(steps, random_mean, label="Random Allocation", color="green", linewidth=2.5, linestyle=":")
    
    plt.title("Cumulative Reward Comparison during Inference", fontsize=16, fontweight="bold")
    plt.xlabel("Time Step ($t$)", fontsize=14)
    plt.ylabel("Cumulative Episode Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12, loc="upper left")
    
    plt.xlim(0, len(shaping_mean) - 1)
    
    plt.tight_layout()
    plt.savefig("model_comparison_eval.png", dpi=300) 
    plt.show()
    
    print("Hoàn thành! Đã lưu biểu đồ thành 'model_comparison_eval.png'")

if __name__ == "__main__":
    main()
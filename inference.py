import torch
import numpy as np
import time


from uav_env import UAVEmergencyEnv
from train_mappo import Actor

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    env = UAVEmergencyEnv(render_mode="human")
    obs, infos = env.reset()
    
    obs_dim = env.observation_space("uav_0").shape[0]
    actor = Actor(obs_dim).to(DEVICE)
    
    # 3. Load Weights file .pt
    print("Đang tải trọng số mô hình từ mappo_checkpoint.pt...")
    checkpoint = torch.load("mappo_checkpoint_rewardshaping.pt", map_location=DEVICE)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval() 


    total_reward = 0.0
    print("Bắt đầu Inference...")
    
    for step in range(env.time_slot):

        if not env.agents:
            break
            
        actions_dict = {}
        

        with torch.no_grad():
            for agent in env.agents:
                obs_array = obs[agent]
                obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=DEVICE)
                

                d_move, d_srv, d_pow = actor.forward(obs_tensor)
                
                a_move = torch.argmax(d_move.logits, dim=-1).item()
                a_srv = torch.argmax(d_srv.logits, dim=-1).item()
                a_pow = torch.argmax(d_pow.logits, dim=-1).item()
                
                actions_dict[agent] = np.array([a_move, a_srv, a_pow])
        
        next_obs, rewards, terms, truncs, infos = env.step(actions_dict)
        
        env.render()
        time.sleep(0.1) 
        
        if env.agents:
            total_reward += rewards[env.agents[0]]
            
        obs = next_obs
        
    print(f"\n[HOÀN THÀNH] Tổng phần thưởng (Reward) đạt được: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
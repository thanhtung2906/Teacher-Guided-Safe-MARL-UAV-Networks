import torch
from uav_env import UAVEmergencyEnv
from train_mappo import Actor

env = UAVEmergencyEnv()
obs, _ = env.reset(seed=1)
actor = Actor(env.observation_space("uav_0").shape[0])
actor.load_state_dict(torch.load("mappo_checkpoint.pt")["actor"])
actor.eval()

for agent in env.agents:
    o = torch.as_tensor(obs[agent], dtype=torch.float32)
    d_move, d_srv, d_pow = actor.forward(o)
    print(agent, "move logits:", d_move.logits.detach().numpy())
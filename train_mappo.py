"""
MAPPO tối giản - 1 process (không SuperSuit), centralized critic + 3-head actor.
Mục tiêu: có kết quả chạy được để nộp, KHÔNG kỳ vọng hội tụ tốt trong thời gian ngắn.

Chạy: python train_mappo.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uav_env import UAVEmergencyEnv


# ---------------- Hyperparameters ----------------
SEED = 0
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
EPOCHS_PER_UPDATE = 10
MINIBATCH_SIZE = 256
NUM_UPDATES = 200       # tăng lên nếu còn thời gian, giảm nếu cần kết quả gấp
MAX_GRAD_NORM = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Decentralized, tham số dùng chung cho mọi UAV (parameter sharing)."""
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.head_move = layer_init(nn.Linear(hidden, 9), std=0.01)
        self.head_srv = layer_init(nn.Linear(hidden, 4), std=0.01)
        self.head_pow = layer_init(nn.Linear(hidden, 3), std=0.01)

    def forward(self, obs):
        h = self.trunk(obs)
        return (Categorical(logits=self.head_move(h)),
                Categorical(logits=self.head_srv(h)),
                Categorical(logits=self.head_pow(h)))

    def act(self, obs, action=None):
        d_move, d_srv, d_pow = self.forward(obs)
        if action is None:
            a_move, a_srv, a_pow = d_move.sample(), d_srv.sample(), d_pow.sample()
        else:
            a_move, a_srv, a_pow = action[..., 0], action[..., 1], action[..., 2]
        logprob = d_move.log_prob(a_move) + d_srv.log_prob(a_srv) + d_pow.log_prob(a_pow)
        entropy = d_move.entropy() + d_srv.entropy() + d_pow.entropy()
        action_out = torch.stack([a_move, a_srv, a_pow], dim=-1)
        return action_out, logprob, entropy


class Critic(nn.Module):
    """Centralized, input = global state() của toàn hệ thống."""
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def compute_gae(rewards, values, dones, gamma, lam):
    """rewards, values, dones: 1D array shape (T,). values phải có T+1 phần tử (bootstrap)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns


def collect_rollout(env, actor, critic):
    """Chạy 1 episode đầy đủ (T = env.time_slot), trả về buffer numpy đã pad+mask agent chết."""
    T, M = env.time_slot, env.num_uavs
    obs_dim = env.observation_space("uav_0").shape[0]
    state_dim = env.state_space.shape[0]

    obs_buf = np.zeros((T, M, obs_dim), dtype=np.float32)
    state_buf = np.zeros((T, state_dim), dtype=np.float32)
    action_buf = np.zeros((T, M, 3), dtype=np.int64)
    logprob_buf = np.zeros((T, M), dtype=np.float32)
    value_buf = np.zeros((T + 1,), dtype=np.float32)   # +1 để bootstrap GAE
    reward_buf = np.zeros((T,), dtype=np.float32)
    mask_buf = np.zeros((T, M), dtype=np.float32)
    done_buf = np.zeros((T,), dtype=np.float32)

    info_log = {"U_R_proxy": [], "psi_qos_sum": [], "psi_col": [], "psi_bat_sum": []}

    obs, infos = env.reset()
    last_obs = {a: obs[a] for a in env.possible_agents}
    alive = {a: True for a in env.possible_agents}

    n_steps = T
    for t in range(T):
        state_t = env.state()
        state_buf[t] = state_t

        obs_t = np.stack([last_obs[a] for a in env.possible_agents])  # (M, obs_dim)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=DEVICE)
            action_out, logprob, _ = actor.act(obs_tensor)
            value = critic(torch.as_tensor(state_t, dtype=torch.float32, device=DEVICE).unsqueeze(0))

        action_np = action_out.cpu().numpy()
        obs_buf[t] = obs_t
        action_buf[t] = action_np
        logprob_buf[t] = logprob.cpu().numpy()
        value_buf[t] = value.item()

        alive_agents = [a for a in env.possible_agents if alive[a]]
        if len(alive_agents) == 0:
            done_buf[t] = 1.0
            n_steps = t
            break

        for i, a in enumerate(env.possible_agents):
            mask_buf[t, i] = 1.0 if alive[a] else 0.0

        actions_dict = {a: action_np[env.agent_name_to_idx[a]] for a in alive_agents}
        next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

        reward_buf[t] = rewards[alive_agents[0]]
        for k, v in infos[alive_agents[0]].items():
            info_log.setdefault(k, []).append(v)

        for a in alive_agents:
            last_obs[a] = next_obs[a]
            if terms[a] or truncs[a]:
                alive[a] = False

        if len(env.agents) == 0:
            done_buf[t] = 1.0
            n_steps = t + 1
            break

    with torch.no_grad():
        final_state = torch.as_tensor(env.state(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        value_buf[n_steps] = critic(final_state).item() if n_steps < T else 0.0

    return {
        "obs": obs_buf[:n_steps], "state": state_buf[:n_steps], "action": action_buf[:n_steps],
        "logprob": logprob_buf[:n_steps], "value": value_buf[:n_steps + 1],
        "reward": reward_buf[:n_steps], "mask": mask_buf[:n_steps], "done": done_buf[:n_steps],
        "n_steps": n_steps, "info_log": info_log,
    }


def ppo_update(actor, critic, opt, rollout):
    obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=DEVICE)          # (T,M,obs_dim)
    state = torch.as_tensor(rollout["state"], dtype=torch.float32, device=DEVICE)      # (T,state_dim)
    action = torch.as_tensor(rollout["action"], dtype=torch.long, device=DEVICE)       # (T,M,3)
    old_logprob = torch.as_tensor(rollout["logprob"], dtype=torch.float32, device=DEVICE)  # (T,M)
    mask = torch.as_tensor(rollout["mask"], dtype=torch.float32, device=DEVICE)        # (T,M)

    advantages, returns = compute_gae(rollout["reward"], rollout["value"], rollout["done"], GAMMA, GAE_LAMBDA)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)       # (T,)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)             # (T,)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T, M = obs.shape[0], obs.shape[1]
    adv_broadcast = advantages.unsqueeze(-1).expand(-1, M)   # (T,M)

    idx = np.arange(T)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}

    for _ in range(EPOCHS_PER_UPDATE):
        np.random.shuffle(idx)
        for start in range(0, T, MINIBATCH_SIZE):
            mb_idx = idx[start:start + MINIBATCH_SIZE]
            mb_obs = obs[mb_idx].reshape(-1, obs.shape[-1])            # (mb*M, obs_dim)
            mb_action = action[mb_idx].reshape(-1, 3)
            mb_old_logprob = old_logprob[mb_idx].reshape(-1)
            mb_mask = mask[mb_idx].reshape(-1)
            mb_adv = adv_broadcast[mb_idx].reshape(-1)

            _, new_logprob, entropy = actor.act(mb_obs, action=mb_action)
            ratio = torch.exp(new_logprob - mb_old_logprob)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            policy_loss = -(torch.min(surr1, surr2) * mb_mask).sum() / mb_mask.sum().clamp(min=1)
            entropy_loss = -(entropy * mb_mask).sum() / mb_mask.sum().clamp(min=1)

            mb_state = state[mb_idx]
            mb_returns = returns[mb_idx]
            new_value = critic(mb_state)
            value_loss = ((new_value - mb_returns) ** 2).mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM)
            opt.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                clip_frac = ((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(-entropy_loss.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)

    return {k: float(np.mean(v)) for k, v in stats.items()}


def main():
    env = UAVEmergencyEnv() 
    obs_dim = env.observation_space("uav_0").shape[0]
    state_dim = env.state_space.shape[0]

    # --- Sanity assertion: bắt lỗi shape ngay lập tức thay vì để lỗi âm thầm giữa training ---
    obs0, _ = env.reset(seed=SEED)
    assert obs0["uav_0"].shape[0] == obs_dim, "obs_dim mismatch - kiểm tra lại observation_space"
    assert env.state().shape[0] == state_dim, "state_dim mismatch - kiểm tra lại state_space"
    print(f"[OK] obs_dim={obs_dim}, state_dim={state_dim}, device={DEVICE}")

    actor = Actor(obs_dim).to(DEVICE)
    critic = Critic(state_dim).to(DEVICE)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    history = {"episode_reward": [], "U_R_mean": [], "psi_qos_mean": [], "psi_col_mean": [],
               "psi_bat_mean": [], "policy_loss": [], "value_loss": [], "entropy": [],
               "approx_kl": [], "clip_frac": [], "n_steps": []}

    for update in range(1, NUM_UPDATES + 1):
        rollout = collect_rollout(env, actor, critic)
        stats = ppo_update(actor, critic, opt, rollout)

        ep_reward = float(rollout["reward"].sum())
        info_log = rollout["info_log"]
        history["episode_reward"].append(ep_reward)
        history["psi_qos_mean"].append(float(np.mean(info_log.get("psi_qos_sum", [0]))))
        history["psi_col_mean"].append(float(np.mean(info_log.get("psi_col", [0]))))
        history["psi_bat_mean"].append(float(np.mean(info_log.get("psi_bat_sum", [0]))))
        history["n_steps"].append(rollout["n_steps"])
        for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
            history[k].append(stats[k])

        print(f"update {update:3d} | n_steps={rollout['n_steps']:3d} | "
              f"reward={ep_reward:10.2f} | policy_loss={stats['policy_loss']:.4f} | "
              f"value_loss={stats['value_loss']:.4f} | entropy={stats['entropy']:.3f} | "
              f"approx_kl={stats['approx_kl']:.4f} | psi_qos={history['psi_qos_mean'][-1]:.3f}")

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, "mappo_checkpoint.pt")

    # --- Plots để nộp báo cáo ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(history["episode_reward"]); axes[0, 0].set_title("Episode reward")
    axes[0, 1].plot(history["policy_loss"]); axes[0, 1].set_title("Policy loss")
    axes[0, 2].plot(history["value_loss"]); axes[0, 2].set_title("Value loss")
    axes[1, 0].plot(history["entropy"]); axes[1, 0].set_title("Entropy")
    axes[1, 1].plot(history["psi_qos_mean"], label="QoS"); axes[1, 1].plot(history["psi_col_mean"], label="Collision")
    axes[1, 1].plot(history["psi_bat_mean"], label="Battery"); axes[1, 1].set_title("Safety violations"); axes[1, 1].legend()
    axes[1, 2].plot(history["n_steps"]); axes[1, 2].set_title("Episode length (steps sống sót)")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print("\nĐã lưu: mappo_checkpoint.pt, training_curves.png")


if __name__ == "__main__":
    main()
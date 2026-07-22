import numpy as np
from collections import deque


class UAVRenderer:
    """Renderer tach rieng khoi env. Lazy-init matplotlib, reuse 1 figure."""

    def __init__(self, L, render_mode="human", num_uavs=3, trail_length=100):
        self.L = L
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.plt = None
        # --- Trajectory trail: luu lich su vi tri gan nhat cua tung UAV ---
        self.trail_length = trail_length
        self.trails = [deque(maxlen=trail_length) for _ in range(num_uavs)]

    def _ensure_init(self):
        if self.fig is not None:
            return
        import matplotlib
        if self.render_mode == "human":
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass
        else:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        if self.render_mode == "human":
            plt.ion()

    def reset_trail(self):
        """Goi ham nay moi khi env.reset() de khong dinh vet cua episode truoc."""
        for trail in self.trails:
            trail.clear()

    def render_frame(self, uav_position, uav_energy, E_max, E_min,
                      user_positions, user_priority, is_emergency,
                      association, step, cumulative_reward, active_agents, num_uavs):
        self._ensure_init()

        # Neu so UAV thay doi (khong nen xay ra, nhung phong ho) -> resize trails
        if len(self.trails) != num_uavs:
            self.trails = [deque(maxlen=self.trail_length) for _ in range(num_uavs)]

        self.ax.clear()
        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(0, self.L)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"step {step} | cum reward: {cumulative_reward:.1f}")

        # --- Users: kich thuoc theo priority, vien do neu emergency ---
        sizes = 20 + user_priority * 15
        edge_colors = ["red" if e else "none" for e in is_emergency]
        self.ax.scatter(user_positions[:, 0], user_positions[:, 1],
                         s=sizes, c="steelblue", alpha=0.6,
                         edgecolors=edge_colors, linewidths=1.3, label="Users")

        # --- Association line UAV -> served users ---
        for m in range(num_uavs):
            served = np.where(association[m] == 1)[0]
            for k in served:
                self.ax.plot([uav_position[m, 0], user_positions[k, 0]],
                              [uav_position[m, 1], user_positions[k, 1]],
                              color="gray", linewidth=0.6, alpha=0.5, zorder=1)

        uav_colors = ["green", "orange", "purple", "brown", "teal"]

        # --- Cap nhat trail: them vi tri hien tai vao lich su cua tung UAV ---
        for m in range(num_uavs):
            alive = f"uav_{m}" in active_agents
            if alive:
                self.trails[m].append(uav_position[m].copy())
            # Neu UAV da chet, KHONG them diem moi -> trail dung lai dung cho vi tri cuoi

        # --- Ve trail: mo dan theo thoi gian (cang cu cang mo), cung mau UAV ---
        for m in range(num_uavs):
            trail = self.trails[m]
            if len(trail) < 2:
                continue
            c = uav_colors[m % len(uav_colors)]
            n = len(trail)
            for i in range(n - 1):
                # alpha tang dan tu diem cu (mo) den diem moi (net hon)
                alpha =  0.5 + 0.5 * (i / max(n - 1, 1)) #0.08
                self.ax.plot(
                    [trail[i][0], trail[i + 1][0]],
                    [trail[i][1], trail[i + 1][1]],
                    color=c, linewidth=2.0, alpha=alpha, zorder=1.5,
                    solid_capstyle="round",
                )

        # --- UAV + energy bar ---
        for m in range(num_uavs):
            alive = f"uav_{m}" in active_agents
            c = uav_colors[m % len(uav_colors)]
            self.ax.scatter(uav_position[m, 0], uav_position[m, 1],
                             marker="^", s=220, c=c,
                             alpha=1.0 if alive else 0.25,
                             edgecolors="black", linewidths=1.2, zorder=3,
                             label=f"uav_{m}" + ("" if alive else " (dead)"))
            e_frac = np.clip(uav_energy[m] / E_max, 0, 1)
            bar_color = "green" if uav_energy[m] > E_min else "red"
            x0, y0 = uav_position[m, 0] - 30, uav_position[m, 1] + 40
            self.ax.plot([x0, x0], [y0, y0], color="lightgray", linewidth=4, zorder=2)
            self.ax.plot([x0, x0 + 60 * e_frac], [y0, y0], color=bar_color, linewidth=4, zorder=3)

        self.ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

        if self.render_mode == "human":
            self.plt.pause(0.01)
            return None
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = self.fig.canvas.get_width_height()
            return buf.reshape(h, w, 4)[:, :, :3]

    def close(self):
        if self.fig is not None:
            self.plt.close(self.fig)
            self.fig, self.ax = None, None
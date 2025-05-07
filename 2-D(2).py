# 1D conduction-based 2D visualization with grouped frames for faster animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# Physical parameters
INCH = 0.0254           # m per inch
height = 4 * INCH  # clip height in m
width = 0.2 * INCH            # display width in m
rho = 3510              # kg/m³
cp = 515                # J/kg·K
kappa = 2000            # W/m·K
alpha = kappa / (rho * cp)

T_amb = 27.0            # °C
T_max = 500.0           # °C
total_time = 4.0        # s

# Discretization in height (1D conduction)
dy = 0.0005             # 0.5 mm
Ny = int(height / dy) + 1

# Horizontal replication dimension
Nx = 100
width_display = width

# Stability-limited time step
dt_max = dy**2 / (2 * alpha)
dt = 0.8 * dt_max
n_steps = int(total_time / dt)
coeff = alpha * dt / dy**2

# Determine frame grouping
fps = 25
target_frames = int(total_time * fps)
group_n = max(1, n_steps // target_frames)
frames = list(range(0, n_steps, group_n))

print(f"dt={dt:.2e}s, total steps={n_steps}, group_n={group_n}, frames={len(frames)}")

# Initialize 1D temperature
T1d = np.full(Ny, T_amb)
T_buf = np.empty_like(T1d)

# Prepare mesh for imshow
y = np.linspace(0, height, Ny)
x = np.linspace(0, width_display, Nx)
X, Y = np.meshgrid(x, y)

# Setup plot
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(
    np.tile(T1d[:, None], (1, Nx)),
    origin='lower',
    cmap='plasma',
    vmin=T_amb,
    vmax=T_max,
    extent=[0, width_display, 0, height],
    aspect='auto'
)
cbar = fig.colorbar(im, ax=ax, label='Temperature (°C)')
cbar.ax.invert_yaxis()
ax.set_xlabel('Width (m)')
ax.set_ylabel('height (m)')

def update(step):
    # step is the simulation index
    # run group_n updates per displayed frame
    for _ in range(group_n):
        # bottom Dirichlet ramp
        time_sim = update.counter * dt
        T_buf[:] = T1d
        T_buf[0] = T_amb + (T_max - T_amb) * (time_sim / total_time)
        # interior update
        T_buf[1:-1] = T1d[1:-1] + coeff * (T1d[2:] + T1d[:-2] - 2*T1d[1:-1])
        # insulated top
        T_buf[-1] = T_buf[-2]
        T1d[:] = T_buf
        update.counter += 1
    # update display
    im.set_data(np.tile(T1d[:, None], (1, Nx)))
    ax.set_title(f't = {min(time_sim, total_time):.2f} s')
    return [im]

update.counter = 0

# Create and save animation
ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
output_path = Path.cwd() / "thermal_clip_fast_fixed.gif"
ani.save(output_path, writer='pillow', fps=fps)
plt.close(fig)
print("✅ Saved GIF ➜", output_path.resolve())
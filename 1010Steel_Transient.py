import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# --------------------------
# Transient Solver for AISI 1010 Steel Thermal Clip
# --------------------------

# Physical parameters
INCH      = 0.0254            # meters per inch
thickness = 1.5 * INCH        # clip thickness (m)
width     = 2.0 * INCH        # clip width (m)
depth     = 4.0 * INCH        # clip depth (m)

# Steel (AISI 1010) properties
rho       = 7832              # kg/m³
cp        = 434               # J/(kg·K)
kappa     = 63.9              # W/(m·K)

# Fixed energy input and ambient
Q_stored  = 407000            # J
T_amb     = 27.0              # °C (start)

# Compute mass and final ΔT
volume    = width * thickness * depth
mass      = rho * volume
deltaT    = Q_stored / (mass * cp)
T_max     = T_amb + deltaT    # ultimate surface temp if you went full 4 s

# We know surface hits 500 °C at ~3.10 s
t_cross   = 3.10              # seconds to stop

total_time = 4.0              # used in ramp calculation

# --------------------------
# Discretization & time step
# --------------------------
dy       = 0.0005             # m
Ny       = int(thickness / dy) + 1
Nx       = 100                # display resolution

alpha    = kappa / (rho * cp)
dt       = 0.8 * (dy**2 / (2 * alpha))
n_steps  = int(total_time / dt)
coeff    = alpha * dt / dy**2

# Build frame list up to t_cross
fps           = 25
group_n       = max(1, n_steps // (int(total_time * fps)))
frames        = [f for f in range(0, n_steps, group_n) if f * dt <= t_cross]

print(f"Stopping at t_cross = {t_cross}s → {len(frames)} frames")

# --------------------------
# Initialize temperature at ambient
# --------------------------
T1d   = np.full(Ny, T_amb)
T_buf = np.empty_like(T1d)

# Prepare mesh for display
y = np.linspace(0, thickness, Ny)
x = np.linspace(0, width, Nx)
X, Y = np.meshgrid(x, y)

# --------------------------
# Plot setup with colorbar capped at 500°C and inverted
# --------------------------
fig, ax = plt.subplots(figsize=(6,4))
im = ax.imshow(
    np.tile(T1d[:, None], (1, Nx)),
    origin='lower',
    cmap='plasma',
    vmin=T_amb,
    vmax=500.0,
    extent=[0, width, 0, thickness],
    aspect='auto'
)
cbar = fig.colorbar(im, ax=ax, label='Temperature (°C)')
ticks = np.linspace(T_amb, 500.0, 6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
cbar.ax.invert_yaxis()  # invert so 500°C (hot) is at bottom of colorbar

ax.set_xlabel('Width (m)')
ax.set_ylabel('Thickness (m)')

# init function
def init():
    im.set_data(np.tile(T1d[:, None], (1, Nx)))
    ax.set_title(f't = 0.00 s, T_surf = {T_amb:.1f} °C')
    return [im]

# update function
def update(step):
    global T1d, T_buf
    for _ in range(group_n):
        t = update.counter * dt
        # bottom Dirichlet ramp from ambient to T_max over 4 s
        T_buf[:]    = T1d
        T_buf[0]    = T_amb + (T_max - T_amb) * (t / total_time)
        # interior conduction
        T_buf[1:-1] = T1d[1:-1] + coeff * (
            T1d[2:] + T1d[:-2] - 2 * T1d[1:-1]
        )
        # insulated top
        T_buf[-1]   = T_buf[-2]
        T1d[:]      = T_buf
        update.counter += 1

    im.set_data(np.tile(T1d[:, None], (1, Nx)))
    ax.set_title(f't = {t:.2f} s, T_surf = {T1d[0]:.1f} °C')
    return [im]

update.counter = 0

# animate & save
ani = animation.FuncAnimation(
    fig, update,
    frames=frames,
    init_func=init,
    blit=False
)
out = Path.cwd() / "thermal_clip_steel_transient_stopped.gif"
ani.save(out, writer='pillow', fps=fps)
plt.close(fig)

print("✅ Saved GIF ➜", out.resolve())
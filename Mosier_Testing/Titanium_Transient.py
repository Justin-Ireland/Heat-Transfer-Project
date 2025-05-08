import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# --------------------------
# Physical and material parameters for titanium (Cartridge)
# --------------------------
INCH      = 0.0254            # meters per inch
thickness = 1.5 * INCH        # clip thickness (m)
width     = 2.0 * INCH        # clip width (m)

# titanium alloy properties
rho       = 4500              # kg/m³
cp        = 522               # J/(kg·K)
kappa     = 21.9               # W/(m·K)
mass      = 0.884             # kg

# Fixed energy input
Q_stored  = 407000            # J
T_amb     = 27.0              # °C ambient (start temperature)
deltaT    = Q_stored / (mass * cp)
T_max     = T_amb + deltaT    # °C final surface temp

total_time = 4.0              # seconds
t_cross    = total_time * (500 - T_amb) / (T_max - T_amb)

# --------------------------
# Discretization and time setup
# --------------------------
dy        = 0.0005            # m
Ny        = int(thickness / dy) + 1
Nx        = 100               # display resolution

alpha     = kappa / (rho * cp)
dt        = 0.8 * (dy**2 / (2 * alpha))
n_steps   = int(total_time / dt)
coeff     = alpha * dt / dy**2

fps           = 25
target_frames = int(total_time * fps)
group_n       = max(1, n_steps // target_frames)
frames        = [f for f in range(0, n_steps, group_n) if f * dt <= t_cross]

# --------------------------
# Initialize temperature profile at ambient
# --------------------------
T1d   = np.full(Ny, T_amb)
T_buf = np.empty_like(T1d)

# Prepare mesh for imshow
y = np.linspace(0, thickness, Ny)
x = np.linspace(0, width, Nx)
X, Y = np.meshgrid(x, y)

# --------------------------
# Plot setup with colorbar from 27°C to 500°C and inverted scale
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
cbar.ax.invert_yaxis()  # Invert colorbar so hot is at bottom

ax.set_xlabel('Width (m)')
ax.set_ylabel('Thickness (m)')

# Initialization function for FuncAnimation
def init():
    im.set_data(np.tile(T1d[:, None], (1, Nx)))
    ax.set_title(f't = 0.00 s, T_surf = {T_amb:.1f} °C')
    return [im]

# Animation update
def update(step):
    global T1d, T_buf
    for _ in range(group_n):
        t = update.counter * dt
        # bottom Dirichlet ramp from ambient
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

# Create and save animation (stops at 500°C)
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
output_path = Path.cwd() / "thermal_clip_titanium_inverted.gif"
ani.save(output_path, writer='pillow', fps=fps)
plt.close(fig)

print(f"✅ Saved GIF ➜ {output_path.resolve()}")

# --------------------------
# Simulation summary report
# --------------------------
print("\nSimulation summary:")
print(f" • Specified total_time          : {total_time:.2f} s")
print(f" • Computed maximum surface temp : {T_max:.1f} °C")
print(f" • Time to reach 500 °C (t_cross): {t_cross:.2f} s")
print(f" • Time step (dt)                : {dt:.6f} s")
print(f" • Total number of steps (n_steps): {n_steps}")
print(f" • Target frames (total_time×fps): {target_frames}")
print(f" • Actual frames generated       : {len(frames)}")
print(f" • Steps per frame (group_n)     : {group_n}")
print(f" • Final surface temperature     : {T1d[0]:.1f} °C")
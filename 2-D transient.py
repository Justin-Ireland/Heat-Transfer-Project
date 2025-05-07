# Re-run all necessary parts after reset to execute the full working script

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# --------------------------
# 2D Heat Equation Solver (Steady-State Setup)
# --------------------------

# Domain parameters (clip cross-section: width × thickness)
x_length = 0.09    # 9 cm
y_length = 0.0381  # 1.5 inches = 3.81 cm
delta_x = 0.005    # 5 mm grid spacing

# Material properties for synthetic diamond
kappa = 2000  # W/m-K
rho = 3510    # kg/m^3
cp = 515      # J/kg-K
h = 125       # W/m^2-K (convection coefficient)

T_amb = 27.0          # °C ambient
T_fixed_bottom = 500.0  # °C heated base

# Grid setup
x_points = round(x_length / delta_x + 1)
y_points = round(y_length / delta_x + 1)

x_range = np.linspace(0, x_length, x_points)
y_range = np.linspace(0, y_length, y_points)
X, Y = np.meshgrid(x_range, y_range)

# --------------------------
# Transient Simulation Setup
# --------------------------

# Initial temperature array
T_transient = np.full((y_points, x_points), T_amb)
T_transient[0, :] = T_fixed_bottom  # bottom boundary fixed temperature

# Thermal diffusivity
alpha = kappa / (rho * cp)

# Time-stepping setup
dt = 0.01  # time step in seconds
total_time = 4.0  # total time in seconds
n_steps = int(total_time / dt)

# Setup figure for animation
fig, ax = plt.subplots(figsize=(6, 4))
pcm = ax.pcolormesh(X, Y, np.flipud(T_transient), cmap="plasma", shading='auto', vmin=T_amb, vmax=T_fixed_bottom)
fig.colorbar(pcm, ax=ax, label="Temperature (°C)")
ax.set_title("Transient Temperature in Thermal Clip")
ax.set_xlabel("Width (m)")
ax.set_ylabel("Thickness (m)")

# Animation update function
def update(frame):
    T_new = T_transient.copy()
    for i in range(1, y_points - 1):
        for j in range(1, x_points - 1):
            T_new[i, j] = T_transient[i, j] + alpha * dt / delta_x**2 * (
                T_transient[i+1, j] + T_transient[i-1, j] +
                T_transient[i, j+1] + T_transient[i, j-1] - 4 * T_transient[i, j]
            )
    # Apply boundary conditions
    T_new[0, :] = T_fixed_bottom         # bottom: fixed high temp
    T_new[:, 0] = T_new[:, 1]            # left: insulated
    T_new[:, -1] = T_new[:, -2]          # right: insulated
    T_new[-1, :] = T_new[-2, :] + (h * delta_x / kappa) * (T_amb - T_transient[-1, :])  # top: convection

    pcm.set_array(np.flipud(T_new).ravel())
    ax.set_title(f"Transient Temperature (t = {frame * dt:.2f} s)")
    T_transient[:, :] = T_new
    return [pcm]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=40, blit=True)

# Save animation as GIF in same directory as script
output_path = Path.cwd() / "thermal_clip_transient.gif"
ani.save(output_path, writer="pillow", fps=25)

output_path.resolve()
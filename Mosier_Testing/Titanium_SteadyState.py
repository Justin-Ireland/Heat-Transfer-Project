import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------
# Steady-State 1D Conduction Solver for Titanium Thermal Clip
# --------------------------

# Physical parameters
INCH = 0.0254             # meters per inch
thickness = 1.5 * INCH    # clip thickness (m)
width = 2.0 * INCH        # clip width (m)

# Titanium properties (pure titanium, kappa ≈ 21.9 W/m·K)
kappa = 21.9              # W/(m·K) -- conduction only, no h required

# Boundary conditions
T_bottom = 500.0          # °C fixed at base
T_top = 27.0              # °C fixed at top (ambient)

# Discretization
dy = 0.0005
Ny = int(thickness / dy) + 1
y = np.linspace(0, thickness, Ny)

# Analytical steady-state 1D conduction solution (linear gradient)
L = thickness
T1d = T_bottom + (T_top - T_bottom) * (y / L)

# Create 2D field by replicating 1D profile across width
Nx = 200
x = np.linspace(0, width, Nx)
X, Y = np.meshgrid(x, y)
T2D = np.tile(T1d[:, None], (1, Nx))

# Temperature range for colormap
minT = T1d.min()
maxT = T1d.max()

# Plot steady-state temperature distribution
plt.figure(figsize=(6, 4))
pcm = plt.pcolormesh(X, Y, T2D, cmap='plasma', shading='auto',
                     vmin=minT, vmax=maxT)
cbar = plt.colorbar(pcm, label='Temperature (°C)')
ticks = np.linspace(minT, maxT, 6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
# Keep high temperature (500 °C) at bottom
cbar.ax.invert_yaxis()

plt.title("Steady-State Temperature Distribution (Titanium)")
plt.xlabel("Width (m)")
plt.ylabel("Thickness (m)")
plt.tight_layout()

# Save figure
output_path = Path.cwd() / "thermal_clip_titanium_steady_state_conduction.png"
plt.savefig(output_path)
plt.show()

print(f"✅ Steady-state conduction plot saved ➜ {output_path.resolve()}")

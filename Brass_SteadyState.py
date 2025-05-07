import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------
# Steady-State Solver for Brass Thermal Clip (Fixed BCs)
# --------------------------

# Physical parameters
INCH      = 0.0254            # meters per inch
thickness = 1.5 * INCH        # clip thickness (m)
width     = 2.0 * INCH        # clip width (m)

# Brass properties
kappa     = 110               # W/(m·K)
h         = 125               # W/(m²·K) convective coefficient

# Boundary conditions
T_bottom  = 500.0             # °C fixed at base
T_amb     = 27.0              # °C ambient at top

# Discretization
dy = 0.0005                   # grid spacing (m)
Ny = int(thickness / dy) + 1  # number of nodes in y
y = np.linspace(0, thickness, Ny)

# Analytical steady-state 1D solution
L = thickness
A = -h * (T_bottom - T_amb) / (kappa + h * L)
T1d = T_bottom + A * y      # Temperature profile from base (y=0) to top (y=L)

# Create 2D field by replicating 1D profile across width
Nx = 200
x = np.linspace(0, width, Nx)
X, Y = np.meshgrid(x, y)
T2D = np.tile(T1d[:, None], (1, Nx))

# Determine actual temperature range for colormap
minT = T1d.min()  # top temperature (~480 °C)
maxT = T_bottom   # bottom temperature (500 °C)

# Plot steady-state temperature distribution
plt.figure(figsize=(6, 4))
pcm = plt.pcolormesh(X, Y, T2D, cmap='plasma', shading='auto',
                     vmin=minT, vmax=maxT)
cbar = plt.colorbar(pcm, label='Temperature (°C)')
# Set colorbar ticks between top and bottom
ticks = np.linspace(minT, maxT, 6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
# **Invert the colorbar** so that 500 °C is at the bottom:
cbar.ax.invert_yaxis()

plt.title("Steady-State Temperature Distribution ")
plt.xlabel("Width (m)")
plt.ylabel("Thickness (m)")
plt.tight_layout()

# Save figure
output_path = Path.cwd() / "thermal_clip_brass_steady_state_gradient_flipped.png"
plt.savefig(output_path)
plt.show()

print(f"✅ Steady-state gradient plot saved ➜ {output_path.resolve()}")
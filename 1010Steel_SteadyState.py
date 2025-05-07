import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------
# Steady-State Solver for AISI 1010 Steel Thermal Clip (Base = 500 °C)
# --------------------------

# Physical parameters
INCH      = 0.0254            # meters per inch
thickness = 1.5 * INCH        # clip thickness (m)
width     = 2.0 * INCH        # clip width (m)

# Steel (AISI 1010) properties
kappa     = 63.9              # W/(m·K)
h         = 125               # W/(m²·K) convective coefficient
T_bottom  = 500.0             # °C fixed at base
T_amb     = 27.0              # °C ambient at top

# Discretization in y
dy = 0.0005                   # 0.5 mm grid spacing
Ny = int(thickness / dy) + 1  # number of nodes in y
y  = np.linspace(0, thickness, Ny)

# Analytical 1D steady-state solution
L = thickness
A = -h * (T_bottom - T_amb) / (kappa + h * L)
T1d = T_bottom + A * y       # Temperature profile from base to top

# Create 2D field by replicating 1D profile across width
Nx = 200
x  = np.linspace(0, width, Nx)
X, Y = np.meshgrid(x, y)
T2D = np.tile(T1d[:, None], (1, Nx))

# Determine actual temperature range for color scaling
minT = T1d.min()  # top temperature (~451 °C)
maxT = T_bottom   # base temperature (500 °C)

# Plot steady-state temperature distribution
plt.figure(figsize=(6, 4))
pcm = plt.pcolormesh(
    X, Y, T2D,
    cmap='plasma',
    shading='auto',
    vmin=minT, vmax=maxT       # use actual gradient range
)
cbar = plt.colorbar(pcm, label='Temperature (°C)')
# Set colorbar ticks between minT and maxT
ticks = np.linspace(minT, maxT, 6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
cbar.ax.invert_yaxis()  # hot (500 °C) at bottom of bar

plt.title("Steady-State Temperature Distribution\nAISI 1010 Steel Thermal Clip (Base = 500 °C)")
plt.xlabel("Width (m)")
plt.ylabel("Thickness (m)")
plt.tight_layout()

# Save figure
output_path = Path.cwd() / "steel_steady_state_capped_gradient.png"
plt.savefig(output_path)
plt.show()

print(f"✅ Steady-state gradient plot saved ➜ {output_path.resolve()}")
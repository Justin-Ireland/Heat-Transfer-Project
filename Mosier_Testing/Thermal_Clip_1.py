# ============================================================
# Thermal Clip Simulation — Program 1
# Synthetic Diamond 2D Transient Heat Conduction
# ------------------------------------------------------------
# Simulates heating from 27 °C to 500 °C with constant heat flux
# Stops automatically when the heated face reaches 500 °C
#
# Author: [Your Name]
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Physical and Simulation Parameters
# ------------------------------------------------------------

# Geometry (m)
width = 0.0508      # 2 inches
height = 0.1016     # 4 inches

# Grid
dx = dy = 0.001     # 1 mm grid spacing
Nx = int(width / dx)
Ny = int(height / dy)

# Material properties — Synthetic Diamond (locked values)
rho = 3510          # kg/m³
cp = 515            # J/kg·K
kappa = 2000        # W/m·K

# Thermal diffusivity
alpha = kappa / (rho * cp)

# Heat flux (W/m²)
q_flux = 54.78e6    # 24.88 MW/m²

# Initial and target temperatures (°C)
T_initial = 27.0
T_target = 500.0

# ------------------------------------------------------------
# 2. Initialize Temperature Field
# ------------------------------------------------------------

T = np.full((Ny, Nx), T_initial)

# ------------------------------------------------------------
# 3. Time Step (Stability Criterion)
# ------------------------------------------------------------

# Fourier number stability condition for explicit scheme:
# Fo = alpha * dt / dx² <= 0.25 for 2D
Fo_limit = 0.24
dt = Fo_limit * dx ** 2 / alpha

# ------------------------------------------------------------
# 4. Time Marching Loop
# ------------------------------------------------------------

time = 0.0
max_time = 10.0    # safety cap at 10 seconds

# Precompute constants
r = alpha * dt / dx ** 2

# Track temperature evolution
time_list = []
max_temp_list = []

while True:
    T_old = T.copy()

    # Update interior points
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            T[i, j] = T_old[i, j] + r * (
                T_old[i + 1, j] + T_old[i - 1, j] +
                T_old[i, j + 1] + T_old[i, j - 1] -
                4 * T_old[i, j]
            )

    # Apply boundary conditions

    # Bottom face (y = 0): constant heat flux
    T[0, :] = T_old[1, :] + (q_flux * dt) / (rho * cp * dy)

    # Other boundaries: insulated (Neumann BC => zero gradient)
    T[-1, :] = T[-2, :]       # Top
    T[:, 0] = T[:, 1]         # Left
    T[:, -1] = T[:, -2]       # Right

    # Update time
    time += dt

    # Record data
    time_list.append(time)
    max_temp_list.append(np.max(T))

    # Console output every 0.5 s simulated
    if len(time_list) % int(0.5 / dt) == 0:
        print(f"Time: {time:.2f} s | Max Temp: {np.max(T):.2f} °C")

    # Check stopping condition
    if np.max(T[0, :]) >= T_target or time >= max_time:
        break

# ------------------------------------------------------------
# 5. Results
# ------------------------------------------------------------

print("\nSimulation complete.")
print(f"Final time: {time:.3f} s")
print(f"Max temperature reached: {np.max(T):.2f} °C")

# Plot temperature history
plt.plot(time_list, max_temp_list)
plt.xlabel('Time (s)')
plt.ylabel('Max Temperature (°C)')
plt.title('Max Temperature vs Time — Synthetic Diamond')
plt.grid(True)
plt.show()

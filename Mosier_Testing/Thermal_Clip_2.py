# ============================================================
# Thermal Clip Simulation — Program 2
# Transient Simulation for Various Alloys
# ------------------------------------------------------------
# Simulates heating from 27 °C to 500 °C with constant heat flux
# for user-selected material.
# Stops when the heated face reaches 500 °C or time exceeds 10 s.
#
# Author: [Your Name]
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Geometry and Simulation Constants
# ------------------------------------------------------------

# Geometry (m)
width = 0.0508      # 2 inches
height = 0.1016     # 4 inches
depth = 0.0381      # 1.5 inches

# Grid
dx = dy = 0.001     # 1 mm grid spacing
Nx = int(width / dx)
Ny = int(height / dy)

# Initial and target temperatures (°C)
T_initial = 27.0
T_target = 500.0

# Heat flux (W/m²) calibrated from Program 1
q_flux = 8.785e6   # 54.785 MW/m²

# ------------------------------------------------------------
# 2. Material Properties Database
# ------------------------------------------------------------

materials = {
    "Steel (AISI 1010)": {"k": 63.9, "cp": 434, "rho": 7832},
    "Aluminum 2024-T6": {"k": 177, "cp": 875, "rho": 2770},
    "Durasteel": {"k": 25, "cp": 535, "rho": 8300},
    "Brass (Cartridge)": {"k": 110, "cp": 380, "rho": 8530},
    "Titanium": {"k": 21.9, "cp": 522, "rho": 4500},
    "Nickel": {"k": 90.7, "cp": 444, "rho": 8900}
}

# ------------------------------------------------------------
# 3. User Material Selection
# ------------------------------------------------------------

print("Available materials:")
for i, name in enumerate(materials.keys()):
    print(f"{i + 1}. {name}")

selection = int(input("Select a material by number: ")) - 1
material_name = list(materials.keys())[selection]
props = materials[material_name]

kappa = props["k"]
cp = props["cp"]
rho = props["rho"]

print(f"\nSelected material: {material_name}")
print(f"k = {kappa} W/m·K | cp = {cp} J/kg·K | rho = {rho} kg/m³")

# ------------------------------------------------------------
# 4. Mass and Energy Storage Capacity
# ------------------------------------------------------------

volume = width * height * depth
mass = rho * volume
delta_T = T_target - T_initial
Q_stored = mass * cp * delta_T

print(f"Mass = {mass:.3f} kg")
print(f"Energy stored (Q) = {Q_stored:.0f} J")

# ------------------------------------------------------------
# 5. Stability and Time Step
# ------------------------------------------------------------

alpha = kappa / (rho * cp)

Fo_limit = 0.24
dt = Fo_limit * dx ** 2 / alpha

# ------------------------------------------------------------
# 6. Initialize Temperature Field
# ------------------------------------------------------------

T = np.full((Ny, Nx), T_initial)

# Precompute constants
r = alpha * dt / dx ** 2

# ------------------------------------------------------------
# 7. Time Marching Loop
# ------------------------------------------------------------

time = 0.0
max_time = 10.0    # stop after 10 seconds if needed

time_list = []
max_temp_list = []

print("\nStarting transient simulation...")

while time < max_time:
    T_old = T.copy()

    # Update interior points
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            T[i, j] = T_old[i, j] + r * (
                T_old[i + 1, j] + T_old[i - 1, j] +
                T_old[i, j + 1] + T_old[i, j - 1] -
                4 * T_old[i, j]
            )

    # Bottom face (y = 0): constant heat flux
    T[0, :] = T_old[1, :] + (q_flux * dt) / (rho * cp * dy)

    # Other boundaries: insulated
    T[-1, :] = T[-2, :]       # Top
    T[:, 0] = T[:, 1]         # Left
    T[:, -1] = T[:, -2]       # Right

    time += dt
    time_list.append(time)
    max_temp = np.max(T)
    max_temp_list.append(max_temp)

    # Print progress every 0.5 simulated seconds
    if len(time_list) % int(0.5 / dt) == 0:
        print(f"Time: {time:.2f} s | Max Temp: {max_temp:.2f} °C")

    # Stop if target temperature is reached
    if max_temp >= T_target:
        break

# ------------------------------------------------------------
# 8. Results
# ------------------------------------------------------------

print("\nSimulation complete.")
print(f"Time to reach {T_target} °C: {time:.3f} s")
print(f"Energy stored (Q) = {Q_stored:.0f} J")

# Plot temperature history
plt.plot(time_list, max_temp_list)
plt.xlabel('Time (s)')
plt.ylabel('Max Temperature (°C)')
plt.title(f'Max Temperature vs Time — {material_name}')
plt.grid(True)
plt.show()

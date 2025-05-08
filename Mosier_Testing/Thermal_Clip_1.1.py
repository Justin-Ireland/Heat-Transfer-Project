# ============================================================
# Thermal Clip Simulation — Program 1 (Auto-Adjusting q_flux)
# Synthetic Diamond 2D Transient Heat Conduction
# ------------------------------------------------------------
# Solves for the q_flux needed to heat the bottom face to 500 °C
# in exactly 3.5 seconds
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

# Material properties — Synthetic Diamond
rho = 3510          # kg/m³
cp = 515            # J/kg·K
kappa = 2000        # W/m·K

# Thermal diffusivity
alpha = kappa / (rho * cp)

# Initial and target temperatures (°C)
T_initial = 27.0
T_target = 500.0

# Stability criterion — time step
Fo_limit = 0.24
dt = Fo_limit * dx ** 2 / alpha

# Simulation time
target_time = 3.5    # seconds

# ------------------------------------------------------------
# 2. Simulation Function
# ------------------------------------------------------------

def run_simulation(q_flux):
    """
    Runs the heat conduction simulation for a given heat flux.
    Returns the max temperature at 3.5 seconds.
    """

    T = np.full((Ny, Nx), T_initial)
    r = alpha * dt / dx ** 2

    time = 0.0

    while time < target_time:
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

        # Other boundaries insulated
        T[-1, :] = T[-2, :]       # Top
        T[:, 0] = T[:, 1]         # Left
        T[:, -1] = T[:, -2]       # Right

        time += dt

    return np.max(T)

# ------------------------------------------------------------
# 3. Root-Finding Loop (Secant Method)
# ------------------------------------------------------------

# Initial guesses for q_flux (W/m²)
q_low = 25e6      # too low
q_high = 75e6     # should overshoot

T_low = run_simulation(q_low)
T_high = run_simulation(q_high)

tolerance = 0.5    # °C
max_iterations = 20

print("\nStarting root-finding for q_flux...")

for iteration in range(max_iterations):
    # Secant method formula
    q_new = q_high - (T_high - T_target) * (q_high - q_low) / (T_high - T_low)

    T_new = run_simulation(q_new)

    print(f"Iteration {iteration + 1}: q_flux = {q_new/1e6:.2f} MW/m² | Max Temp = {T_new:.2f} °C")

    if abs(T_new - T_target) < tolerance:
        print("\nTarget temperature reached within tolerance.")
        break

    # Update guesses for next iteration
    q_low, T_low = q_high, T_high
    q_high, T_high = q_new, T_new

else:
    print("\nWarning: Maximum iterations reached without convergence.")

# ------------------------------------------------------------
# 4. Final Results
# ------------------------------------------------------------

print(f"\nFinal q_flux needed: {q_new/1e6:.3f} MW/m²")
print(f"Achieved temperature at 3.5 s: {T_new:.2f} °C")


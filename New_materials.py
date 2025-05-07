"""
Diamond_Clip.py

2D transient heat conduction in a sci-fi barrel (or similar part).
Explicit finite difference in cylindrical r–z coordinates.
Modified to apply a *constant heat flux* to allow material comparisons.

Author : <your name>
Date   : 06-May-2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MATERIALS and constant heat flux setting
# ─────────────────────────────────────────────────────────────────────────────

materials = {
    'Ti-6Al-4V': {'rho': 4420, 'cp': 560, 'k': 7.2},
    'Steel 4340': {'rho': 7850, 'cp': 475, 'k': 44.5},
    'Al 6061-T6': {'rho': 2700, 'cp': 896, 'k': 167},
}

material_name = 'Ti-6Al-4V'   # <<< Change this for different runs

# Grab properties
rho = materials[material_name]['rho']
cp  = materials[material_name]['cp']
k   = materials[material_name]['k']
alpha = k / (rho * cp)

# Constant heat flux (from your previous titanium run or set as desired)
q_dot_inner = 3_271_395    # W/m²

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Geometry and mesh
# ─────────────────────────────────────────────────────────────────────────────

L   = 0.60
r_i = 0.006
r_o = 0.014

Nz  = 120
Nr  = 40

dz = L / (Nz - 1)
dr = (r_o - r_i) / (Nr - 1)

# Stability-limited time step
dt_stable = 0.25 * min(dr, dz)**2 / alpha
dt = 0.8 * dt_stable

print(f"Δt = {dt*1e3:.2f} ms  (stability limited)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Firing schedule and shot timing
# ─────────────────────────────────────────────────────────────────────────────

rpm = 600
n_shots = 120
Δt_shot = 60 / rpm

shot_times = np.arange(0, n_shots * Δt_shot, Δt_shot)

# Original code (we no longer calculate q″ this way):
# q_per_shot = 30_000  # J
# A_inner = 2 * np.pi * r_i * dz
# q_node = q_per_shot / A_inner
# pulse_len = dt
# q_flux = q_node / pulse_len

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Initial conditions
# ─────────────────────────────────────────────────────────────────────────────

Tinf = 298.0
Tini = 298.0
T = np.full((Nr, Nz), Tini)

h_ext = 55.0  # convection coefficient

frames, times = [], []

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Time marching (main loop)
# ─────────────────────────────────────────────────────────────────────────────

t_final = shot_times[-1] + 300.0
Nsteps = int(np.ceil(t_final / dt))

for n in range(Nsteps):
    t = n * dt

    T_old = T.copy()

    # Apply constant heat flux at each shot time
    if np.isclose(t, shot_times, atol=dt/2).any():
        # NEW: constant q″ used
        T_old[0, :] += (q_dot_inner * dt) / (rho * cp * dr)

        # Original code (now replaced):
        # T_old[0, :] += q_node / (rho * cp * dr)

    # Interior nodes (explicit update)
    for j in range(1, Nr - 1):
        r = r_i + j * dr
        for i in range(1, Nz - 1):
            d2T_dr2 = (T_old[j+1, i] - 2*T_old[j, i] + T_old[j-1, i]) / dr**2
            dT_dr_over_r = (T_old[j+1, i] - T_old[j-1, i]) / (2 * r * dr)
            d2T_dz2 = (T_old[j, i+1] - 2*T_old[j, i] + T_old[j, i-1]) / dz**2
            T[j, i] = T_old[j, i] + alpha * dt * (d2T_dr2 + dT_dr_over_r + d2T_dz2)

    # Boundary conditions
    T[:, 0] = T[:, 1]       # axial symmetry at z = 0
    T[:, -1] = T[:, -2]     # axial symmetry at z = L

    T[0, :] = T[1, :]       # inner wall: adiabatic except for heat pulse

    # Outer surface convection
    j = Nr - 1
    r = r_o
    Bi = h_ext * dr / k
    T[j, :] = (T_old[j-1, :] + Bi * Tinf) / (1 + Bi)

    # Save frames every 0.5 s and PRINT progress
    if n % int(0.5 / dt) == 0:
        out_time = t
        Tmax = np.max(T) - 273.15
        Tmin = np.min(T) - 273.15
        print(f"Time elapsed: {out_time:.2f} s | Max Temp: {Tmax:.2f} °C | Min Temp: {Tmin:.2f} °C")
        frames.append(T.copy())
        times.append(t)

print("Simulation finished.")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Animation and results
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots()
cmap = plt.cm.plasma
im = ax.imshow(frames[0] - 273.15, cmap=cmap, origin='lower',
               extent=[0, L*1e3, r_i*1e3, r_o*1e3],
               vmin=0, vmax=np.max(frames)-273.15,
               aspect='auto')
cb = plt.colorbar(im, ax=ax)
cb.set_label('T (°C)')
ax.set_xlabel('z (mm)')
ax.set_ylabel('r (mm)')
title = ax.set_title('t = 0.0 s')

def update(frame):
    im.set_data(frame - 273.15)
    k = frames.index(frame)
    title.set_text(f't = {times[k]:.1f} s')
    return im, title

ani = animation.FuncAnimation(fig, update, frames=frames, interval=60)
plt.tight_layout()

# Save GIF with material name
gif_filename = f'nonlumped_{material_name}.gif'
ani.save(gif_filename, fps=15)
print(f'  • saved {gif_filename}')

plt.show()

# Report peak temperature
peak_T = np.max(frames)
peak_t = times[np.argmax([np.max(f) for f in frames])]
print(f"Peak T = {peak_T - 273.15:.1f} °C at t = {peak_t:.2f} s")

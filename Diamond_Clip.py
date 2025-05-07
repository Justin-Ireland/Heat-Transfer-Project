"""
thermal_clip_diamond.py
=======================

Metric dual‑model heating of a Mass‑Effect thermal clip
(2″ × 4″ × 1.5″ solid; synthetic diamond).

• Constant heat flux q″ applied to three faces:
      – Bottom  (2″ × 4″   face at y = 0)
      – Side‑X‑min  (1.5″ × 4″ face at x = 0)
      – Side‑X‑max  (1.5″ × 4″ face at x = Lx)
  The **top 2″ × 4″ face (y = Ly) is insulated**; all other faces are too.

• q″ chosen so a lumped model hits 500 °C in 4 s.
• Both lumped and non‑lumped simulations stop automatically when
  ⟨T⟩ ≥ 500 °C.

Outputs:
    lumped_diamond.gif
    nonlumped_diamond.gif
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import plasma


# ───────────────────────────────────────────────────────────────────────────
# 1.  Geometry (metres)  – 2″ × 4″ × 1.5″ block
# ───────────────────────────────────────────────────────────────────────────
INCH = 0.0254
Lx = 2.0 * INCH          # 0.0508 m
Lz = 4.0 * INCH          # 0.1016 m
Ly = 1.5 * INCH          # 0.0381 m      (thickness / height)
VOLUME = Lx * Ly * Lz

# Heated faces: bottom + two long sides
A_bottom   = Lx * Lz
A_side_x0  = Ly * Lz
A_side_xL  = Ly * Lz
A_HEATED   = A_bottom + A_side_x0 + A_side_xL   # m²

# ───────────────────────────────────────────────────────────────────────────
# 2.  Synthetic‑diamond properties
# ───────────────────────────────────────────────────────────────────────────
rho = 3510            # kg m⁻³
cp  = 515             # J kg⁻¹ K⁻¹  (25 °C)
k   = 1000            # W m⁻¹ K⁻¹
MASS = rho * VOLUME

# ───────────────────────────────────────────────────────────────────────────
# 3.  Target conditions
# ───────────────────────────────────────────────────────────────────────────
T_AMB_C = 25.0
T_MAX_C = 500.0
T_BASE  = 4.0                 # s design time to hit 500 °C (lumped)

ΔT      = T_MAX_C - T_AMB_C
ENERGY  = MASS * cp * ΔT
POWER   = ENERGY / T_BASE
q_flux  = POWER / A_HEATED    # W m⁻²

print(f"Heat‑flux needed  q″ = {q_flux:,.0f}  W m⁻²")

# ───────────────────────────────────────────────────────────────────────────
# 4‑A.  Lumped simulation
# ───────────────────────────────────────────────────────────────────────────
def run_lumped(dt=0.01):
    t, T = [0.0], [T_AMB_C]
    while T[-1] < T_MAX_C:
        T.append(T[-1] + (q_flux * A_HEATED * dt) / (MASS * cp))
        t.append(t[-1] + dt)
    return np.array(t), np.array(T)

# ───────────────────────────────────────────────────────────────────────────
# 4‑B.  3‑D explicit finite‑difference simulation
# ───────────────────────────────────────────────────────────────────────────
NX, NY, NZ = 40, 30, 80                # grid resolution (~1.3‑1.7 mm steps)

def run_nonlumped():
    dx, dy, dz = Lx/NX, Ly/NY, Lz/NZ
    alpha = k / (rho * cp)
    dt = 0.4 / (alpha * ((1/dx**2)+(1/dy**2)+(1/dz**2)))   # stability
    T = np.full((NX, NY, NZ), T_AMB_C, dtype=float)

    # ghost‑cell temperature increment for constant flux BC (Fourier law)
    gx = dx * q_flux / k
    gy = dy * q_flux / k
    gz = dz * q_flux / k   # not used (z‑faces insulated)

    t_hist, Tav_hist = [0.0], [T_AMB_C]
    step = 0

    while Tav_hist[-1] < T_MAX_C:
        T_old = T.copy()

        # interior update
        T[1:-1,1:-1,1:-1] = (
            T_old[1:-1,1:-1,1:-1] +
            alpha*dt*(
                (T_old[2:,1:-1,1:-1]-2*T_old[1:-1,1:-1,1:-1]+T_old[:-2,1:-1,1:-1])/dx**2 +
                (T_old[1:-1,2:,1:-1]-2*T_old[1:-1,1:-1,1:-1]+T_old[1:-1,:-2,1:-1])/dy**2 +
                (T_old[1:-1,1:-1,2:]-2*T_old[1:-1,1:-1,1:-1]+T_old[1:-1,1:-1,:-2])/dz**2
            )
        )

        # Boundary faces ---------------------------------------------------
        # x = 0 (heated)
        T[0,:,:]    = T_old[1,:,:] + gx
        # x = Lx (heated)
        T[-1,:,:]   = T_old[-2,:,:] + gx
        # y = 0 (heated bottom)
        T[:,0,:]    = T_old[:,1,:] + gy
        # y = Ly (TOP insulated)  ∂T/∂y = 0
        T[:,-1,:]   = T_old[:,-2,:]
        # z‑faces (front/back) insulated
        T[:,:,0]    = T_old[:,:,1]
        T[:,:,-1]   = T_old[:,:,-2]
        # ------------------------------------------------------------------

        step += 1
        if step % 10 == 0:                 # thin out saved points
            t_hist.append(step*dt)
            Tav_hist.append(T.mean())

    return np.array(t_hist), np.array(Tav_hist), dt

# ───────────────────────────────────────────────────────────────────────────
# 5.  Run simulations
# ───────────────────────────────────────────────────────────────────────────
tL, TL = run_lumped()
tN, TN, dtN = run_nonlumped()

print(f"Lumped      : {tL[-1]:.2f} s to 500 °C")
print(f"Non‑lumped  : {tN[-1]:.2f} s to 500 °C  (Δt = {dtN*1e3:.2f} ms)")

# ───────────────────────────────────────────────────────────────────────────
# 6.  GIF utilities
# ───────────────────────────────────────────────────────────────────────────
def make_gif(time, temp, tag):
    norm = plt.Normalize(vmin=temp[0], vmax=T_MAX_C)
    fig, ax = plt.subplots(figsize=(3.4, 3))
    ax.axis("off")
    rect = plt.Rectangle((0,0),1,1,color=plasma(norm(temp[0])))
    ax.add_patch(rect)
    txt = ax.text(0.5,1.03,"",ha="center",va="bottom",transform=ax.transAxes)

    def update(i):
        rect.set_color(plasma(norm(temp[i])))
        txt.set_text(f"{tag}: t={time[i]:.2f}s  Tavg={temp[i]:.0f}°C")
        return rect, txt

    ani = animation.FuncAnimation(fig, update, frames=len(time),
                                  interval=100, blit=True)
    fname = f"{tag.lower()}_diamond.gif"
    ani.save(fname, writer="pillow", fps=10)
    plt.close(fig)
    print("  • saved", Path(fname).resolve())

make_gif(tL, TL, "LUMPED")
make_gif(tN, TN, "NONLUMPED")

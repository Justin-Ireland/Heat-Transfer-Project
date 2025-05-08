import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
#collaborative code for group inspired by 5-90transient,
#simulation parameters

width     = 0.0508     # 2 in
thickness = 0.0381     # 1.5 in

dx = dy = 0.001        # 1 mm grid spacing
Nx = int(width     / dx) + 1
Ny = int(thickness / dy) + 1

T_initial = 27.0
T_target  = 500.0
q_flux    = 11.785e6    # W/m²

#material properties to select when prompted in CLI for each group member's material
materials = {
    "Diamond": {"k": 1000, "cp": 515, "rho": 3510},
    "Steel (AISI 1010)": {"k":  63.9, "cp": 434, "rho": 7832},
    "Aluminum 2024-T6": {"k": 177.0, "cp": 875, "rho": 2770},
    "Durasteel":         {"k":  25.0, "cp": 535, "rho": 8300},
    "Brass (Cart.)":     {"k": 110.0, "cp": 380, "rho": 8530},
    "Titanium":          {"k":  21.9, "cp": 522, "rho": 4500},
    "Nickel":            {"k":  90.7, "cp": 444, "rho": 8900},
    "Silicon":           {"k": 429, "cp": 235, "rho": 10500}
}

#displays materials with number selection and the needed properties of each
print("Available materials:")
for i, name in enumerate(materials):
    print(f" {i+1}. {name}")
sel   = int(input("Select material [1–6]: ")) - 1
mname = list(materials)[sel]
kappa = materials[mname]["k"]
cp    = materials[mname]["cp"]
rho   = materials[mname]["rho"]
print(f"\nSelected: {mname} (k={kappa}, cp={cp}, rho={rho})\n")

#equations for area, volume, mass, change in temp, and formula for Q stored
depth   = thickness  #thickness as the second dimension
area    = width * depth
volume  = width * depth * thickness
mass    = rho * volume
deltaT  = T_target - T_initial
Q_req   = mass * cp * deltaT

alpha   = kappa/(rho*cp) #diffusivity
Fo      = 0.24
dt      = Fo * dx**2 / alpha
r       = alpha * dt / dx**2

max_time      = 10.0
time          = 0.0

#temperature array for simulation with Nx and Ny being resolution with a set initial temp
T = np.full((Ny, Nx), T_initial)

#this is here so we can begin generating a GIF
snapshots = []
times      = []
frame_dt   = 0.1
next_snap  = 0.0

print("Starting transient simulation with 3-sided heating...")

#loop to generate plot that shows it is heating up until max time, stops at maximum
while time < max_time:
    T_old = T.copy()

    #interior diffusion of array
    T[1:-1, 1:-1] = (
        T_old[1:-1,1:-1]
        + r * (
            T_old[2:,1:-1] + T_old[:-2,1:-1]
          + T_old[1:-1,2:] + T_old[1:-1,:-2]
          - 4*T_old[1:-1,1:-1]
        )
    )

    #3 sided heating flux with one insulated surface
    #bottom edge y=0 from surface
    T[0, :] = T_old[1, :] + (q_flux * dt)/(rho*cp*dy)
    #left edge x=0 from surface
    T[:, 0] = T_old[:, 1] + (q_flux * dt)/(rho*cp*dx)
    #right edge x=Nx−1
    T[:, -1] = T_old[:, -2] + (q_flux * dt)/(rho*cp*dx)
    #top edge (y=Ny−1): insulated
    T[-1, :] = T[-2, :]

    time += dt
    Tmax = T.max()

    #make gif
    if time >= next_snap:
        snapshots.append(T.copy())
        times.append(time)
        next_snap += frame_dt
    #loop to break gif at max T
    if Tmax >= T_target:
        print(f"Reached {T_target}°C at t={time:.3f}s")
        break

print("Simulation complete.\n") #display sim completion

#creates gif as subpot with axes
fig, ax = plt.subplots(figsize=(5,4))
cax = ax.imshow(snapshots[0], origin='lower',
                cmap='plasma', vmin=T_initial, vmax=T_target,
                extent=[0, width, 0, thickness], aspect='auto')
cbar = fig.colorbar(cax, ax=ax, label='Temperature (°C)')

def update(i):
    """
    updates title to display time of sim to reach max temp
    """
    ax.set_title(f't = {times[i]:.2f} s')
    cax.set_data(snapshots[i])
    return [cax]

ani = animation.FuncAnimation(
    fig, update, frames=len(snapshots), #animates
    interval=100, blit=True
)
#saves to our project folder
gif_path = Path.cwd() / f"transient_brass_{mname.replace(' ','_')}.gif"
ani.save(gif_path, writer='pillow')
plt.close(fig)
print(f"✅ Transient GIF saved to {gif_path}")


#plot and save steady state image in same program
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(T, origin='lower',
               cmap='plasma', vmin=T_initial, vmax=T_target,
               extent=[0, width, 0, thickness], aspect='auto')
fig.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title(f'Last Frame (t={time:.2f}s)')
ax.set_xlabel('Width (m)')
ax.set_ylabel('Thickness (m)')
png_path = Path.cwd() / f"last_frame_3sided_{mname.replace(' ','_')}.png"
fig.savefig(png_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Steady-state image saved to {png_path}") #saves steady state image

import numpy as np 
import matplotlib.pyplot as plt 

### Define parameters 
m = 1.0 
k = 1.0 
T_period = 2 * np.pi * 5
dt = 0.001
n_steps = int(T_period / dt)
t = np.linspace(0, T_period, n_steps)

### Initialise arrays 
x = np.zeros(n_steps)
v = np.zeros(n_steps)
a = np.zeros(n_steps)
K = np.zeros(n_steps)
U = np.zeros(n_steps)
E = np.zeros(n_steps)

### Define initial conditions 
x[0] = 1e-4
v[0] = 1e-4
a[0] = - (k / m) * x[0]

### Compute intial kinetic, potential and total energy
K[0] = 0.5 * m * v[0]**2
U[0] = 0.5 * k * x[0]**2
E[0] = K[0] + U[0]

for i in range(n_steps - 1):
        ### Update position, velocity and acceleration 
        x[i+1] = x[i] + v[i] * dt + (0.5 * a[i] * dt**2)
        a[i + 1] = - (k / m) * x[i + 1]
        v[i + 1] = v[i] + (0.5 * dt * (a[i] + a[i + 1]))

        ### Compute the updated energies 
        K[i + 1] = 0.5 * m * v[i + 1]**2
        U[i + 1] = 0.5 * k * x[i + 1]**2
        E[i + 1] = K[i + 1] + U[i + 1]

### Create three subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

### First subplot
axs[0, 0].plot(t, E)
axs[0, 0].set_title(f"Total Energy vs Time (dt = {dt})")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("E(t) - E(0)")
axs[0, 0].axhline(E[0], color='green', linestyle='--', linewidth=1)
axs[0, 0].grid(True)

### Second subplot
axs[0, 1].plot(x, v)
axs[0, 1].set_title(f"Phase Space plot (dt = {dt})")
axs[0, 1].set_xlabel("Position, x")
axs[0, 1].set_ylabel("Velocity, v")
axs[0, 1].grid(True)
axs[0, 1].axis("equal")

### Third subplot
axs[1, 0].plot(t, E - E[0])
axs[1, 0].set_title("Zoomed-In Energy Deviation vs Time")
axs[1, 0].set_xlabel("Time")
axs[1, 0].set_ylabel("E(t) - E(0)")
axs[1, 0].set_ylim(-1e-14, 1e-14)
axs[1, 0].axhline(0, color='gray', linestyle='--', linewidth=1)
axs[1, 0].grid(True)

### Fourth subplot
axs[1, 1].plot(t, E - E[0])
axs[1, 1].set_title("Zoomed-In Energy Deviation vs Time")
axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("E(t) - E(0)")
axs[1, 1].set_ylim(-1e-13, 1e-13)
axs[1, 1].axhline(0, color='gray', linestyle='--', linewidth=1)
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()


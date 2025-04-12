import numpy as np 
import matplotlib.pyplot as plt

### Define parameters in atomic units where appropriate
m = 1.0 
k = 1.0 
T_period = 2 * np.pi * 5

### Define time step values, equispaced on a logarithmic scale
dt_values = np.logspace(-3, 0, 10)

### Create an empty list to keep track of energy change value s
dE_values = []

### Create a for loop to run simulation over the different time steps 
for dt in dt_values:
    n_steps = int(T_period / dt)

    ### Set initial arrays to 0 
    x = np.zeros(n_steps)
    a = np.zeros(n_steps)
    v = np.zeros(n_steps)
    K = np.zeros(n_steps)
    U = np.zeros(n_steps)
    E = np.zeros(n_steps)

    ### Set initial conditions 
    x[0] = 1.0
    v[0] = 1.0
    a[0] = - (k / m) * x[0]

    ### Compute intial energies 
    K[0] = 0.5 * m * v[0]**2
    U[0] = 0.5 * k * x[0]**2
    E[0] = K[0] + U[0]

    ### Incorporate a Velocity-Verlet Algorithm 
    for i in range(n_steps - 1):
        ### Update position, velocity and acceleration 
        x[i+1] = x[i] + v[i] * dt + (0.5 * a[i] * dt**2)
        a[i + 1] = - (k / m) * x[i + 1]
        v[i + 1] = v[i] + (0.5 * dt * (a[i] + a[i + 1]))

        ### Compute the updated energies 
        K[i + 1] = 0.5 * m * v[i + 1]**2
        U[i + 1] = 0.5 * k * x[i + 1]**2
        E[i + 1] = K[i + 1] + U[i + 1]
        
    dE = (1 / n_steps) * np.sum(np.abs((E - E[0]) / E[0]))
    dE_values.append(dE)

# Fit log-log slope
log_dt = np.log10(dt_values)
log_dE = np.log10(dE_values)
slope, intercept = np.polyfit(log_dt, log_dE, 1)

### Log-log plot of energy change 
plt.figure(figsize=(8, 5))
plt.loglog(dt_values, dE_values, linestyle='-')
plt.xlabel(r'Time step $\delta t$')
plt.ylabel(r'Average relative energy deviation $\Delta E$')
plt.title('Average Energy Deviation vs. Time Step (Velocity-Verlet)')
plt.text(dt_values[5], dE_values[7], f"Slope ≈ {slope:.4f}", fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()
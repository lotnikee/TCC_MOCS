import numpy as np 
import matplotlib.pyplot as plt

### Define parameters 
m = 1.0 
k = 1.0 
T = 10.0 

### Define time step values 
dt_values = np.logspace(-3, 0, 10)  # Logarithmically spaced dt from 0.001 to 1.0

### Create  an empty list to keep track of energy change values
delta_E_values = []

### Loop values over different time steps 
for dt in dt_values:
    n_steps = int(T / dt)

    ### Set initial arrays to 0
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a = np.zeros(n_steps)
    K = np.zeros(n_steps)
    U = np.zeros(n_steps)
    E = np.zeros(n_steps)

    ### Set initial conditions 
    x[0] = 1.0
    v[0] = 0.0
    a[0] = - ( k / m) * x[0]

    ### Compute initial energies 
    K[0] = 0.5 * m * v[0] ** 2
    U[0] = 0.5 * k * x[0] ** 2
    E[0] = K[0] + U[0]

    ### Velocity Verlet Algorithm
    for i in range(n_steps - 1):
        x[i+1] = x[i] + v[i]*dt + (0.5 * a[i] *dt ** 2)
        a[i+1] = - (k / m) * x[i+1]
        v[i+1] = v[i] + 0.5 * (a[i] + a[i+1]) * dt

        ### Compute new energies  
        K[i+1] = 0.5 * m * v[i+1] ** 2
        U[i+1] = 0.5 * k * x[i+1] ** 2
        E[i+1] = K[i+1] + U[i+1]

    ### Take the average change in energy 
    delta_E = np.mean(np.abs((E - E[0]) / E[0]))
    delta_E_values.append(delta_E)

### Log-log plot of energy change 
plt.figure(figsize=(8, 5))
plt.loglog(dt_values, delta_E_values, linestyle='-', label=r'$\Delta E$')
plt.xlabel(r'Time step $\delta t$')
plt.ylabel(r'Energy deviation $\Delta E$')
plt.title('Energy Conservation vs. Time Step')
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

### Second set of parameters to check displacement and velocity behaviour 
delta_t = 0.01
n_steps = int(T / delta_t)
time = np.linspace(0, T, n_steps)

### Set initial arrays back to 0
x = np.zeros(n_steps)
v = np.zeros(n_steps)
a = np.zeros(n_steps)
K = np.zeros(n_steps)
U = np.zeros(n_steps)
E = np.zeros(n_steps)

### Set initial conditions 
x[0] = 1.0
v[0] = 0.0
a[0] = - ( k / m) * x[0]

### Compute initial energies 
K[0] = 0.5 * m * v[0] ** 2
U[0] = 0.5 * k * x[0] ** 2
E[0] = K[0] + U[0]

### Velocity Verlet Algorithm
for i in range(n_steps - 1):
    x[i+1] = x[i] + v[i]*delta_t + (0.5 * a[i] *delta_t ** 2)
    a[i+1] = - (k / m) * x[i+1]
    v[i+1] = v[i] + 0.5 * (a[i] + a[i+1]) * delta_t

### Compute new energies  
K = 0.5 * m * v ** 2
U = 0.5 * k * x ** 2
E = K + U

### Take the average change in energy 
delta_E = np.mean(np.abs((E - E[0]) / E[0]))
delta_E_values.append(delta_E)

### Plotting everything in subplots 
plt.figure(figsize=(12, 9))

### Displacement plot
plt.subplot(3, 1, 1)
plt.plot(time, x, label='Displacement (x)', linestyle='-', color='blue')
plt.title('Harmonic Oscillator Simulation')
plt.ylabel('Displacement')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()

### Velocity plot
plt.subplot(3, 1, 2)
plt.plot(time, v, label='Velocity (v)', linestyle='-', color='green')
plt.ylabel('Velocity')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()

### Energy fluctuation
plt.subplot(3, 1, 3)
plt.plot(time, E, label='Total Energy (E)', linestyle='-', color='orange')
plt.xlabel('Time')
plt.ylabel('Energy Fluctutation')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()
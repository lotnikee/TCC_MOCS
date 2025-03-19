import numpy as np 
import matplotlib.pyplot as plt

### Define parameters
m = 1.0 
k = 1.0 
dt = 0.1
T = 10.0

### Initialise positions and velocities 
x0 = 1.0
v0 = 0.0

### Derive other constants 
omega = np.sqrt(k/m)
n_steps = int(T/dt)

time = np.linspace(0, T, n_steps)
x = np.zeros(n_steps)
v = np.zeros(n_steps)
a = np.zeros(n_steps)
K = np.zeros(n_steps)
U = np.zeros(n_steps)
E = np.zeros(n_steps)

### List to store time, position and velocity data 
trajectory = []
energy = []
delta_energy = []

### Initial conditions 
x[0] = x0
v[0] = v0
a[0] = - (k/m) * x0

### Include energy components
K[0] = 0.5 * m * v0**2
U[0] = 0.5 * m * x0**2
E[0] = K[0] + U[0]

### Store intial conditions
trajectory.append((time[0], x[0], v[0], E[0]))
energy.append(E[0])

### Velocity Verlet Algorithm integration
for i in range(n_steps - 1):
   
    ### Update position 
    x[i+1] = x[i] + v[i] * dt + (0.5 * a[i] * dt**2)
   
    ### Update acceleration 
    a_new = - (k/m) * x[i+1]

    ### Update velocity 
    v[i+1] = v[i] + 0.5 * (a[i] + a_new) * dt

    ### Store new acceleration 
    a[i+1] = a_new

    ### Update kinetic and potential energy
    K[i+1] = 0.5 * m * v[i+1]**2
    U[i+1] = 0.5 * k * x[i+1]**2
    
    ### Update total energy 
    E_new = K[i+1] + U[i+1]

    ### Store new energy 
    E[i+1] = E_new

    ### Calculate energy differerence 
    E_norm = np.abs((E[i+1] - E[0]) / E[0])

    ### Calculate energy difference
    delta_energy.append(E_norm)    

    ### Store trajectory data
    trajectory.append((time[i+1], x[i+1], v[i+1], E[i+1]))
    energy.append(E[i+1])

# Plot results
plt.figure(figsize=(8,5))
plt.plot(time, x, label='Position x(t)')
plt.plot(time, v, label='Velocity v(t)', linestyle='dashed')
plt.plot(time, E, label='Energy')
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Time (s)')
plt.ylabel('Position & Velocity')
plt.legend()
plt.title('1D Harmonic Oscillator - Velocity Verlet')
plt.show()

print(trajectory)
print(len(trajectory))

print(delta_energy)
print(energy)



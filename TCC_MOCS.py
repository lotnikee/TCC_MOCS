import numpy as np 
import matplotlib.pyplot as plt

# Define parameters
m = 1.0 
k = 1.0 
T = 10.0
delta_t = 0.01

# Number of steps
n_steps = int(T / delta_t)

# Time array
time = np.linspace(0, T, n_steps)

# Initialize arrays
x = np.zeros(n_steps)
v = np.zeros(n_steps)
a = np.zeros(n_steps)
K = np.zeros(n_steps)
U = np.zeros(n_steps)
E = np.zeros(n_steps)

# Initial conditions
x[0] = 1.0
v[0] = 0.0
a[0] = - (k / m) * x[0]

# Initial energy
K[0] = 0.5 * m * v[0]**2
U[0] = 0.5 * k * x[0]**2
E[0] = K[0] + U[0]

# Velocity Verlet algorithm
for i in range(n_steps - 1):
    x[i+1] = x[i] + v[i] * delta_t + 0.5 * a[i] * delta_t**2
    a[i+1] = - (k / m) * x[i+1]
    v[i+1] = v[i] + 0.5 * (a[i] + a[i+1]) * delta_t

K = 0.5 * m * v**2
U = 0.5 * k * x**2  
E = K + U

# Plotting everything in subplots
plt.figure(figsize=(12, 9))

# Displacement
plt.subplot(3, 1, 1)
plt.plot(time, x, label='Displacement (x)', color='blue')
plt.ylabel('Displacement')
plt.title('Harmonic Oscillator Simulation')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()

# Velocity
plt.subplot(3, 1, 2)
plt.plot(time, v, label='Velocity (v)', color='orange')
plt.ylabel('Velocity')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()

# Total Energy
plt.subplot(3, 1, 3)
plt.plot(time, E, label='Total Energy (E)', color='green')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()

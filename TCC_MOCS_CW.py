import numpy as np 
import matplotlib.pyplot as plt

### Define parameters
m = 1.0 
k = 1.0 
T = 10.0

### Define different time step values 
dt_values = np.logspace(-3, 0, 10)
delta_E_values = []

### Loop over different timesteps 
for dt in dt_values:
    n_steps = int(T / dt)

    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a = np.zeros(n_steps)
    E = np.zeros(n_steps)

    ### List to store time, position and velocity data 
    trajectory = []
    energy = []
    delta_energy = []

    ### Initial conditions 
    x[0] = 1.0
    v[0] = 0.0
    a[0] = - (k/m) * x[0]

    ### Include energy components
    K0 = 0.5 * m * v[0]**2
    U0 = 0.5 * m * x[0]**2
    E[0] = K0 + U0
    E0 = E[0]

    ### Store intial conditions
    trajectory.append((x[0], v[0], E[0]))
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
        K = 0.5 * m * v[i+1]**2
        U  = 0.5 * k * x[i+1]**2
        E[i + 1] = K + U

    ### Calculate energy differerence 
    E_norm = np.abs((E[i+1] - E[0]) / E[0])

    ### Calculate energy difference
    delta_energy.append(E_norm)    

    ### Store trajectory data
    trajectory.append((x[i+1], v[i+1], E[i+1]))
    energy.append(E[i+1])

    ### Compute the final Delta E over the entire trajectory
    delta_E = np.mean(delta_energy)

### Log-Log Plot of Delta E vs Time Step
plt.figure(figsize=(8, 5))
plt.loglog(dt_values, delta_E_values, marker='o', linestyle='-', label=r'$\Delta E$')
plt.xlabel(r'Time step $\delta t$')
plt.ylabel(r'Energy deviation $\Delta E$')
plt.title('Energy Conservation vs. Time Step')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

### Print trajectory and energies
print(trajectory)
print(energy)
print(delta_energy)


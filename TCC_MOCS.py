import numpy as np
import matplotlib.pyplot as plt

### Define parameters
m = 1.0
k = 1.0
T = 10.0  

### Define different time step values
dt_values = np.logspace(-3, 0, 10)  # Logarithmically spaced dt from 0.001 to 1.0
delta_E_values = []

### Loop over different timesteps
for dt in dt_values:
    n_steps = int(T / dt)

    ### Initialise arrays
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a = np.zeros(n_steps)
    E = np.zeros(n_steps)

    ### Initial conditions
    x[0] = 1.0
    v[0] = 0.0
    a[0] = - (k / m) * x[0]

    ### Compute initial energy
    K0 = 0.5 * m * v[0] ** 2
    U0 = 0.5 * k * x[0] ** 2
    E[0] = K0 + U0
    E0 = E[0]

    ### Velocity Verlet Algorithm
    for i in range(n_steps - 1):
        x[i + 1] = x[i] + v[i] * dt + (0.5 * a[i] * dt ** 2)
        a_new = - (k / m) * x[i + 1]
        v[i + 1] = v[i] + 0.5 * (a[i] + a_new) * dt
        a[i + 1] = a_new

        # Compute energy
        K = 0.5 * m * v[i + 1] ** 2
        U = 0.5 * k * x[i + 1] ** 2
        E[i + 1] = K + U

    ### Compute Delta E
    delta_E = np.mean(np.abs((E - E0) / E0))
    delta_E_values.append(delta_E)

### Log-Log Plot of Delta E vs Time Step
plt.figure(figsize=(8, 5))
plt.loglog(dt_values, delta_E_values, marker='o', linestyle='-', label=r'$\Delta E$')
plt.xlabel(r'Time step $\delta t$')
plt.ylabel(r'Energy deviation $\Delta E$')
plt.title('Energy Conservation vs. Time Step')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

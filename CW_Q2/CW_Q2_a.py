import matplotlib.pyplot as plt 
import numpy as np 
import random 

random.seed(42)
np.random.seed(42)

### Set-up and parameters
k = 1.0 
beta = 1.0
N_steps = 10**6
equilibration_steps = N_steps // 2
production_steps = N_steps // 2

### Initial conditions
accepted_x = []
current_x = random.random()

def potential_energy(x, k):
    return 0.5 * k * x**2

### Metropolis MC loop
for step in range(1, N_steps + 1):

    ### Define the displacement parameters 
    delta = random.uniform(-0.1, 0.1)
    proposed_x = current_x + delta

    ### Determine whether or not the new position falls within the boundary conditions
    if 0 <= proposed_x <= 1:

        ### Calculate the energy E(x)
        E_x = potential_energy(current_x, k)

        ### Calculate the energy E(x')
        E_x_new = potential_energy(proposed_x, k)

        ### Determine π(x) and π(x')
        π_x = np.exp(- beta * E_x)
        π_x_new = np.exp(- beta * E_x_new)

        ### Determine the ratio between π(x) and π(x')
        ratio = π_x_new / π_x

        ### Define the acceptance criterion 
        A = min(1, ratio)
        current_x = proposed_x
        if step > equilibration_steps:
            accepted_x.append(current_x)

### Metropolis MC Processing by plotting accepted states
plt.figure()
plt.hist(accepted_x, bins=200, edgecolor='black') 
plt.title("Histogram of accepted x' values ")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

### Determine the average proposed_x value 
print(np.mean(accepted_x))



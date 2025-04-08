import matplotlib.pyplot as plt 
import numpy as np 
import random 

### Set-up and parameters
k = 1.0 
beta = 1.0
N_steps = 10**6
equilibration_steps = N_steps // 2
production_steps = N_steps // 2

### Initial conditions
accepted_x = []
current_x = random.uniform(0.1, 0.9)

def potential_energy(x, k):
    return 0.5 * k * x**2

### Metropolis MC loop
for step in range(1, N_steps + 1):

    ### Define the displacement parameters 
    phi = random.uniform(1, 1.1)

    if random.random() < 0.5:
        phi = 1.0 / phi
    proposed_x = current_x * phi

    ### Determine whether or not the new position falls within the boundary conditions
    if proposed_x <= 1.0 and proposed_x >= 1e-10:

        ### Calculate the energies E(x) and E(x')
        E_x = potential_energy(current_x, k)
        E_x_new = potential_energy(proposed_x, k)

        ### Determine π(x) and π(x')
        π_x = np.exp(- beta * E_x)
        π_x_new = np.exp(- beta * E_x_new)

        ### Proposal density ratio
        proposal_ratio = abs(current_x / proposed_x)

        ### Determine the ratio between π(x) and π(x')
        ratio = (π_x_new / π_x) * proposal_ratio

        ### Define the acceptance criterion 
        A = min(1, ratio)

        if random.random() < A:
            current_x = proposed_x
        if step > equilibration_steps:
            accepted_x.append(current_x)

print("Number of samples:", len(accepted_x))
print("Min:", min(accepted_x), "Max:", max(accepted_x))
print("First few:", accepted_x[:10])


### Metropolis MC Processing by plotting accepted states
plt.figure()
plt.hist(accepted_x, bins="auto", edgecolor='black') 
plt.title("Histogram of accepted x' values ")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

### Determine the average proposed_x value 
print(np.mean(accepted_x))



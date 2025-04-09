import numpy as np 
import matplotlib.pyplot as plt 
import random 

### Build the lattice 
L = 10
lattice_L  = np.zeros((L, L), dtype=int)

### Set up the iteration steps for the Metropolis Monte Carlo simulation
N_steps = 10**6
equilibration_phase = N_steps // 2
production_phase = N_steps // 2

### Randomly generate an initial lattice with randomised spins 
init_random = np.random.random((L, L))

lattice_L[init_random>=0.5] = 1
lattice_L[init_random<0.5] = -1

### Calculate the initial energy of the lattice 
def calculate_energy(lattice_L):
    energy_j = 0
    for i in range(L):
        for j in range(L):
            S = lattice_L[i, j]
            ### Count only the right and down neighbours to avoid double counting 
            right = lattice_L[i, (j + 1) % L]
            down = lattice_L[(i + 1) % L, j]
            energy_j += -S * (right + down)
    return energy_j

E_j = calculate_energy(lattice_L)
print(E_j)


### Create a plot with the two lattices side by side
plt.imshow(lattice_L)
plt.show()

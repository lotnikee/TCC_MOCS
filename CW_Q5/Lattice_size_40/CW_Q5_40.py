import numpy as np 
import matplotlib.pyplot as plt 
import random 

### Random seeds for reproducibility
random.seed(13)
np.random.seed(13)

### Define a range of temperatures
T_values = np.linspace(1.5, 3.5, 100)

### Include empty lists to store values
energy_values = []
magnetisation_values = []
heat_capacity = []
magnetic_susceptibility = []

### Build the lattice 
L =40
lattice_L  = np.zeros((L, L), dtype=int)

### Set up the iteration steps for the Metropolis Monte Carlo simulation
N_steps = 10**6
equilibration_phase = N_steps // 2
production_phase = N_steps // 2

for T in T_values:
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
    
    ### Keep track of the different values 
    E_j = calculate_energy(lattice_L)
    energy_list_T = []
    magnetisation_list_T = []

    ### Start the Metropolis MC simulation
    for step in range(1, N_steps + 1): 
        i = random.randint(0, L-1)
        j = random.randint(0, L-1)
        current_state = lattice_L[i, j]
    
        top = lattice_L[(i - 1) % L][j]
        bottom = lattice_L[(i + 1) % L][j]
        right = lattice_L[i][(j - 1) % L]
        left = lattice_L[i][(j + 1) % L]

        neighbour_sum = top + bottom + left + right 
        delta_E_j = 0.5 * current_state * neighbour_sum

        ### If the energy change is negative or within an accepted criterion, always accept spin change as this is energetically favourable
        if delta_E_j <= 0 or random.random() < np.exp(-delta_E_j / T):
            lattice_L[i, j] *= -1
            E_j += delta_E_j
    
        ### If the simulation is in its production phase, keep track of accepted values for x and append the list
        if step > equilibration_phase:
            energy_list_T.append(E_j)
            M = np.sum(lattice_L)
            magnetisation_list_T.append(np.abs(M))
    
    ### Normalise everything per spin 
    N = L * L
    energy_array = np.array(energy_list_T)
    magnetisation_array = np.array(magnetisation_list_T)

    E_mean = np.mean(energy_array) / N
    E_squared_mean = np.mean(energy_array ** 2) / (N ** 2)
    M_mean = np.mean(magnetisation_array) / N
    M_squared_mean = np.mean(magnetisation_array ** 2) / (N ** 2)

    ### Store the different observables
    energy_values.append(E_mean)
    magnetisation_values.append(M_mean)
    C = (E_squared_mean - (E_mean ** 2)) / (T ** 2)
    X = (M_squared_mean - (M_mean ** 2)) / (T ** 2)
    heat_capacity.append(C)
    magnetic_susceptibility.append(X)

### Create three subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot results
axs[0, 0].plot(T_values, energy_values, linestyle='-')
axs[0, 0].set_xlabel("Temperature, T")
axs[0, 0].set_ylabel("Mean Energy per Spin ⟨E⟩")
axs[0, 0].set_title("Average Energy vs. Temperature")
axs[0, 0].grid(True, linestyle="--", linewidth=0.5)

axs[0, 1].plot(T_values, heat_capacity, linestyle='-')
axs[0, 1].set_xlabel("Temperature, T")
axs[0, 1].set_ylabel("Heat Capacity per Spin, C")
axs[0, 1].set_title("Heat Capacity vs. Temperature")
axs[0, 1].set_xscale("log")
axs[0, 1].grid(True, linestyle="--", linewidth=0.5)

axs[1, 0].plot(T_values, magnetisation_values, linestyle='-')
axs[1, 0].set_xlabel("Temperature, T")
axs[1, 0].set_ylabel("Average Magnetisation per Spin ⟨|M|⟩")
axs[1, 0].set_title("Magnetisation vs. Temperature")
axs[1, 0].grid(True, linestyle="--", linewidth=0.5)

axs[1, 1].plot(T_values, magnetic_susceptibility, linestyle='-')
axs[1, 1].set_xlabel("Temperature, T")
axs[1, 1].set_ylabel("Magnetic Susceptibility per Spin, C")
axs[1, 1].set_title("Magnetic Susceptibility vs. Temperature")
axs[1, 1].set_xscale("log")
axs[1, 1].grid(True, linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()

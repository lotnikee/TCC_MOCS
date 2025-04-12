import numpy as np 
import matplotlib.pyplot as plt 
import random 

### Define some constants 
T_values = np.linspace(1.5, 3.5, 100)

### Include an empty list to store energy values 
energy_values = []
magnetisation_values = []
heat_capacity = []
magnetic_susceptibility = []

### Build the lattice 
L = 50
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
    energy_list_T = []
    magnetisation_list_T = []
    E_j = calculate_energy(lattice_L)

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
    
# Plot results
plt.figure(figsize=(8, 5))
plt.plot(T_values, energy_values, linestyle='-')
plt.xlabel("Temperature, T")
plt.ylabel("Mean Energy per Spin ⟨E⟩")
plt.title("Average Energy vs. Temperature")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

plt.figure()
plt.plot(T_values, magnetisation_values, linestyle='-')
plt.xlabel("Temperature, T")
plt.ylabel("Average Magnetization per Spin ⟨|M|⟩")
plt.title("Magnetization vs. Temperature")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

plt.figure()
plt.plot(T_values, heat_capacity, linestyle='-')
plt.xlabel("Temperature, T")
plt.ylabel("Heat Capacity per Spin, C")
plt.title("Heat Capacity vs. Temperature")
plt.xscale("log")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

plt.figure()
plt.plot(T_values, magnetic_susceptibility, linestyle='-')
plt.xlabel("Temperature, T")
plt.ylabel("Magnetic Susceptibility per Spin, C")
plt.title("Magnetic Susceptibility vs. Temperature")
plt.xscale("log")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()


plt.imshow(lattice_L)
plt.show()

import numpy as np 
import matplotlib.pyplot as plt 
import random 

### Simulation parameters
lattice_sizes = [20, 40, 60]
T_values = np.linspace(1.5, 3.5, 100)
MC_steps = 5000
equilibration_steps = 1000
random_seed = 13

### Define a simulation function 
def run_simulation(L, T_values, MC_steps, equilibration_steps, seed=13):
    random.seed(seed)
    np.random.seed(seed)

    ### Set up the number of Monte Carlo steps
    N_steps = (MC_steps + equilibration_steps) * L * L 
    sample_interval = L * L

    ### Create empty lists to store observable values 
    energy_values = []
    magnetisation_values = []
    heat_capacity = []
    magnetic_susceptibility = []

    ### Build and initialise the lattice 
    lattice_L = np.zeros((L, L), dtype=int)

    def calculate_energy(lattice_L):
        energy_j = 0
        for i in range(L):
            for j in range(L):
                S = lattice_L[i, j]
                right = lattice_L[i, (j + 1) % L]
                down = lattice_L[(i + 1) % L, j]
                energy_j += S * (right + down)
        return energy_j
    
    for T in T_values:
        ### Initialise random spin configuration on the lattice
        init_random = np.random.random((L, L))
        lattice_L[init_random >= 0.5] = 1
        lattice_L[init_random < 0.5] = -1

        E_j = calculate_energy(lattice_L)
        energy_list_T = []
        magnetisation_list_T = []

        for step in range(1, N_steps + 1):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            current_state = lattice_L[i, j]

            top = lattice_L[(i - 1) % L, j]
            bottom = lattice_L[(i + 1) % L, j]
            right = lattice_L[i, (j-1) % L]
            left = lattice_L[i, (j + 1) % L]
            neighbour_sum = top + bottom + left + right

            delta_E_j = 2 * current_state * neighbour_sum

            if delta_E_j <= 0 or random.random() < np.exp(-delta_E_j / T):
                lattice_L[i, j] += -1
                E_j += delta_E_j

            if step > (equilibration_steps * L * L) and step % sample_interval == 0:
                energy_list_T.append(E_j)
                M = np.sum(lattice_L)
                magnetisation_list_T.append(np.abs(M))

        ### Normalise everything per spin 
        N = L * L
        energy_array = np.array(energy_list_T)
        magnetisation_array = np.array(magnetisation_list_T)

        ### Calculating energy and magnetisation values
        E_mean = np.mean(energy_array) / N
        E_squared_mean = np.mean(energy_array ** 2) / (N ** 2)
        M_mean = np.mean(magnetisation_array) / N
        M_squared_mean = np.mean(magnetisation_array ** 2) / (N ** 2)

        ### Calculating heat capacity and magnetic susceptibility 
        C = (E_squared_mean - (E_mean ** 2)) / (T ** 2)
        X = (M_squared_mean - (M_mean ** 2)) / (T ** 2)

        ### Store the different observables
        energy_values.append(E_mean)
        magnetisation_values.append(M_mean)
        heat_capacity.append(C)
        magnetic_susceptibility.append(X)

    return { 
    'energy': energy_values, 
    'magnetisation': magnetisation_values,
    'heat_capacity': heat_capacity, 
    'susceptibility': magnetic_susceptibility
    }

def plot_observable(simulation_results, T_values, observable_key, ylabel, title, logscale=False):
    plt.figure(figsize=(8, 6))
    for L, result in simulation_results.items():
        plt.plot(T_values, result[observable_key], label=f"L = {L}")
    plt.xlabel("Temperature, T")
    plt.ylabel(ylabel)
    plt.title(title)
    if logscale:
        plt.xscale("log")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================
# === RUNNING THE SIMULATIONS ===
# ============================
simulation_results = {}

for L in lattice_sizes:
    print(f"\nRunning simulation for lattice size L = {L}...")
    results = run_simulation(L, T_values, MC_steps, equilibration_steps, seed=random_seed)
    simulation_results[L] = results

# ============================
# === GENERATE OVERLAY PLOTS ===
# ============================
plot_observable(simulation_results, T_values, 'magnetisation', 
                "Average Magnetisation per Spin ⟨|M|⟩", 
                "Magnetisation vs. Temperature")

plot_observable(simulation_results, T_values, 'heat_capacity', 
                "Heat Capacity per Spin, C", 
                "Heat Capacity vs. Temperature", 
                logscale=True)

plot_observable(simulation_results, T_values, 'susceptibility', 
                "Magnetic Susceptibility per Spin, χ", 
                "Magnetic Susceptibility vs. Temperature", 
                logscale=True)

plot_observable(simulation_results, T_values, 'energy', 
                "Mean Energy per Spin ⟨E⟩", 
                "Energy vs. Temperature")
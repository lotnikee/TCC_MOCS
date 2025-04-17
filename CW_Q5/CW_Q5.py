import numpy as np 
import matplotlib.pyplot as plt 
import random 
from tqdm import tqdm 

### Simulation parameters
lattice_sizes = [20, 40, 60]
MC_steps = 8000
equilibration_steps = 1000
random_seed = 13
n_runs = 2

### Fine resolution near the critical temperature at zero field 
T_core = np.linspace(2.0, 2.6, 50)
### Broader sampling on either side of the critical temperature at zero field 
T_low = np.linspace(1.5, 2.0, 10, endpoint=False)
T_high = np.linspace(2.6, 3.5, 10, endpoint=False)
### Combine the regions 
T_values = np.concatenate((T_low, T_core, T_high))

### Define a simulation function 
def run_simulation(L, T_values, MC_steps, equilibration_steps, seed=13, n_runs=1):
    sample_interval = L * L
    N = L * L

    ### Create empty lists to store observable values 
    energy_values = []
    magnetisation_values = []
    heat_capacity = []
    magnetic_susceptibility = []
    energy_stds = []
    magnetisation_stds = []
    heat_capacity_stds = []
    magnetic_susceptibility_stds = []

    ### Define calculation of the lattice energy, avoiding double counting
    def calculate_energy(lattice_L):
        energy_j = 0
        for i in range(L):
            for j in range(L):
                S = lattice_L[i, j]
                right = lattice_L[i, (j + 1) % L]
                bottom = lattice_L[(i + 1) % L, j]
                energy_j += -S * (right + bottom)
        return energy_j
    
    ### Code a progress bar per temperature value to estimate how long the simulation is going to take
    for T in tqdm(T_values, desc=f"Simulating L = {L}", leave=False):
        E_T_runs , M_T_runs, E2_T_runs, M2_T_runs = [], [], [], []

        for run in range(n_runs):
            run_seed = seed + run
            random.seed(run_seed)
            np.random.seed(run_seed)

            ### Initialise random spin configuration on the lattice
            lattice_L = np.random.choice([-1, 1], size =(L, L))
            E_j = calculate_energy(lattice_L)

            energy_list_T = []
            magnetisation_list_T = []

            N_steps = (MC_steps + equilibration_steps) * N

            ### Build a random lattice 
            for step in range(1, N_steps + 1):
                i = random.randint(0, L - 1)
                j = random.randint(0, L - 1)
                current_state = lattice_L[i, j]

                ### Determine the lattice energy
                top = lattice_L[(i - 1) % L, j]
                bottom = lattice_L[(i + 1) % L, j]
                right = lattice_L[i, (j - 1) % L]
                left = lattice_L[i, (j + 1) % L]
                neighbour_sum = top + bottom + left + right
                delta_E_j = 2 * current_state * neighbour_sum

                ### If energy is negative or less than acceptance criterion, accept lattice and its energy
                if delta_E_j <= 0 or random.random() < np.exp(-delta_E_j / T):
                    lattice_L[i, j] *= -1
                    E_j += delta_E_j

                ### Only append lists when outside of equilibration steps
                if step > (equilibration_steps * L * L) and step % sample_interval == 0:
                    energy_list_T.append(E_j)
                    M = np.sum(lattice_L)
                    magnetisation_list_T.append(np.abs(M))

            ### Normalise everything per spin 
            energy_array = np.array(energy_list_T) / N
            magnetisation_array = np.array(magnetisation_list_T) / N 

            ### Append empty run lists with average energy and magnetisation normalised per spin
            E_T_runs.append(np.mean(energy_array))
            E2_T_runs.append(np.mean(energy_array ** 2))
            M_T_runs.append(np.mean(magnetisation_array))
            M2_T_runs.append(np.mean(magnetisation_array ** 2))

        ### Calculating energy and magnetisation values
        E_mean = np.mean(E_T_runs)
        E2_mean = np.mean(E2_T_runs)
        M_mean = np.mean(M2_T_runs)
        M2_mean = np.mean(M2_T_runs)

        ### Calculating heat capacity and magnetic susceptibility 
        C = (E2_mean - (E_mean ** 2)) / (T ** 2)
        X = (M2_mean - (M_mean ** 2)) / (T ** 2)

        ### Store the different observables
        energy_values.append(E_mean)
        magnetisation_values.append(M_mean)
        heat_capacity.append(C)
        magnetic_susceptibility.append(X)

        ### Standard deviations for error bars (unbiased, sample standard deviatioin)
        energy_stds.append(np.std(E_T_runs, ddof=1))
        magnetisation_stds.append(np.std(M_T_runs, ddof=1))
        heat_capacity_stds.append(np.std(E2_T_runs, ddof=1) / (T ** 2))
        magnetic_susceptibility_stds.append(np.std(E2_T_runs, ddof=1) / (T ** 2))

    return { 
        'energy': energy_values, 
        'magnetisation': magnetisation_values,
        'heat_capacity': heat_capacity, 
        'susceptibility': magnetic_susceptibility,
        'energy_std': energy_stds,
        'magnetisation_std': magnetisation_stds, 
        'heat_capacity_std': heat_capacity_stds, 
        'magnetic_susceptibility_std': magnetic_susceptibility_stds
    }

### Define a function for plotting the simulation results
def plot_observable(simulation_results, T_values, observable_key, ylabel, title, logscale=False):
    plt.figure(figsize=(8, 6))
    for L, result in simulation_results.items():
        y = result[observable_key]
        yerr = result.get(observable_key + '_std', None)
        plt.errorbar(T_values, y, yerr=yerr, label=f"L = {L}", capsize=3, fmt='-o', markersize=3)
    plt.xlabel("Temperature, T")
    plt.ylabel(ylabel)
    plt.title(title)
    if logscale:
        plt.xscale("log")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

### Running the simulation
simulation_results = {}

for L in lattice_sizes:
    print(f"\nRunning simulation for lattice size L = {L}...")
    results = run_simulation(L, T_values, MC_steps, equilibration_steps, seed=random_seed, n_runs=n_runs)
    simulation_results[L] = results

### Plotting overlays of the different lattice sizes 
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
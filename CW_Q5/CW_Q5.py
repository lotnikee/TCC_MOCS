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
    for T in tqdm(T_values, desc=f"Simulating L = {L}", dynamic_ncols=True):
        all_energy_samples = []
        all_magnetisation_samples = []

        for run in range(n_runs):
            run_seed = seed + run
            random.seed(run_seed)
            np.random.seed(run_seed)

            ### Initialise random spin configuration on the lattice
            lattice_L = np.random.choice([-1, 1], size =(L, L))
            E_j = calculate_energy(lattice_L)

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
                    all_energy_samples.append(E_j / N)
                    M = np.sum(lattice_L)
                    all_magnetisation_samples.append(np.abs(M) / N)

            ### Convert to arrays 
            energy_array = np.array(all_energy_samples) 
            magnetisation_array = np.array(all_magnetisation_samples) 

        ### Calculating energy and magnetisation values
        E_mean = np.mean(energy_array)
        E2_mean = np.mean(energy_array ** 2)
        M_mean = np.mean(magnetisation_array)
        M2_mean = np.mean(magnetisation_array ** 2)

        ### Calculating heat capacity and magnetic susceptibility 
        C = (E2_mean - E_mean ** 2) / (T ** 2)
        X = (M2_mean - M_mean ** 2) / (T ** 2)

        ### Store the different observables
        energy_values.append(E_mean)
        magnetisation_values.append(M_mean)
        heat_capacity.append(C)
        magnetic_susceptibility.append(X)

    return { 
        'energy': energy_values, 
        'magnetisation': magnetisation_values,
        'heat_capacity': heat_capacity, 
        'magnetic_susceptibility': magnetic_susceptibility,
    }

### Define a function for plotting the simulation results
def plot_observable(simulation_results, T_values, observable_key, ylabel, title, logscale=False):
    plt.figure(figsize=(8, 6))
    for L, result in simulation_results.items():
        y = result[observable_key]
        plt.plot(T_values, y, label=f"L = {L}")
    plt.xlabel("Temperature, T")
    plt.ylabel(ylabel)
    plt.title(title)
    if logscale:
        plt.xscale("log")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

### Define a function that will estimate the transition temperature
def print_Tc_estimates(simulation_results, T_values):
    print("\nEstimated Transition Temperatures (T_c):\n")
    for L, results in simulation_results.items():
        chi = np.array(results['magnetic_susceptibility'])
        C = np.array(results['heat_capacity'])
        Tc_chi = T_values[np.argmax(chi)]
        Tc_C = T_values[np.argmax(C)]
        print(f"L = {L}:  T_c from χ(T) = {Tc_chi:.3f},  T_c from C(T) = {Tc_C:.3f}")


### Running the simulation
simulation_results = {}

for L in tqdm(lattice_sizes, desc="Lattice sizes", dynamic_ncols=True):
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

plot_observable(simulation_results, T_values, 'magnetic_susceptibility', 
                "Magnetic Susceptibility per Spin, χ", 
                "Magnetic Susceptibility vs. Temperature", 
                logscale=True)

plot_observable(simulation_results, T_values, 'energy', 
                "Mean Energy per Spin ⟨E⟩", 
                "Energy vs. Temperature")

### Print transition temperatures
print_Tc_estimates(simulation_results, T_values)
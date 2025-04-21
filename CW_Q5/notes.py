import numpy as np 
import random
from tqdm import tqdm 
import multiprocessing as mp 

### Set up the temperature range, including more steps near the critical temperature 
T_low = np.linspace(1.5, 2.0, 10, endpoint=False)
T_core = np.linspace(2.0, 2.6, 50)
T_high = np.linspace(2.6, 3.5, 10, endpoint=False)
T_values = np.concatenate((T_low, T_core, T_high))

### Define simulation parameters for different lattice sizes 
### Monte Carlo steps, fraction of which are used to equilibrate, number of runs 
def set_simulation_parameters(L):
    if L == 20: 
        return 10000, 0.2, 2
    elif L == 40: 
        return 12000, 0.25, 3
    else: 
        return 15000, 0.25, 3
    
### Define a function to calculate the initial lattice energy 
### Only count right and bottom neighbours to avoid double counting 
def calculate_initial_energy(lattice): 
    L = lattice.shape[0]
    energy = 0
    for i in range(L): 
        for j in range(L):
            S = lattice[i, j]
            right = lattice[i, (j + 1) % L]
            bottom = lattice[(i + 1) % L, j]
            energy += -S * (right + bottom)
    return energy 

### Define a function for the Metropolis Monte Carlo simulation 
def metropolis_step(lattice, i, j, T): 
    L = lattice.shape[0]
    spin = lattice[i, j]
    neighbours = lattice[(i - 1) % L, j] + lattice[(i + 1) % L, j] + \
                 lattice[i, (j - 1) % L] + lattice[i, (j + 1) % L]
    delta_E = 2 * spin * neighbours

    ### If process is exothermic or meets acceptance criterion, flip the spin
    if delta_E <= 0 or random.random() < np.exp(-delta_E / T): 
        lattice[i, j] += -1
        return delta_E
    return 0 

def run_simulation(L, T_values, seed=13):
    total_steps, equil_fraction, n_runs = set_simulation_parameters(L)
    equilibration_steps = int(equil_fraction * total_steps)
    production_steps = total_steps - equilibration_steps

    N = L * L
    sample_interval = N

    energy_values, magnetisation_values = [], []
    heat_capacity, magnetic_susceptibility = [], []

    for T in tqdm(T_values, desc=f"Simulating L = {L}", dynamic_ncols=True):
        all_energy_samples = []
        all_magnetisation_samples = []

        for run in range(n_runs):
            run_seed = seed + run
            random.seed(run_seed)
            np.random.seed(run_seed)

            lattice = np.random.choice([-1, 1], size=(L, L))
            E = calculate_initial_energy(lattice)
            total_mc_steps = total_steps * N
            equilibration_limit = equilibration_steps * N

            for step in range(1, total_mc_steps + 1):
                i, j = random.randint(0, L - 1), random.randint(0, L - 1)
                E += metropolis_step(lattice, i, j, T)

                if step > equilibration_limit and step % sample_interval == 0:
                    all_energy_samples.append(E / N)
                    M = np.sum(lattice)
                    all_magnetisation_samples.append(np.abs(M) / N)

        energy_array = np.array(all_energy_samples)
        magnetisation_array = np.array(all_magnetisation_samples)

        E_mean = np.mean(energy_array)
        E2_mean = np.mean(energy_array ** 2)
        M_mean = np.mean(magnetisation_array)
        M2_mean = np.mean(magnetisation_array ** 2)

        C = (E2_mean - E_mean ** 2) / (T ** 2)
        X = (M2_mean - M_mean ** 2) / (T ** 2)

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

def run_single_L(L):
    print(f"\nRunning simulation for lattice size L = {L}...")
    results = run_simulation(L, T_values, seed=13)
    return (L, results)

if __name__ == '__main__':
    lattice_sizes = [20, 40, 60]
    with mp.Pool(processes=len(lattice_sizes)) as pool:
        output = list(tqdm(pool.imap(run_single_L, lattice_sizes), total=len(lattice_sizes)))

    simulation_results = dict(output)

    np.savez_compressed("ising_simulation_results.npz",
                        simulation_results=simulation_results,
                        T_values=T_values)
    print("\nSimulation data saved to 'ising_simulation_results.npz'")
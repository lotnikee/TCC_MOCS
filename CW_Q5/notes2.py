import numpy as np
import random
from tqdm import tqdm
import multiprocessing as mp
from numba import njit

# Temperature range with finer sampling near the critical point
T_low = np.linspace(1.5, 2.0, 10, endpoint=False)
T_core = np.linspace(2.0, 2.6, 50)
T_high = np.linspace(2.6, 3.5, 10, endpoint=False)
T_values = np.concatenate((T_low, T_core, T_high))

# Define simulation parameters for each lattice size
@njit
def set_simulation_parameters(L):
    if L == 20:
        return 10000, 0.25, 5
    elif L == 40:
        return 20000, 0.25, 10
    else:
        return 50000, 0.3, 10

# Initial lattice energy (right and bottom neighbors only to avoid double-counting)
@njit
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

# Single Metropolis step
@njit
def metropolis_step(lattice, i, j, T):
    L = lattice.shape[0]
    spin = lattice[i, j]
    neighbours = lattice[(i - 1) % L, j] + lattice[(i + 1) % L, j] + \
                 lattice[i, (j - 1) % L] + lattice[i, (j + 1) % L]
    delta_E = 2 * spin * neighbours
    if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
        lattice[i, j] *= -1
        return delta_E
    return 0.0

# JIT-compiled inner simulation run
@njit
def run_single_mc_run(L, T, total_steps, equil_steps, N, seed):
    np.random.seed(seed)
    lattice = np.random.choice(np.array([-1, 1]), size=(L, L))
    E = calculate_initial_energy(lattice)

    equil_limit = equil_steps * N
    total_mc_steps = total_steps * N
    sample_interval = N

    energy_samples = []
    magnetisation_samples = []

    for step in range(1, total_mc_steps + 1):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        E += metropolis_step(lattice, i, j, T)

        if step > equil_limit and step % sample_interval == 0:
            energy_samples.append(E / N)
            M = np.sum(lattice)
            magnetisation_samples.append(np.abs(M) / N)

    return np.array(energy_samples), np.array(magnetisation_samples)

# Run all MC runs for one lattice size
def run_simulation(L, T_values, seed=13):
    total_steps, equil_fraction, n_runs = set_simulation_parameters(L)
    equil_steps = int(equil_fraction * total_steps)
    N = L * L

    energy_values, magnetisation_values = [], []
    heat_capacity, magnetic_susceptibility = [], []

    for T in tqdm(T_values, desc=f"Simulating L = {L}", dynamic_ncols=True):
        all_energy_samples = []
        all_magnetisation_samples = []

        for run in range(n_runs):
            run_seed = seed + run
            energy_array, magnetisation_array = run_single_mc_run(L, T, total_steps, equil_steps, N, run_seed)
            all_energy_samples.extend(energy_array)
            all_magnetisation_samples.extend(magnetisation_array)

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

# Wrapper to run one L in parallel
def run_single_L(L):
    print(f"\nRunning simulation for lattice size L = {L}...")
    results = run_simulation(L, T_values, seed=13)
    return (L, results)

# Main parallel execution block
if __name__ == '__main__':
    lattice_sizes = [20, 40, 60]
    with mp.Pool(processes=len(lattice_sizes)) as pool:
        output = list(tqdm(pool.imap(run_single_L, lattice_sizes), total=len(lattice_sizes)))

    simulation_results = dict(output)

    np.savez_compressed("optimised_simulation_results.npz",
                        simulation_results=simulation_results,
                        T_values=T_values)
    print("\nSimulation data saved to 'optimised_simulation_results.npz'")

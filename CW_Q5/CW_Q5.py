import numpy as np 
import random 
from tqdm import tqdm 
import multiprocessing as mp 
from numba import njit
### Note that the numba and multiprocessing packages were used to cut down on total runtime 
### This allowed the simulation to be scaled up significantly, which in turn led to much better results

### Temperature range for sampling around the critical temperature
T_low = np.linspace(1.5, 2.0, 10, endpoint=False)
T_core = np.linspace(2.0, 2.6, 50)
T_high = np.linspace(2.6, 3.5, 10, endpoint=False)
T_values = np.concatenate((T_low, T_core, T_high))

### Define a simulation function 
### Function takes a lattice size and returns its Monte Carlo steps, the ratio used for equilibration and the number of independent runs
@njit
def set_simulation_parameters(L):
    if L == 20:
        return 10000, 0.25, 5
    elif L == 40:
        return 20000, 0.25, 10
    else: 
        return 50000, 0.3, 10
    
### Define an initial lattice and calculate its energy, avoiding double counting
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

### Define what a Metropolis step looks like 
@njit 
def metropolis_step(lattice, i, j, T):
    L = lattice.shape[0]
    spin = lattice[i, j]
    neighbours = lattice[(i - 1) % L, j] + lattice[(i + 1) % L, j] + lattice[i, (j - 1) % L] + lattice[i, (j + 1) % L]
    delta_E = 2 * spin * neighbours
    ### Metropolis Monte Carlo acceptance criterion: accept if exothermic or less than exp(-∆E/(T*k_B))
    if delta_E <0 or np.random.rand() < np.exp(-delta_E / T):
        ### If state accepted, flip spin 
        lattice[i, j] *= -1
        return delta_E
    return 0.0

### Define what a simulation run looks like 
@njit 
def run_single_mc(L, T, total_steps, equil_steps, N, seed):
    np.random.seed(seed)
    ### Randomly assign spins to an L X L lattice
    lattice = np.random.choice(np.array([-1, 1]), size=(L, L))
    ### Calculate current lattice energy
    E = calculate_initial_energy(lattice)

    ### Define start and endpoint for the simulation 
    equil_limit = equil_steps * N
    total_mc_steps = total_steps * N
    sample_interval = N

    ### Set up empty lists to capture energy and magnetisation values 
    energy_samples = []
    magnetisation_samples = []

    ### Determine how the lattice sites are chosen (randomly)
    for step in range(1, total_mc_steps + 1):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        E += metropolis_step(lattice, i, j, T)

        ### Ensure that results are only appended to the list after equilibration (i.e. during the production phase)
        ### Note that results are normalised by spin, i.e. energy and magnetisation per spin
        if step > equil_limit and step % sample_interval == 0:
            energy_samples.append(E / N)
            M = np.sum(lattice)
            magnetisation_samples.append(np.abs(M) / N)
    return np.array(energy_samples), np.array(magnetisation_samples)

### Run all the Metropolis Monte Carlo steps for one lattice size
def run_simulation(L, T_values, seed=13):
    total_steps, equil_fraction, n_runs = set_simulation_parameters(L)
    equil_steps = int(equil_fraction * total_steps)
    N = L * L 

    energy_values, magnetisation_values = [], []
    heat_capacity, magnetic_susceptibility = [], []

    ### Run simulation like this to allow for progress update bars while the simulation is running. Great way of figuring out how to balance runtime and scale of the simulation 
    for T in tqdm(T_values, desc=f"Simulating L = {L}", dynamic_ncols=True):
        all_energy_samples = []
        all_magnetisation_samples = []

        ### Set up the multiple independent runs, assinging each run its own random seed to ensure each run is independent of the other
        for run in range(n_runs):
            run_seed = seed + run
            energy_array, magnetisation_array = run_single_mc(L, T, total_steps, equil_steps, N, run_seed)
            all_energy_samples.extend(energy_array)
            all_magnetisation_samples.extend(magnetisation_array)

        ### Collect the energy and magnetisation samples from all the independent runs 
        energy_array = np.array(all_energy_samples)
        magnetisation_array = np.array(all_magnetisation_samples)

        ### Taking expectation values of E, E^2, M and M^2
        E_mean = np.mean(energy_array)
        E_square_mean = np.mean(energy_array ** 2)
        M_mean = np.mean(magnetisation_array)
        M_square_mean = np.mean(magnetisation_array ** 2)

        ### Calculating heat capacity and magnetic susceptibility 
        C = (E_square_mean - E_mean ** 2) / (T ** 2)
        X = (M_square_mean - M_mean ** 2) / (T ** 2)

        ### Append lists with expectation values and calculated observables 
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

### Define a function to run the lattice sizes in parallel 
def run_single_L(L):
    print(f"\nRunning simulation for lattice size L = {L}...")
    results = run_simulation(L, T_values, seed=13)
    return (L, results)

### Main parallel execution 
if __name__ == '__main__':
    lattice_sizes = [20, 40, 60]
    with mp.Pool(processes=len(lattice_sizes)) as pool:
        output = list(tqdm(pool.imap(run_single_L, lattice_sizes), total=len(lattice_sizes)))
    simulation_results = dict(output)

    ### Save results in a .npz file for further analysis 
    np.savez_compressed("optimised_simulation_results.npz",
                        simulation_results = simulation_results,
                        T_values = T_values)
    print("\nSimulation data saved to 'optimised_simulation_results.npz")
import matplotlib.pyplot as plt 
import numpy as np 
import random 

# Set-up and parameters
k = 1.0 
beta = 1.0
N_steps = 10**6
equilibration_steps = N_steps // 2

# Initial conditions
accepted_x = []
current_x = random.random() # avoid underflow

def potential_energy(x, k):
    return 0.5 * k * x**2

accepted_moves = 0  # optional: track acceptance rate

# Metropolis-Hastings MC loop
for step in range(1, N_steps + 1):

    # Generate phi ∈ [1, 1.1]
    phi = random.uniform(1.0, 1.1)
    if random.random() < 0.5:
        phi = 1.0 / phi

    proposed_x = current_x * phi

    # Skip if out of domain or dangerously small
    if proposed_x <= 1.0 and proposed_x >= 1e-10:

        # Energies
        E_x = potential_energy(current_x, k)
        E_x_new = potential_energy(proposed_x, k)

        # π(x) and π(x')
        pi_x = np.exp(-beta * E_x)
        pi_x_new = np.exp(-beta * E_x_new)

        # Proposal ratio (Jacobian)
        proposal_ratio = abs(proposed_x / current_x)

        # Metropolis-Hastings acceptance criterion
        A = min(1, (pi_x_new / pi_x) * proposal_ratio)

        if random.random() < A:
            current_x = proposed_x
            accepted_moves += 1

    # Record after equilibration
    if step > equilibration_steps:
        accepted_x.append(current_x)

# Plot
if accepted_x:
    plt.figure()
    plt.hist(accepted_x, bins=200, range=(0, 1), edgecolor='black')
    plt.title("Histogram of accepted x' values (Metropolis-Hastings)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    print(f"Mean accepted x: {np.mean(accepted_x):.5f}")
    print(f"Acceptance rate: {accepted_moves / N_steps:.4f}")
else:
    print("No accepted samples collected.")

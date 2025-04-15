import matplotlib.pyplot as plt 
import numpy as np 
import random 

### Include random seeds for reproducibility 
random.seed(13)
np.random.seed(13)

### Set up the iteration steps for the Metropolis Monte Carlo simulation and delta constant 
delta = 0.1
t = 10**7
equilibration_steps = t // 2
production_steps = t // 2

### Set up an empty list to keep track of x values that have been accepted by the algorithm
accepted_x = []

### Assign an initial value to x 
current_x = random.random()
accepted = 0

### Run a short Metropolis Monte Carlo simulation
for step in range(0, t + 1):

    ### Define the displacement parameters 
    if random.random() < 0.5: 
        phi = random.uniform(1, 1 + delta)
    else: 
        phi = random.uniform(1/(1 + delta), 1)
    proposed_x = current_x * phi

    ### Determine whether or not the new position falls within the boundary conditions
    if 1.0e-3 < proposed_x < 1.0:
        acceptance_ratio = min(1.0, 1.0 / (phi**2))
        if random.random() < acceptance_ratio:
            current_x = proposed_x
            if step > equilibration_steps:
                accepted += 1

    ### If the simulation is in its production phase, keep track of accepted values for x and append the list
        if step > equilibration_steps:
            accepted_x.append(current_x)

### Determine the average proposed_x value 
print(np.mean(accepted_x))

### Print acceptance rate
print("Acceptance rate:", accepted / production_steps)

### Plot a histogram of all the accepted values to show uniformity
plt.figure()
plt.hist(accepted_x, bins="auto", edgecolor='black') 
plt.title("Distribution of Sampled 'x' Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()



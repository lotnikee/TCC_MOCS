import matplotlib.pyplot as plt 
import numpy as np 
import random 

### Define 1D Harmonic Oscillator constants in atomic units 
### Working in a Metropolis Monte Carlo simulation, so E = P.E.
k = 1.0 
beta = 1.0

### Determine the initial position of x by generating a random number between [0, 1)
random_x = random.random()
print(f"x = ", random_x)

### Define the displacement parameters 
delta = random.uniform(-2, 2)
print(f"Delta = ", delta)

delta_x = random_x + delta
print(f"x_1 = ", delta_x)

if 0 <= delta_x <= 1:
    ### Calculate the energy E(x)
    E_x = 0.5 * k * random_x**2
    print(f"Energy E(x) = ", E_x)

    ### Calcualte the energy E(x')
    E_x_new = 0.5 * k * delta_x**2
    print(f"Energy E(x') = ", E_x_new)

    ### Determine π(x) and π(x')
    π_x = np.exp(- beta * E_x)
    π_x_new = np.exp(- beta * E_x_new)

    ### Determine the ratio between π(x) and π(x')
    ratio = π_x_new / π_x
    print(f"Probability is:", ratio)

    ### Define the acceptance criterion 
    A = min(1, ratio)

    if 0 <= A <= 1: 
        print(True)
    else: 
        print(False)
else:
    print(False)








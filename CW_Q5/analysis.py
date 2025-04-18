import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# === Load simulation data ===
data = np.load("ising_simulation_results.npz", allow_pickle=True)
simulation_results = data['simulation_results'].item()
T_values = data['T_values']

# === Tc estimation ===
def estimate_Tc_all_methods(sim_results, T_vals):
    print("Tc estimates using three different methods:\n")
    for L, results in sim_results.items():
        chi = np.array(results['magnetic_susceptibility'])
        C = np.array(results['heat_capacity'])

        # 1. Maximum value method
        Tc_chi_max = T_vals[np.argmax(chi)]
        Tc_C_max = T_vals[np.argmax(C)]

        # 2. Gradient (steepest slope)
        dchi_dT = np.abs(np.gradient(chi, T_vals))
        dC_dT = np.abs(np.gradient(C, T_vals))
        Tc_chi_grad = T_vals[np.argmax(dchi_dT)]
        Tc_C_grad = T_vals[np.argmax(dC_dT)]

        # 3. Curvature (second derivative after smoothing)
        chi_smooth = savgol_filter(chi, window_length=7, polyorder=3)
        C_smooth = savgol_filter(C, window_length=7, polyorder=3)
        d2chi_dT2 = np.abs(np.gradient(np.gradient(chi_smooth, T_vals), T_vals))
        d2C_dT2 = np.abs(np.gradient(np.gradient(C_smooth, T_vals), T_vals))
        Tc_chi_curve = T_vals[np.argmax(d2chi_dT2)]
        Tc_C_curve = T_vals[np.argmax(d2C_dT2)]

        print(f"L = {L}:")
        print(f"  Max χ(T):       T_c = {Tc_chi_max:.5f}")
        print(f"  Max dχ/dT:      T_c = {Tc_chi_grad:.5f}")
        print(f"  Max d²χ/dT²:    T_c = {Tc_chi_curve:.5f}")
        print(f"  Max C(T):       T_c = {Tc_C_max:.5f}")
        print(f"  Max dC/dT:      T_c = {Tc_C_grad:.5f}")
        print(f"  Max d²C/dT²:    T_c = {Tc_C_curve:.5f}\n")

# === Unified plot of all observables ===
def plot_all_observables(sim_results, T_vals):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    labels = ['energy', 'magnetisation', 'heat_capacity', 'magnetic_susceptibility']
    ylabels = [
        'Mean Energy per Spin ⟨E⟩',
        'Mean Magnetisation per Spin ⟨|M|⟩',
        'Heat Capacity per Spin, C',
        'Magnetic Susceptibility per Spin, χ'
    ]
    titles = [
        'Energy vs. Temperature',
        'Magnetisation vs. Temperature',
        'Heat Capacity vs. Temperature',
        'Magnetic Susceptibility vs. Temperature'
    ]

    for idx, key in enumerate(labels):
        ax = axs[idx]
        for L, results in sim_results.items():
            ax.plot(T_vals, results[key], label=f"L = {L}")
        ax.set_xlabel("Temperature, T")
        ax.set_ylabel(ylabels[idx])
        ax.set_title(titles[idx])
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()
        if key in ['heat_capacity', 'magnetic_susceptibility']:
            ax.set_xscale("log")

    plt.tight_layout()
    plt.show()

# === Run analysis ===
estimate_Tc_all_methods(simulation_results, T_values)
plot_all_observables(simulation_results, T_values)



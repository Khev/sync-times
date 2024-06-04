import os
import subprocess

# Parameters
Ns = [100]  # make sure a perfect square
N_trials_list = [10**5]
scripts = [
    "tsync_kuramoto_mean_field.py",
    "tsync_kuramoto_nonidentical.py",
    "tsync_kuramoto_kij.py",
    "tsync_kuramoto_noise.py",
    "tsync_kuramoto_graphs.py",
    "tsync_make_plots.py"  # Adding the plot script
]

labels = ['mean field', 'omega_i', 'Kij', 'noise', 'graphs', 'make plots']

# Color for labels
color = "\033[95m"  # Magenta
reset_color = "\033[0m"

# Ensure parallel and num_workers are used
parallel = True
num_workers = 9

# Loop over the combinations of N and N_trials
for N in Ns:
    for N_trials in N_trials_list:
        for label, script in zip(labels, scripts):
            # Print label with color
            print(f"{color}-----------------------------")
            print(f"{label}")
            print(f"-----------------------------{reset_color}")

            command = [
                "python", script,
                f"--N={N}",
                f"--N_trials={N_trials}",
                "--parallel",
                f"--num_workers={num_workers}"
            ]

            # Execute the command
            subprocess.run(command)

print("All scripts executed for all parameter combinations.")

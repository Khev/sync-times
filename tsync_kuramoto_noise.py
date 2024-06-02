import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, norm
import argparse
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Function to simulate the Kuramoto model with noise using Euler-Maruyama method
def kuramoto_model_with_noise(theta, omega, K, D, dt, noise_type):
    Z = np.mean(np.exp(1j * theta))
    deterministic_part = omega + K * np.imag(Z * np.exp(-1j * theta))
    if noise_type == 'gaussian':
        noise = np.random.normal(0, np.sqrt(D * dt), size=theta.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-np.sqrt(D * dt), np.sqrt(D * dt), size=theta.shape)
    stochastic_part = noise
    dtheta_dt = deterministic_part + stochastic_part
    return dtheta_dt

def euler_maruyama_kuramoto_model_with_noise(theta, t, omega, K, D, dt, noise_type):
    dtheta = kuramoto_model_with_noise(theta, omega, K, D, dt, noise_type)
    theta_new = theta + dtheta * dt
    return theta_new

def run_trial(args):
    N, Tmax, dt, K, omega, D, noise_type, epsilon = args
    theta0 = np.random.uniform(0, 2*np.pi, N)
    t = 0
    theta = theta0
    while t < Tmax:
        theta_new = euler_maruyama_kuramoto_model_with_noise(theta, t, omega, K, D, dt, noise_type)
        t += dt
        R = np.abs(np.mean(np.exp(1j * theta_new)))
        if np.abs(R - 1) <= epsilon:
            return t
        theta = theta_new    
    return Tmax

def find_Tsync_with_noise(N, Tmax, dt, K, omega, D, noise_type, N_trials=1000, epsilon=0.05, parallel=False, num_workers=1):
    if parallel:
        with Pool(num_workers) as pool:
            args = [(N, Tmax, dt, K, omega, D, noise_type, epsilon) for _ in range(N_trials)]
            T_syncs = pool.map(run_trial, args)
    else:
        T_syncs = []
        progress_step = max(N_trials // 10, 1)
        for trial in range(N_trials):
            if trial % progress_step == 0:
                logging.info(f"    Running trial {trial + 1} of {N_trials}...")
            T_syncs.append(run_trial((N, Tmax, dt, K, omega, D, noise_type, epsilon)))
    return T_syncs

def plot_histogram(Tsync, title, filename):
    plt.hist(Tsync, density=True, bins=20, alpha=0.6, label='Synchronization Times')
    params = gumbel_r.fit(Tsync)
    x = np.linspace(min(Tsync), max(Tsync), 100)
    pdf = gumbel_r.pdf(x, *params)
    plt.plot(x, pdf, 'r-', lw=2, label='Gumbel PDF')
    plt.title(title)
    plt.xlabel('Synchronization Time')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main(N=100, N_trials=1000, T_max=500, dt=0.005, K=5.0, parallel=False, num_workers=1):
    mu = 0  # Mean frequency
    data_dir = "data_tsync_kuramoto_noise"

    D_values = [0.1, 1.0]
    noise_types = ['gaussian', 'uniform']
    results = {}
    
    total_start_time = datetime.now()
    for D in D_values:
        for noise_type in noise_types:
            logging.info(f"Processing noise type: {noise_type}, D: {D}")
            
            omega = np.zeros(N)  # Constant natural frequencies
            Tsync = find_Tsync_with_noise(N, T_max, dt, K, omega, D, noise_type, N_trials=N_trials, parallel=parallel, num_workers=num_workers)
            if Tsync:
                results[(noise_type, D)] = Tsync
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                data_filename = os.path.join(data_dir, f"mean_field_N_{N}_{noise_type}_D_{D}_Ntrial_{N_trials}.csv")
                df = pd.DataFrame({'Synchronization Time': Tsync})
                df.to_csv(data_filename, index=False)
                #plot_histogram(Tsync, f'Mean Field N={N}, {noise_type} noise, D={D}', f'{data_filename[:-4]}.png')
                logging.info(f"Finished processing noise type: {noise_type}, D: {D}")
            else:
                logging.info(f"No valid synchronization times for noise type: {noise_type}, D: {D}")

    logging.info(f"Total time taken for all noise types: {str(datetime.now() - total_start_time).split('.')[0]}")

    # Plot all histograms in a grid
    fig, axes = plt.subplots(len(D_values), len(noise_types), figsize=(20, 15))
    for ax, ((noise_type, D), Tsync) in zip(axes.flat, results.items()):
        ax.hist(Tsync, density=True, bins=20, alpha=0.6, label='Synchronization Times')
        params = gumbel_r.fit(Tsync)
        x = np.linspace(min(Tsync), max(Tsync), 100)
        pdf = gumbel_r.pdf(x, *params)
        ax.plot(x, pdf, 'r-', lw=2, label='Gumbel PDF')
        ax.set_title(f'N={N}, Ntrial={N_trials} {noise_type.capitalize()} Noise, D={D}')
        ax.set_xlabel('Synchronization Time')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'histograms_N_{N}_Ntrial_{N_trials}.png'))
    #plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kuramoto Synchronization Simulation with Noise')
    parser.add_argument('--N', type=int, default=100, help='Number of nodes in the network')
    parser.add_argument('--N_trials', type=int, default=100, help='Number of trials for synchronization')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using multiprocessing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    args = parser.parse_args()

    main(N=args.N, N_trials=args.N_trials, parallel=args.parallel, num_workers=args.num_workers)
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

# Function to simulate the Kuramoto model
def kuramoto_model(theta, t, omega, K):
    Z = np.mean(np.exp(1j * theta))
    dtheta_dt = omega + K * np.imag(Z * np.exp(-1j * theta))
    return dtheta_dt

def rk4_kuramoto_model(theta, t, omega, K, dt):
    k1 = dt * kuramoto_model(theta, t, omega, K)
    k2 = dt * kuramoto_model(theta + 0.5 * k1, t + 0.5 * dt, omega, K)
    k3 = dt * kuramoto_model(theta + 0.5 * k2, t + 0.5 * dt, omega, K)
    k4 = dt * kuramoto_model(theta + k3, t + dt, omega, K)
    theta_new = theta + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return theta_new

def run_trial(args):
    N, Tmax, dt, K, omega, epsilon, r_ss = args
    theta0 = np.random.uniform(0, 2*np.pi, N)
    t = 0
    theta = theta0
    while t < Tmax:
        theta_new = rk4_kuramoto_model(theta, t, omega, K, dt)
        t += dt
        R = np.abs(np.mean(np.exp(1j * theta_new)))
        if np.abs(R - r_ss) <= epsilon:
            return t
        theta = theta_new    
    return Tmax

def find_Tsync(N, Tmax, dt, K, omega, mu, gamma, N_trials=1000, epsilon=0.05, parallel=False, num_workers=1):
    r_ss = np.sqrt(1 - 2 * gamma / K)
    if r_ss <= 0:
        logging.info(f"Invalid r_ss for mu={mu}, gamma={gamma}, K={K}")
        return []
    
    if parallel:
        with Pool(num_workers) as pool:
            args = [(N, Tmax, dt, K, omega, epsilon, r_ss) for _ in range(N_trials)]
            T_syncs = pool.map(run_trial, args)
    else:
        T_syncs = []
        progress_step = max(N_trials // 10, 1)
        for trial in range(N_trials):
            if trial % progress_step == 0:
                logging.info(f"    Running trial {trial + 1} of {N_trials}...")
            T_syncs.append(run_trial((N, Tmax, dt, K, omega, epsilon, r_ss)))
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

# Function to generate deterministic Cauchy random variables
def deterministic_cauchy(size, loc, scale):
    p = np.linspace(1/(size+1), 1-1/(size+1), size)
    return loc + scale * np.tan(np.pi * (p - 0.5))

# Function to generate deterministic Gaussian random variables
def deterministic_gaussian(size, loc, scale):
    p = np.linspace(1/(size+1), 1-1/(size+1), size)
    return norm.ppf(p, loc=loc, scale=scale)

def main(N=100, N_trials=1000, T_max=500, dt=0.005, K=5.0, parallel=False, num_workers=1):
    mu = 0  # Mean frequency
    data_dir = "data_tsync_kuramoto_omega_i"

    gamma_values = [0.1, 0.2]
    dists = ['uniform', 'gaussian', 'cauchy']
    results = {}
    
    total_start_time = datetime.now()
    for gamma in gamma_values:
        for distribution in dists:
            logging.info(f"Processing distribution: {distribution}, gamma: {gamma}")
            
            if distribution == 'uniform':
                omega = np.linspace(mu - gamma, mu + gamma, N)
            elif distribution == 'gaussian':
                omega = deterministic_gaussian(N, mu, gamma)
            elif distribution == 'cauchy':
                omega = deterministic_cauchy(N, mu, gamma)
            
            Tsync = find_Tsync(N, T_max, dt, K, omega, mu, gamma, N_trials=N_trials, parallel=parallel, num_workers=num_workers)
            if Tsync:
                results[(distribution, gamma)] = Tsync
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                data_filename = os.path.join(data_dir, f"mean_field_N_{N}_{distribution}_gamma_{gamma}_Ntrial_{N_trials}.csv")
                df = pd.DataFrame({'Synchronization Time': Tsync})
                df.to_csv(data_filename, index=False)
                #plot_histogram(Tsync, f'Mean Field N={N}, {distribution}, gamma={gamma}', f'{data_filename[:-4]}.png')
                logging.info(f"Finished processing distribution: {distribution}, gamma: {gamma}")
            else:
                logging.info(f"No valid synchronization times for distribution: {distribution}, gamma: {gamma}")

    logging.info(f"Total time taken for all distributions: {str(datetime.now() - total_start_time).split('.')[0]}")

    # Plot all histograms in a grid
    fig, axes = plt.subplots(len(gamma_values), 3, figsize=(20, 15))
    for ax, ((distribution, gamma), Tsync) in zip(axes.flat, results.items()):
        ax.hist(Tsync, density=True, bins=20, alpha=0.6, label='Synchronization Times')
        params = gumbel_r.fit(Tsync)
        x = np.linspace(min(Tsync), max(Tsync), 100)
        pdf = gumbel_r.pdf(x, *params)
        ax.plot(x, pdf, 'r-', lw=2, label='Gumbel PDF')
        ax.set_title(f'N={N}, Ntrial={N_trials} {distribution}, gamma={gamma}')
        ax.set_xlabel('Synchronization Time')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'histograms_N_{N}_Ntrial_{N_trials}.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kuramoto Synchronization Simulation with Omega Distribution')
    parser.add_argument('--N', type=int, default=100, help='Number of nodes in the network')
    parser.add_argument('--N_trials', type=int, default=2, help='Number of trials for synchronization')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using multiprocessing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    args = parser.parse_args()

    main(N=args.N, N_trials=args.N_trials, parallel=args.parallel, num_workers=args.num_workers)
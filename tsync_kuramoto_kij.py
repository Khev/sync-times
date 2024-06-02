import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, expon
import argparse
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def kuramoto_model(theta, t, omega, K):
    N = len(theta)
    complex_order = np.exp(-1j * theta)
    interaction_term = np.dot(K, np.exp(1j * theta))
    dtheta_dt = omega + (complex_order * interaction_term).imag / N
    return dtheta_dt

def rk4_kuramoto_model(theta, t, omega, K, dt):
    k1 = dt * kuramoto_model(theta, t, omega, K)
    k2 = dt * kuramoto_model(theta + 0.5 * k1, t + 0.5 * dt, omega, K)
    k3 = dt * kuramoto_model(theta + 0.5 * k2, t + 0.5 * dt, omega, K)
    k4 = dt * kuramoto_model(theta + k3, t + dt, omega, K)
    theta_new = theta + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return theta_new

def run_trial(args):
    N, Tmax, dt, K, omega, epsilon = args
    theta0 = np.random.uniform(0, 2*np.pi, N)
    t = 0
    theta = theta0
    while t < Tmax:
        theta_new = rk4_kuramoto_model(theta, t, omega, K, dt)
        t += dt
        R = np.abs(np.mean(np.exp(1j * theta_new)))
        if R >= 1 - epsilon:
            return t
        theta = theta_new
    return Tmax

def find_Tsync(N, Tmax, dt, K, omega, N_trials=1000, epsilon=0.05, parallel=False, num_workers=1):
    if parallel:
        with Pool(num_workers) as pool:
            args = [(N, Tmax, dt, K, omega, epsilon) for _ in range(N_trials)]
            T_syncs = pool.map(run_trial, args)
    else:
        T_syncs = []
        progress_step = max(N_trials // 10, 1)
        for trial in range(N_trials):
            if trial % progress_step == 0:
                logging.info(f"    Running trial {trial + 1} of {N_trials}...")
            T_syncs.append(run_trial((N, Tmax, dt, K, omega, epsilon)))
    return T_syncs

def generate_exponential_k_matrix(N, mean):
    K = expon.rvs(scale=mean, size=(N, N))
    np.fill_diagonal(K, 0)  # Ensuring no self-interaction
    return K

def generate_uniform_k_matrix(N, mean):
    K = np.random.uniform(0, 2*mean, size=(N, N))
    np.fill_diagonal(K, 0)  # Ensuring no self-interaction
    return K

def generate_lognormal_k_matrix(N, mean, sigma):
    K = np.random.lognormal(mean, sigma, size=(N, N))
    np.fill_diagonal(K, 0)  # Ensuring no self-interaction
    return K

def generate_beta_k_matrix(N, a, b, scale):
    K = np.random.beta(a, b, size=(N, N)) * scale  # Scale to match support
    np.fill_diagonal(K, 0)  # Ensuring no self-interaction
    return K

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

def main(N=100, N_trials=1000, T_max=100, dt=0.005, parallel=False, num_workers=1):
    omega = np.zeros(N)
    graph = 'mean_field'
    data_dir = "data_tsync_kuramoto_kij"

    # Define distributions with specific mu values
    distributions = {
        'uniform': 1,
        'beta': 2,
        'exponential': 3
    }

    # Generate and synchronize networks
    results = {}

    total_start_time = datetime.now()
    for distribution, mu in distributions.items():
        logging.info(f"Processing distribution: {distribution}, mu: {mu}")
        
        if distribution == 'uniform':
            K = generate_uniform_k_matrix(N, mu)
            K_label = f"Kij = Uniform(0, {2*mu})"
        elif distribution == 'exponential':
            K = generate_exponential_k_matrix(N, mu)
            K_label = f"Kij = Exponential({mu})"
        elif distribution == 'lognormal':
            K = generate_lognormal_k_matrix(N, mu, 0.5)
            K_label = f"Kij = LogNormal(mean={mu}, sigma=0.5)"
        elif distribution == 'beta':
            K = generate_beta_k_matrix(N, 2, 2, 2 * mu)
            K_label = f"Kij = Beta(a=2, b=2, scaled to 0-{2*mu})"
        
        Tsync = find_Tsync(N, T_max, dt, K, omega, N_trials=N_trials, parallel=parallel, num_workers=num_workers)
        results[(distribution, mu)] = (Tsync, K_label)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_filename = os.path.join(data_dir, f"{graph}_N_{N}_{distribution}_mu_{mu:.2f}_Ntrial_{N_trials}.csv")
        df = pd.DataFrame({'Synchronization Time': Tsync})
        df.to_csv(data_filename, index=False)
        #plot_histogram(Tsync, f'{graph.capitalize()} N={N}, {K_label}', f'{data_filename[:-4]}.png')
        logging.info(f"Finished processing distribution: {distribution}, mu: {mu}")

    logging.info(f"Total time taken for all distributions: {str(datetime.now() - total_start_time).split('.')[0]}")

    # Plot all histograms in a grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for i, (ax, ((distribution, mu), (Tsync, K_label))) in enumerate(zip(axes.flat, results.items())):
        ax.hist(Tsync, density=True, bins=20, alpha=0.6, label='Synchronization Times')
        params = gumbel_r.fit(Tsync)
        x = np.linspace(min(Tsync), max(Tsync), 100)
        pdf = gumbel_r.pdf(x, *params)
        ax.plot(x, pdf, 'r-', lw=2, label='Gumbel PDF')
        ax.set_title(f'N={N}, Ntrial = {N_trials}, {K_label}')
        if i % 2 == 0:  # Only label the y-axis of the leftmost column
            ax.set_ylabel('Probability Density')
        if i >= len(distributions) - 2:  # Only label the x-axis of the lowest row
            ax.set_xlabel('Synchronization Time')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'histograms_N_{N}_Ntrial_{N_trials}.png'))
    #plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kuramoto Synchronization Simulation with Various K Distributions')
    parser.add_argument('--N', type=int, default=100, help='Number of nodes in the network')
    parser.add_argument('--N_trials', type=int, default=100, help='Number of trials for synchronization')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using multiprocessing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    args = parser.parse_args()

    main(N=args.N, N_trials=args.N_trials, parallel=args.parallel, num_workers=args.num_workers)
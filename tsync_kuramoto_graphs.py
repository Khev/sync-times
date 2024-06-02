import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
import networkx as nx
import logging
from datetime import datetime
import argparse
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Function to simulate the Kuramoto model
def kuramoto_model(theta, t, omega, K, A):
    N = len(theta)
    complex_order = np.exp(-1j * theta)
    interaction_term = np.dot(A, np.exp(1j * theta))
    dtheta_dt = omega + K * (complex_order * interaction_term).imag / N
    return dtheta_dt

def rk4_kuramoto_model(theta, t, omega, K, A, dt):
    k1 = dt * kuramoto_model(theta, t, omega, K, A)
    k2 = dt * kuramoto_model(theta + 0.5 * k1, t + 0.5 * dt, omega, K, A)
    k3 = dt * kuramoto_model(theta + 0.5 * k2, t + 0.5 * dt, omega, K, A)
    k4 = dt * kuramoto_model(theta + k3, t + dt, omega, K, A)
    theta_new = theta + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return theta_new

def run_trial(args):
    N, Tmax, dt, K, A, omega, epsilon = args
    theta0 = np.random.uniform(0, 2 * np.pi, N)
    t = 0
    theta = theta0
    while t < Tmax:
        theta_new = rk4_kuramoto_model(theta, t, omega, K, A, dt)
        t += dt
        R = np.abs(np.mean(np.exp(1j * theta_new)))
        if R >= 1 - epsilon:
            return t
        theta = theta_new    
    return Tmax

def find_Tsync(N, Tmax, dt, K, A, omega, N_trials=1000, epsilon=0.05, parallel=False, num_workers=1):
    if parallel:
        with Pool(num_workers) as pool:
            args = [(N, Tmax, dt, K, A, omega, epsilon) for _ in range(N_trials)]
            T_syncs = pool.map(run_trial, args)
    else:
        T_syncs = []
        progress_step = max(N_trials // 10, 1)
        for trial in range(N_trials):
            if trial % progress_step == 0:
                logging.info(f"    Running trial {trial + 1} of {N_trials}...")
            T_syncs.append(run_trial((N, Tmax, dt, K, A, omega, epsilon)))
    return T_syncs

def make_chain(N, k):
    G = nx.path_graph(N)
    for i in range(N):
        for j in range(1, k + 1):
            G.add_edge(i, (i + j) % N)
    return nx.to_numpy_array(G)

def make_ER(N, p):
    G = nx.erdos_renyi_graph(N, p)
    return nx.to_numpy_array(G)

def make_ring(N, k):
    G = nx.watts_strogatz_graph(N, k, p=0)
    return nx.to_numpy_array(G)

def make_lattice_2d(N, k):
    G = nx.grid_2d_graph(N, N, periodic=False)
    for (i, j) in G.nodes():
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                if (di != 0 or dj != 0) and (abs(di) + abs(dj) <= k):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < N:
                        G.add_edge((i, j), (ni, nj))
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return nx.to_numpy_array(G)

def make_star(N):
    G = nx.star_graph(N - 1)
    return nx.to_numpy_array(G)

def make_small_world(N, k, p):
    G = nx.watts_strogatz_graph(N, k, p)
    return nx.to_numpy_array(G)

def make_scale_free(N, m):
    G = nx.barabasi_albert_graph(N, m)
    return nx.to_numpy_array(G)

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

def main(N=25, N_trials=1000, T_max=1000, dt=0.005, K=1.0, parallel=False, num_workers=1):
    omega = np.zeros(N)
    data_dir = "data_tsync_kuramoto_graphs"

    topologies = ['mean_field', 'ring', 'chain', 'lattice_2d', 'ER', 'scale_free']
    generators = {
        'mean_field': lambda N, _: np.ones((N, N)),
        'ER': make_ER,
        'ring': make_ring,
        'chain': make_chain,
        'lattice_2d': make_lattice_2d,
        'scale_free': make_scale_free
    }
    params = {
        'mean_field': (N, 0), 
        'ER': (N, 0.5), 
        'ring': (N, int(0.4 * N)), 
        'chain': (N, int(0.4 * N)),
        'lattice_2d': (int(np.sqrt(N)), int(0.4 * np.sqrt(N))),
        'scale_free': (N, 4)
    }
    results = {}
    total_start_time = datetime.now()

    for topology in topologies:
        logging.info(f"Processing topology: {topology.capitalize()}")
        start_time = datetime.now()
        gen_params = params[topology]
        A = generators[topology](*gen_params)
        Tsync = find_Tsync(N, T_max, dt, K, A, omega, N_trials=N_trials, parallel=parallel, num_workers=num_workers)
        results[topology] = Tsync
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_filename = os.path.join(data_dir, f"{topology}_N_{N}_Ntrial_{N_trials}.csv")
        df = pd.DataFrame({'Synchronization Time': Tsync})
        df.to_csv(data_filename, index=False)
        #plot_histogram(Tsync, f'{topology.capitalize()} K, N, Ntrial = {K}, {N}, {N_trials}', f'{data_filename[:-4]}.png')
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Finished processing topology: {topology.capitalize()}")
        logging.info(f"Time taken for {topology.capitalize()}: {str(duration).split('.')[0]}")

    total_duration = datetime.now() - total_start_time
    logging.info(f"Total time taken for all topologies: {str(total_duration).split('.')[0]}")

    # Plot all histograms in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, (topology, Tsync) in zip(axes.flat, results.items()):
        ax.hist(Tsync, density=True, bins=20, alpha=0.6, label='Synchronization Times')
        params = gumbel_r.fit(Tsync)
        x = np.linspace(min(Tsync), max(Tsync), 100)
        pdf = gumbel_r.pdf(x, *params)
        ax.plot(x, pdf, 'r-', lw=2, label='Gumbel PDF')
        ax.set_title(f'{topology.capitalize()} K, N, Ntrial = {K}, {N}, {N_trials}')
        ax.legend()
        ax.grid(True)
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('Synchronization Time')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel('Probability Density')

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'histograms_N_{N}_Ntrial_{N_trials}.png'))
    #plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kuramoto Synchronization Simulation')
    parser.add_argument('--N', type=int, default=25, help='Number of nodes in the network')
    parser.add_argument('--N_trials', type=int, default=100, help='Number of trials for synchronization')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using multiprocessing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    args = parser.parse_args()

    main(N=args.N, N_trials=args.N_trials, parallel=args.parallel, num_workers=args.num_workers)

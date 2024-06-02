import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
import networkx as nx
import logging
from datetime import datetime, timedelta

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

def find_Tsync(N, Tmax, dt, K, A, omega, N_trials=10, epsilon=0.05):
    T_syncs = []
    progress_step = max(N_trials // 10, 1)  # Ensure at least one progress print
    for trial in range(N_trials):
        if trial % progress_step == 0:
            logging.info(f"    Running trial {trial + 1} of {N_trials}...")
        theta0 = np.random.uniform(0, 2*np.pi, N)        
        t = 0
        theta = theta0
        while t < Tmax:
            theta_new = rk4_kuramoto_model(theta, t, omega, K, A, dt)
            t += dt
            R = np.abs(np.mean(np.exp(1j * theta_new)))
            if R >= 1 - epsilon:
                T_syncs.append(t)
                break
            theta = theta_new    
    return T_syncs

def make_chain(N, k):
    G = nx.path_graph(N)
    for i in range(N):
        for j in range(1, k+1):
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
        for di in range(-k, k+1):
            for dj in range(-k, k+1):
                if (di != 0 or dj != 0) and (abs(di) + abs(dj) <= k):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < N:
                        G.add_edge((i, j), (ni, nj))
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return nx.to_numpy_array(G)

def make_star(N):
    G = nx.star_graph(N-1)
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

def compute_aic(params, data):
    loc, scale = params
    n = len(data)
    log_likelihood = np.sum(gumbel_r.logpdf(data, loc=loc, scale=scale))
    k = 2  # Number of parameters for Gumbel distribution
    aic = 2 * k - 2 * log_likelihood
    return aic, log_likelihood

# Parameters
N_trials = 10**4
T_max = 1000
dt = 0.005
K = 1.0
data_dir = "data_tsync_kuramoto_graphs"


# Loop over different N values
N_values = [16,25,49]
topologies = ['mean_field', 'ER', 'ring', 'chain', 'lattice_2d', 'small_world', 'scale_free']
topologies = ['mean_field', 'ring', 'chain', 'lattice_2d', 'ER', 'scale_free', 'small_world'] # maybe drop SW?
generators = {
    'mean_field': lambda N, _: np.ones((N, N)),
    'ER': make_ER,
    'ring': make_ring,
    'chain': make_chain,
    'lattice_2d': make_lattice_2d,
    'star': make_star,
    'small_world': make_small_world,
    'scale_free': make_scale_free
}

# Generate and synchronize networks
total_start_time = datetime.now()
results = {N: {} for N in N_values}

for N in N_values:
    omega = np.zeros(N)
    params = {
        'mean_field': (N, 0), 
        'ER': (N, 0.5), 
        'ring': (N, int(0.4 * N)), 
        'chain': (N, int(0.4 * N)),
        'lattice_2d': (int(np.sqrt(N)), int(0.4 * np.sqrt(N))),
        'star': (N, ),
        'small_world': (N, int(0.4 * N), 0.1),
        'scale_free': (N, 4)
    }

    for topology in topologies:
        logging.info(f"Processing topology: {topology.capitalize()}")
        start_time = datetime.now()
        gen_params = params[topology]
        A = generators[topology](*gen_params)
        Tsync = find_Tsync(N, T_max, dt, K, A, omega, N_trials=N_trials)
        results[N][topology] = Tsync
        data_dir = "data_tsync_kuramoto"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_filename = os.path.join(data_dir, f"{topology}_N_{N}_K_{K:.2f}.csv")
        df = pd.DataFrame({'Synchronization Time': Tsync})
        df.to_csv(data_filename, index=False)
        plot_histogram(Tsync, f'{topology.capitalize()} N={N}, K={K:.2f}, Ntrials={N_trials}', f'{data_filename[:-4]}.png')
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        logging.info(f"Time taken for {topology.capitalize()}, N={N}: {str(timedelta(seconds=elapsed_time.total_seconds()))}\n")

    logging.info(f"Total time taken for all topologies: {datetime.now() - total_start_time}")

# Plot all histograms
fig, axes = plt.subplots(len(N_values), 2, figsize=(20, 10 * len(N_values)))

for i, N in enumerate(N_values):
    for j, (topology, Tsync) in enumerate(results[N].items()):
        ax_hist = axes[i, 0]
        ax_hist.hist(Tsync, density=True, bins=20, alpha=0.6, label=f'{topology.capitalize()}')
        params = gumbel_r.fit(Tsync)
        ax_hist.set_title(f'N={N}, Ntrials={N_trials}', fontsize=14)
        if ax_hist.is_last_row():
            ax_hist.set_xlabel('Synchronization Time', fontsize=14)
        if ax_hist.is_first_col():
            ax_hist.set_ylabel('Probability Density', fontsize=14)
        ax_hist.legend()
        ax_hist.grid(True)
        
        # Scaling collapse
        mean = params[0]
        scale = params[1]
        Tsync_standardized = (Tsync - mean) / scale
        hist, bin_edges = np.histogram(Tsync_standardized, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax_collapse = axes[i, 1]
        ax_collapse.plot(bin_centers, hist, 'o', label=f'{topology.capitalize()}', markerfacecolor='none')
        
    if ax_collapse.is_last_row():
        ax_collapse.set_xlabel('Standardized Synchronization Time', fontsize=14)
    if ax_collapse.is_first_col():
        ax_collapse.set_ylabel('Probability Density', fontsize=14)
    ax_collapse.grid(True)
    ax_collapse.legend()
    # Plot standard Gumbel distribution
    x_standard = np.linspace(-5, 15, 1000)
    standard_gumbel_pdf = gumbel_r.pdf(x_standard)
    ax_collapse.plot(x_standard, standard_gumbel_pdf, label='Standard Gumbel', color='black')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'collapse_N_{N}_Ntrial_{N_trials}.png'))
plt.show()


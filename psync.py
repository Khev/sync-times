import os
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import warnings
from multiprocessing import Pool, freeze_support

warnings.filterwarnings("ignore", \
                         category=FutureWarning, \
                    message="adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.")

class PeskinModelEventDriven:
    def __init__(self, N, S0, gamma, A):
        self.N = N
        self.S0 = S0
        self.gamma = gamma
        self.A = A
        self.x = np.random.rand(N)  # Initialize voltages randomly
        self.time = 0

    def next_event(self):
        # Calculate the time to the next firing event
        T_next_fire = (1 - self.x) / (self.S0 - self.gamma * self.x)
        min_T = np.min(T_next_fire)
        return min_T

    def update_voltages(self, T_next):
        if self.gamma == 0:
            # When gamma is 0, voltage increases linearly
            self.x += self.S0 * T_next
        else:
            # Update voltages using the exact solution of the differential equation
            self.x = (self.x - self.S0 / self.gamma) * np.exp(-self.gamma * T_next) + self.S0 / self.gamma

    def step(self):
        T_next = self.next_event()
        self.time += T_next

        # Update voltages
        self.update_voltages(T_next)

        # Find neurons that reached threshold
        fired = np.where(self.x >= 1)[0]

        # Emit pulses
        if len(fired) > 0:
            pulse = len(fired) / self.N
            for i in range(self.N):
                if i not in fired:
                    self.x[i] += pulse * np.sum(self.A[i, fired])

        # Reset all oscillators that reached or exceeded the threshold
        self.x[self.x >= 1] = 0

    def run(self, Tmax):
        self.time = 0
        self.x = np.random.rand(self.N)  # Re-initialize voltages randomly for each run
        history = []
        time_history = []
        cluster_history = []
        while self.time < Tmax:
            history.append(self.x.copy())
            time_history.append(self.time)
            cluster_history.append(self.compute_cluster_fraction())
            self.step()
            # Check for synchronization
            if self.check_synchronization():
                return np.array(history), np.array(time_history), np.array(cluster_history), self.time
        return np.array(history), np.array(time_history), np.array(cluster_history), -1  # Did not sync

    def compute_cluster_fraction(self):
        # Identify clusters of synchronized neurons
        unique_phases = np.unique(np.round(self.x, decimals=5))  # Using rounded values to identify clusters
        num_clusters = len(unique_phases)
        cluster_fraction = num_clusters / self.N
        return cluster_fraction

    def check_synchronization(self):
        # Check if all neurons are synchronized (i.e., have the same voltage)
        return np.allclose(self.x, self.x[0], atol=1e-5)

    def run_trials(self, N_trials, Tmax, parallel=False, num_workers=8):
        if parallel:
            with Pool(num_workers) as pool:
                results = pool.starmap(self.run, [(Tmax,) for _ in range(N_trials)])
        else:
            results = [self.run(Tmax) for _ in range(N_trials)]
        
        sync_count = sum(1 for result in results if result[3] != -1)
        sync_times = [result[3] for result in results]
        return sync_count, sync_times

class KuramotoModel:
    def __init__(self, N, K, A, dt):
        self.N = N
        self.K = K
        self.A = A
        self.dt = dt
        self.theta = np.random.rand(N) * 2 * np.pi  # Initialize phases randomly
        self.time = 0
        self.eps = 0.01

    def step(self):
        # Use complex exponentials to calculate the interaction term
        exp_theta = np.exp(1j * self.theta)
        interaction = np.dot(self.A, exp_theta)
        dtheta = (self.K / self.N) * np.imag(interaction * np.conj(exp_theta))
        self.theta += dtheta * self.dt
        self.theta = self.theta % (2 * np.pi)  # Keep phases within [0, 2π]

    def run(self, Tmax):
        self.time = 0
        self.theta = np.random.rand(self.N) * 2 * np.pi  # Re-initialize phases randomly for each run
        history = []
        time_history = []
        order_parameter_history = []
        max_steps = int(Tmax / self.dt)
        for step in range(max_steps):
            history.append(self.theta.copy())
            time_history.append(self.time)
            order_parameter_history.append(self.compute_order_parameter())
            self.step()
            self.time += self.dt
            # Check for synchronization
            if self.compute_order_parameter() > 1 - self.eps:  # Threshold for synchronization
                return np.array(history), np.array(time_history), np.array(order_parameter_history), self.time
        return np.array(history), np.array(time_history), np.array(order_parameter_history), -1  # Did not sync

    def compute_order_parameter(self):
        r = np.abs(np.sum(np.exp(1j * self.theta)) / self.N)
        return r

    def run_trials(self, N_trials, Tmax, parallel=False, num_workers=8):
        if parallel:
            with Pool(num_workers) as pool:
                results = pool.starmap(self.run, [(Tmax,) for _ in range(N_trials)])
        else:
            results = [self.run(Tmax) for _ in range(N_trials)]
        
        sync_count = sum(1 for result in results if result[3] != -1)
        sync_times = [result[3] for result in results]
        return sync_count, sync_times

def generate_1d_chain(N, k):
    G = nx.path_graph(N)
    for node in range(N):
        for neighbor in range(1, k+1):
            if node + neighbor < N:
                G.add_edge(node, node + neighbor)
    A = nx.adjacency_matrix(G).toarray()
    return np.array(A)

def generate_1d_ring(N, k):
    G = nx.cycle_graph(N)
    for node in range(N):
        for neighbor in range(1, k+1):
            G.add_edge(node, (node + neighbor) % N)
            G.add_edge(node, (node - neighbor) % N)
    A = nx.adjacency_matrix(G).toarray()
    return np.array(A)

def generate_2d_lattice(N, k):
    L = int(np.sqrt(N))
    G = nx.grid_2d_graph(L, L)
    for node in G.nodes():
        for neighbor in range(1, k+1):
            if node[0] + neighbor < L:
                G.add_edge(node, (node[0] + neighbor, node[1]))
            if node[1] + neighbor < L:
                G.add_edge(node, (node[0], node[1] + neighbor))
    A = nx.adjacency_matrix(nx.convert_node_labels_to_integers(G)).toarray()
    return np.array(A)

def generate_2d_periodic_lattice(N, k):
    L = int(np.sqrt(N))
    G = nx.grid_2d_graph(L, L, periodic=True)
    for node in G.nodes():
        for neighbor in range(1, k+1):
            G.add_edge(node, ((node[0] + neighbor) % L, node[1]))
            G.add_edge(node, (node[0], (node[1] + neighbor) % L))
    A = nx.adjacency_matrix(nx.convert_node_labels_to_integers(G)).toarray()
    return np.array(A)

def generate_er_graph(N, k):
    p = k / (N / 2)
    G = nx.erdos_renyi_graph(N, p)
    A = nx.adjacency_matrix(G).toarray()
    return np.array(A)

def soft_configuration_model(N, m, p):
    G = nx.complete_graph(m + 1)
    for t in range(m + 1, N):
        new_node_edges = []
        if np.random.rand() < p:
            new_node_edges = [(t, i) for i in range(t)]
        else:
            existing_nodes = list(G.nodes)
            degree_sum = sum(dict(G.degree(existing_nodes)).values())
            attachment_prob = [G.degree(i) / degree_sum for i in existing_nodes]
            new_node_edges = np.random.choice(existing_nodes, m, replace=False, p=attachment_prob)
            new_node_edges = [(t, i) for i in new_node_edges]
        G.add_edges_from(new_node_edges)
    A = nx.adjacency_matrix(G).toarray()
    return np.array(A)

def measure_sync_probability(graph_type, N, S0, K, p_values, N_trials, Tmax, gamma, dt, parallel=False, num_workers=8):
    sync_probabilities_peskin = []
    sync_times_peskin = []
    sync_probabilities_kuramoto = []
    sync_times_kuramoto = []
    for p in p_values:
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} - Starting p={p:.2f} for {graph_type}')
        if p == 0:
            sync_probabilities_peskin.append(0)
            sync_times_peskin.append([-1]*N_trials)
            sync_probabilities_kuramoto.append(0)
            sync_times_kuramoto.append([-1]*N_trials)
            continue

        k = int(p * (N // 2))
        if graph_type == "1d_chain":
            A = generate_1d_chain(N, k)
        elif graph_type == "1d_ring":
            A = generate_1d_ring(N, k)
        elif graph_type == "2d_lattice":
            A = generate_2d_lattice(N, k)
        elif graph_type == "2d_periodic_lattice":
            A = generate_2d_periodic_lattice(N, k)
        elif graph_type == "er_graph":
            A = generate_er_graph(N, k)
        elif graph_type == "soft_configuration_model":
            A = soft_configuration_model(N, m=3, p=p)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        peskin_model = PeskinModelEventDriven(N, S0, gamma, A)
        kuramoto_model = KuramotoModel(N, K, A, dt)

        sync_count_peskin, sync_times_peskin_p = peskin_model.run_trials(N_trials, Tmax, parallel=parallel, num_workers=num_workers)
        sync_count_kuramoto, sync_times_kuramoto_p = kuramoto_model.run_trials(N_trials, Tmax / dt, parallel=parallel, num_workers=num_workers)

        sync_probabilities_peskin.append(sync_count_peskin / N_trials)
        sync_times_peskin.append(sync_times_peskin_p)
        sync_probabilities_kuramoto.append(sync_count_kuramoto / N_trials)
        sync_times_kuramoto.append(sync_times_kuramoto_p)

        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} - Completed p={p:.2f} for {graph_type}')
    
    return (sync_probabilities_peskin, sync_times_peskin, 
            sync_probabilities_kuramoto, sync_times_kuramoto)

def plot_results(ax, p_values, peskin_sync_probabilities, kuramoto_sync_probabilities, graph_type, show_legend=False, show_xlabel=False, inset_label=None):
    ax.plot(p_values, peskin_sync_probabilities, marker='o', linestyle='-', color='b', label='Peskin')
    ax.plot(p_values, kuramoto_sync_probabilities, marker='o', linestyle='-', color='r', label='Kuramoto')
    if show_xlabel:
        ax.set_xlabel('edge saturation $p$')
    if show_legend:
        ax.set_ylabel('probability of sync')
    ax.set_title(f'{graph_type}')
    if show_legend:
        ax.legend(loc='upper left', fontsize=12)
    ax.grid(True)

    # Add inset label
    if inset_label:
        ax.text(0.05, 0.95, inset_label, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

if __name__ == "__main__":
    freeze_support()
    
    # Parameters
    N = 4  # Number of neurons/oscillators
    max_steps = 10000  # Number of steps (events)
    Tmax = 200
    dt = 0.1  # Time step
    N_trials = 2  # Number of trials to run
    p_values = np.linspace(0, 1, 21)  # Edge probability values
    S0 = 1.0  # Input stimulus
    gamma = 0.0  # Decay rate
    K = 1.0  # Coupling strength
    parallel = True  # Use parallel processing
    num_workers = 2  # Number of parallel workers
    data_dir = 'data_psync'
    graph_types = ['1d_chain', '1d_ring', 'er_graph', '2d_lattice', '2d_periodic_lattice', 'soft_configuration_model']

    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Set up the plot grid
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    # Run for each graph type
    for i, graph_type in enumerate(graph_types):
        peskin_sync_probabilities, peskin_sync_times, kuramoto_sync_probabilities, kuramoto_sync_times = measure_sync_probability(
            graph_type, N, S0, K, p_values, N_trials, Tmax, gamma, dt, parallel=parallel, num_workers=num_workers)

        # Prepare data for CSV
        peskin_data = []
        for idx, p in enumerate(p_values):
            for j in range(N_trials):
                peskin_data.append((p, peskin_sync_times[idx][j] != -1, peskin_sync_times[idx][j]))

        kuramoto_data = []
        for idx, p in enumerate(p_values):
            for j in range(N_trials):
                kuramoto_data.append((p, kuramoto_sync_times[idx][j] != -1, kuramoto_sync_times[idx][j]))

        df_peskin = pd.DataFrame(peskin_data, columns=['p', 'isSync', 'Tsync'])
        df_kuramoto = pd.DataFrame(kuramoto_data, columns=['p', 'isSync', 'Tsync'])

        df_peskin.to_csv(os.path.join(data_dir, f'{graph_type}_peskin_N_{N}_Ntrial_{N_trials}.csv'), index=False)
        df_kuramoto.to_csv(os.path.join(data_dir, f'{graph_type}_kuramoto_N_{N}_Ntrial_{N_trials}.csv'), index=False)

        # Determine if we need to show legend, x-label, and y-label
        show_legend = (i == 0)
        show_xlabel = (i >= 3)
        show_ylabel = (i == 0 or i == 4)

        # Plot results
        inset_label = chr(97 + i)  # 'a', 'b', 'c', etc.
        plot_results(axs[i], p_values, peskin_sync_probabilities, kuramoto_sync_probabilities, graph_type, 
                     show_legend=show_legend, show_xlabel=show_xlabel, inset_label=f'({inset_label})')
        if show_ylabel:
            axs[i].set_ylabel('probability of sync')

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'all_graphs_N_{N}_Ntrial_{N_trials}_T_{Tmax}.png'))
    plt.show()


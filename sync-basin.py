import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
import logging
import warnings
from datetime import datetime
from multiprocessing import Pool, freeze_support

warnings.filterwarnings("ignore", \
                         category=FutureWarning, \
                    message="adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

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

    def run(self, Tmax, tolerance=1e-3, velocity_threshold=1e-3, consecutive_steps=100):
        self.time = 0
        self.theta = np.random.rand(self.N) * 2 * np.pi  # Re-initialize phases randomly for each run
        history = []
        time_history = []
        velocity_history = []
        max_steps = int(Tmax / self.dt)
        settled_counter = 0
        
        for step in range(max_steps):
            previous_theta = self.theta.copy()
            history.append(self.theta.copy())
            time_history.append(self.time)
            self.step()
            self.time += self.dt
            velocity = np.abs((self.theta - previous_theta) / self.dt)
            velocity_history.append(np.mean(velocity))
            
            # Check for steady state
            if np.mean(velocity) < velocity_threshold:
                settled_counter += 1
                if settled_counter >= consecutive_steps:
                    break
            else:
                settled_counter = 0
        
        T_settle = self.time if settled_counter >= consecutive_steps else -1  # Did not settle within tolerance
        return np.array(history), np.array(time_history), np.array(velocity_history), self.time, T_settle

    def find_winding_number(self, theta):
        N = len(theta)
        # Compute phase differences
        phase_diff = np.diff(np.unwrap(theta))
        # Total phase difference (cumulative)
        total_phase_diff = np.sum(phase_diff)
        # Compute winding number
        q = total_phase_diff / (2 * np.pi)
        return round(q)

    def run_trial(self, Tmax):
        _, _, _, Tsync, T_settle = self.run(Tmax)
        q = self.find_winding_number(self.theta)
        return (q, T_settle)

    def run_trials(self, N_trials, Tmax, parallel=True, num_workers=8):
        if parallel:
            with Pool(num_workers) as pool:
                results = pool.starmap(self.run_trial, [(Tmax,) for _ in range(N_trials)])
        else:
            results = [self.run_trial(Tmax) for _ in range(N_trials)]
        return results

def generate_1d_ring(N, k):
    G = nx.cycle_graph(N)
    for node in range(N):
        for neighbor in range(1, k+1):
            G.add_edge(node, (node + neighbor) % N)
            G.add_edge(node, (node - neighbor) % N)
    A = nx.adjacency_matrix(G).toarray()
    return np.array(A)

def save_results_and_plot(N, N_trials, Tmax, results, save_dir):
    # Extract q values and settling times
    q_values, T_settle_values = zip(*results)

    # Calculate the probability distribution of q
    q_counts = Counter(q_values)
    q_probs = {q: count / N_trials for q, count in q_counts.items()}

    # Normalize the probabilities
    total_prob = sum(q_probs.values())
    q_probs_normalized = {q: prob / total_prob for q, prob in q_probs.items()}

    # Exclude T_settle = -1 when calculating mean T
    filtered_results = [(q, T) for q, T in results if T != -1]

    # Calculate mean T_settle for each q
    mu_T = {q: np.mean([T for q_val, T in filtered_results if q_val == q]) for q in q_counts.keys()}

    # Combine prob(q) and mean(T)q in a single 1x2 plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the normalized probability distribution of q
    axes[0].bar(q_probs_normalized.keys(), q_probs_normalized.values())
    axes[0].set_xlabel("Winding number q")
    axes[0].set_ylabel("Normalized Probability")

    # Plot mean(T) versus q
    axes[1].plot(mu_T.keys(), mu_T.values(), 'o')
    axes[1].set_xlabel("Winding number q")
    axes[1].set_ylabel("Mean Settling Time μ(T)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"N_{N}_Ntrial_{N_trials}_Tmax_{Tmax}.png"))
    #plt.show()

    # Plot the histograms of T_settle for each q
    unique_q = sorted(q_counts.keys())
    num_plots = len(unique_q)
    cols = 4
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for ax, q in zip(axes, unique_q):
        T_q = [T for q_val, T in filtered_results if q_val == q]
        ax.hist(T_q, bins=20, edgecolor='k', alpha=0.7, density=True)
        ax.set_title(f"q = {q}")
        ax.set_xlabel("T_settle")
        ax.set_ylabel("Frequency")

    # Remove any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"N_{N}_Ntrial_{N_trials}_Tmax_{Tmax}_histograms.png"))
    #plt.show()

    # Save the results
    results_df = pd.DataFrame(results, columns=["q", "T_settle"])
    results_df.to_csv(os.path.join(save_dir, f"N_{N}_Ntrial_{N_trials}_Tmax_{Tmax}.csv"), index=False)

    logger.info(f"Results saved to {save_dir}")

# Main execution
if __name__ == '__main__':
    freeze_support()

    # Create save directory
    save_dir = "data_sync_basin"
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over different values of N and number of trials
    N_values = [5, 20]
    trial_values = [10**1]
    parallel = True
    num_workers = 8

    k = 1
    K = 1
    dt = 0.01
    Tmax = 300.0

    for N in N_values:
        for N_trials in trial_values:
            logger.info(f"Running simulation for N = {N}, N_trials = {N_trials}")
            
            # Generate the adjacency matrix for a 1D ring graph
            A = generate_1d_ring(N, k)
            
            # Initialize the Kuramoto model
            kuramoto_model = KuramotoModel(N, K, A, dt)
            
            # Run the trials
            results = kuramoto_model.run_trials(N_trials, Tmax, parallel=parallel, num_workers=num_workers)
            
            # Save the results and plot
            save_results_and_plot(N, N_trials, Tmax, results, save_dir)


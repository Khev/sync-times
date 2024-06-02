import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, norm
import argparse

def plot_kuramoto(N, N_trials):
    # Create directory to save figures
    dir_save = 'figures_tsync'
    os.makedirs(dir_save, exist_ok=True)

    # Panel (a): Varying K
    K_values = [1, 2, 3]
    data_vary_k = []
    labels_vary_k = [f'K={K}' for K in K_values]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    best_fit_params_vary_k = {}

    for K in K_values:
        fname = f'data_tsync_kuramoto_mean_field/mean_field_N_{N}_K_{K}_Ntrial_{N_trials}.csv'
        df = pd.read_csv(fname)
        Tsync = df['Synchronization Time']
        data_vary_k.append(Tsync)

    # Panel (b): Different Omega Distributions
    gamma_values = [0.1, 0.2]
    distributions = ['uniform', 'gaussian', 'cauchy']
    data_omega_distributions = []
    labels_omega_distributions = [
        f'$\omega \sim$ Uni(0,γ={gamma})' if dist == 'uniform' else f'$\omega \sim$ Norm(0,γ={gamma})' \
              if dist == 'gaussian' else f'$\omega \sim$ Cauchy(0,γ={gamma})'
        for gamma in gamma_values for dist in distributions
    ]
    best_fit_params_omega_distributions = {}

    for gamma in gamma_values:
        for dist in distributions:
            fname = f'data_tsync_kuramoto_omega_i/mean_field_N_{N}_{dist}_gamma_{gamma}_Ntrial_{N_trials}.csv'
            df = pd.read_csv(fname)
            Tsync = df['Synchronization Time']
            data_omega_distributions.append(Tsync)

   # Panel (c): Distributed K
    mu_values = [1, 2, 3]
    data_dist_k = []
    labels_dist_k = [
        '$K_{ij} \\sim$ Uni(0, 2)',
        '$K_{ij} \\sim$ Beta(2)',
        '$K_{ij} \\sim$ Exp(3)'
    ]
    best_fit_params_dist_k = {}

    # The filenames provided:
    filenames_dist_k = {
        1: 'data_tsync_kuramoto_kij/mean_field_N_25_uniform_mu_1.00_Ntrial_10000.csv',
        2: 'data_tsync_kuramoto_kij/mean_field_N_25_beta_mu_2.00_Ntrial_10000.csv',
        3: 'data_tsync_kuramoto_kij/mean_field_N_25_exponential_mu_3.00_Ntrial_10000.csv'
    }

    for mu in mu_values:
        fname = filenames_dist_k[mu]
        df = pd.read_csv(fname)
        Tsync = df['Synchronization Time']
        data_dist_k.append(Tsync)

    # Panel (d): Different Noise Distributions
    D_values = [0.1, 1.0]
    noise_types = ['gaussian', 'uniform']
    data_noise_distributions = []
    labels_noise_distributions = [
        f'$η \sim$ Norm(0, D={D})' if noise == 'gaussian' else f'$η \sim$ Uni(0, D={D})'
        for D in D_values for noise in noise_types
    ]
    best_fit_params_noise_distributions = {}

    for D in D_values:
        for noise in noise_types:
            fname = f'data_tsync_kuramoto_noise/mean_field_N_{N}_{noise}_D_{D}_Ntrial_{N_trials}.csv'
            df = pd.read_csv(fname)
            Tsync = df['Synchronization Time']
            data_noise_distributions.append(Tsync)

    # Panel (e): Different Graph Topologies
    topologies = ['ring', 'chain', 'lattice_2d', 'ER', 'scale_free']
    labels_topologies = ['1d ring', '1d chain', '2d lattice', 'erdos-renyii', 'scale free']
    data_topologies = []
    best_fit_params_topologies = {}

    for topology in topologies:
        fname = f'data_tsync_kuramoto_graphs/{topology}_N_{N}_Ntrial_{N_trials}.csv'
        df = pd.read_csv(fname)
        Tsync = df['Synchronization Time']
        data_topologies.append(Tsync)

    # Create 3x2 grid plot
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))

    # Panel (a)
    for i, Tsync in enumerate(data_vary_k):
        axes[0, 0].hist(Tsync, bins=30, alpha=0.5, label=labels_vary_k[i], density=True, edgecolor='black', linewidth=0.5, color=colors[i])
        params = gumbel_r.fit(Tsync)
        best_fit_params_vary_k[labels_vary_k[i]] = params
        x = np.linspace(Tsync.min(), Tsync.max(), 1000)
        pdf = gumbel_r.pdf(x, *params)
        axes[0, 0].plot(x, pdf, linestyle='dotted', color=colors[i])

    axes[0, 0].set_xlabel('Synchronization Time', fontsize=14)
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    axes[0, 0].set_xlim([0, 12])
    axes[0, 0].text(0.05, 0.95, '(a)', transform=axes[0, 0].transAxes, fontsize=24, verticalalignment='top')

    # Panel (b)
    for i, Tsync in enumerate(data_omega_distributions):
        axes[0, 1].hist(Tsync, bins=30, alpha=0.5, label=labels_omega_distributions[i], density=True, edgecolor='black', linewidth=0.5, color=colors[i % len(colors)])
        params = gumbel_r.fit(Tsync)
        best_fit_params_omega_distributions[labels_omega_distributions[i]] = params
        x = np.linspace(Tsync.min(), Tsync.max(), 1000)
        pdf = gumbel_r.pdf(x, *params)
        axes[0, 1].plot(x, pdf, linestyle='dotted', color=colors[i % len(colors)])

    axes[0, 1].set_xlabel('Synchronization Time', fontsize=14)
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    axes[0, 1].set_xlim([0, 4])
    axes[0, 1].text(0.05, 0.95, '(b)', transform=axes[0, 1].transAxes, fontsize=24, verticalalignment='top')

    # Panel (c)
    for i, Tsync in enumerate(data_dist_k):
        axes[1, 0].hist(Tsync, bins=30, alpha=0.5, label=labels_dist_k[i], density=True, edgecolor='black', linewidth=0.5, color=colors[i])
        params = gumbel_r.fit(Tsync)
        best_fit_params_dist_k[labels_dist_k[i]] = params
        x = np.linspace(Tsync.min(), Tsync.max(), 1000)
        pdf = gumbel_r.pdf(x, *params)
        axes[1, 0].plot(x, pdf, linestyle='dotted', color=colors[i])

    axes[1, 0].set_xlabel('Synchronization Time', fontsize=14)
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    axes[1, 0].set_xlim([0, 12])
    axes[1, 0].text(0.05, 0.95, '(c)', transform=axes[1, 0].transAxes, fontsize=24, verticalalignment='top')

    # Panel (d)
    for i, Tsync in enumerate(data_noise_distributions):
        axes[1, 1].hist(Tsync, bins=30, alpha=0.5, label=labels_noise_distributions[i], density=True, edgecolor='black', linewidth=0.5, color=colors[i % len(colors)])
        params = gumbel_r.fit(Tsync)
        best_fit_params_noise_distributions[labels_noise_distributions[i]] = params
        x = np.linspace(Tsync.min(), Tsync.max(), 1000)
        pdf = gumbel_r.pdf(x, *params)
        axes[1, 1].plot(x, pdf, linestyle='dotted', color=colors[i % len(colors)])

    axes[1, 1].set_xlabel('Synchronization Time', fontsize=14)
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    axes[1, 1].set_xlim([0, 4])
    axes[1, 1].text(0.05, 0.95, '(d)', transform=axes[1, 1].transAxes, fontsize=24, verticalalignment='top')

    # Panel (e)
    for i, Tsync in enumerate(data_topologies):
        axes[2, 0].hist(Tsync, bins=30, alpha=0.5, label=labels_topologies[i], density=True, edgecolor='black', linewidth=0.5, color=colors[i])
        params = gumbel_r.fit(Tsync)
        best_fit_params_topologies[labels_topologies[i]] = params
        x = np.linspace(Tsync.min(), Tsync.max(), 1000)
        pdf = gumbel_r.pdf(x, *params)
        axes[2, 0].plot(x, pdf, linestyle='dotted', color=colors[i])

    axes[2, 0].set_xlabel('Synchronization Time', fontsize=14)
    axes[2, 0].legend(fontsize=12)
    axes[2, 0].grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    axes[2, 0].set_xlim([0, 40])
    axes[2, 0].text(0.05, 0.95, '(e)', transform=axes[2, 0].transAxes, fontsize=24, verticalalignment='top')

    # Panel (f): Universal Scaling Collapse
    data_standardized_all = []
    labels_all = labels_vary_k + labels_omega_distributions + labels_dist_k + labels_noise_distributions + labels_topologies

    # Standardize the data using Gumbel scale parameter
    for Tsync, params in zip(data_vary_k, best_fit_params_vary_k.values()):
        mean = params[0]
        scale = params[1]
        Tsync_standardized = (Tsync - mean) / scale
        data_standardized_all.append(Tsync_standardized.values)

    for Tsync, params in zip(data_omega_distributions, best_fit_params_omega_distributions.values()):
        mean = params[0]
        scale = params[1]
        Tsync_standardized = (Tsync - mean) / scale
        data_standardized_all.append(Tsync_standardized.values)

    for Tsync, params in zip(data_dist_k, best_fit_params_dist_k.values()):
        mean = params[0]
        scale = params[1]
        Tsync_standardized = (Tsync - mean) / scale
        data_standardized_all.append(Tsync_standardized.values)

    for Tsync, params in zip(data_noise_distributions, best_fit_params_noise_distributions.values()):
        mean = params[0]
        scale = params[1]
        Tsync_standardized = (Tsync - mean) / scale
        data_standardized_all.append(Tsync_standardized.values)

    for Tsync, params in zip(data_topologies, best_fit_params_topologies.values()):
        mean = params[0]
        scale = params[1]
        Tsync_standardized = (Tsync - mean) / scale
        data_standardized_all.append(Tsync_standardized.values)

    # Plot standardized data as open, colored circles
    axes[2, 1].set_xlabel('Standardized Synchronization Time', fontsize=14)
    axes[2, 1].set_ylabel('Density', fontsize=14)

    # Plot standard Gumbel distribution as a thick black line
    x_standard = np.linspace(-3, 3, 1000)
    standard_gumbel_pdf = gumbel_r.pdf(x_standard)
    axes[2, 1].plot(x_standard, standard_gumbel_pdf, 'k-', linewidth=2, label='Standard Gumbel')

    # Plot standardized data
    for i, Tsync_standardized in enumerate(data_standardized_all):
        hist, bin_edges = np.histogram(Tsync_standardized, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        axes[2, 1].plot(bin_centers, hist, 'o', label=labels_all[i], color=colors[i % len(colors)], markerfacecolor='none')

    axes[2, 1].legend(fontsize=12, ncol=2)
    axes[2, 1].grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    axes[2, 1].text(0.05, 0.95, '(f)', transform=axes[2, 1].transAxes, fontsize=24, verticalalignment='top')
    axes[2, 1].set_xlim([-3, 10])
    axes[2, 1].set_ylim([0, 0.4])

    # Remove y-axis labels for all plots except the leftmost column
    for ax in axes.flat:
        if ax not in axes[:, 0]:
            ax.set_ylabel('')

    # Remove borders from all plots
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(dir_save, f'kuramoto_combined_plots_N_{N}_Ntrial_{N_trials}.png'), dpi=300)

    # Show the plot
    plt.close()

    # Print the parameters of best fit
    print("Best fit parameters for vary K:")
    for label, params in best_fit_params_vary_k.items():
        print(f"{label}: loc={params[0]:.2f}, scale={params[1]:.2f}")

    print("\nBest fit parameters for omega distributions:")
    for label, params in best_fit_params_omega_distributions.items():
        print(f"{label}: loc={params[0]:.2f}, scale={params[1]:.2f}")

    print("\nBest fit parameters for dist K:")
    for label, params in best_fit_params_dist_k.items():
        print(f"{label}: loc={params[0]:.2f}, scale={params[1]:.2f}")

    print("\nBest fit parameters for noise distributions:")
    for label, params in best_fit_params_noise_distributions.items():
        print(f"{label}: loc={params[0]:.2f}, scale={params[1]:.2f}")

    print("\nBest fit parameters for different graph topologies:")
    for label, params in best_fit_params_topologies.items():
        print(f"{label}: loc={params[0]:.2f}, scale={params[1]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Kuramoto Synchronization Results')
    parser.add_argument('--N', type=int, default=25, help='Number of nodes in the network')
    parser.add_argument('--N_trials', type=int, default=10**4, help='Number of trials for synchronization')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using multiprocessing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    args = parser.parse_args()

    plot_kuramoto(args.N, args.N_trials)

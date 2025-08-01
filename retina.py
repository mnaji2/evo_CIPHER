import anndata as ad
import scanpy as sc
import numpy as np
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def load_data(file_path):
    if file_path is None:
        return None
    return ad.read_h5ad(file_path)

def filter_genes_and_cells(adata, min_mean_expr=1.0, min_cells_per_perturb=100):
    X_matrix = adata.X
    if hasattr(X_matrix, 'toarray'):
        X_matrix = X_matrix.toarray()
    mean_expression = np.array(X_matrix.mean(axis=0)).flatten()
    genes_to_keep = mean_expression > min_mean_expr
    perturbed_genes = adata.obs.loc[adata.obs['gene'] != 'control', 'gene'].unique()
    gene_names_in_adata_var = adata.var.index.astype(str)
    perturbed_gene_mask = np.isin(gene_names_in_adata_var, perturbed_genes)
    genes_to_keep = genes_to_keep | perturbed_gene_mask
    adata_filtered_genes = adata[:, genes_to_keep].copy()
    counts = adata_filtered_genes.obs['gene'].value_counts()
    perturbations_to_keep = counts[counts >= min_cells_per_perturb].index.tolist()
    if 'non-targeting' not in perturbations_to_keep:
        perturbations_to_keep.append('non-targeting')
    if 'control' not in perturbations_to_keep:
        if 'control' in adata_filtered_genes.obs['gene'].unique():
            perturbations_to_keep.append('control')
    adata_final = adata_filtered_genes[adata_filtered_genes.obs['gene'].isin(perturbations_to_keep)].copy()
    return adata_final

def match_genes(adata_unperturbed, adata_perturb):
    genes_unperturbed = adata_unperturbed.var['feature_name'].astype(str) if 'feature_name' in adata_unperturbed.var.columns else adata_unperturbed.var_names.astype(str)
    genes_perturb = adata_perturb.var_names.astype(str)
    common_genes = sorted(list(set(genes_unperturbed) & set(genes_perturb)))
    adata_unperturbed.var_names = genes_unperturbed
    adata_unperturbed.var_names_make_unique()
    adata_perturb.var_names = genes_perturb
    adata_perturb.var_names_make_unique()
    adata_unperturbed_filtered = adata_unperturbed[:, [g for g in common_genes if g in adata_unperturbed.var_names]].copy()
    adata_perturb_filtered = adata_perturb[:, [g for g in common_genes if g in adata_perturb.var_names]].copy()
    return adata_unperturbed_filtered, adata_perturb_filtered, common_genes

def normalize_and_log1p(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata

def calculate_covariance_matrix(adata):
    X_matrix = adata.X
    if hasattr(X_matrix, 'toarray'):
        X_matrix = X_matrix.toarray()
    X_centered = X_matrix - np.mean(X_matrix, axis=0, keepdims=True)
    cov_matrix = np.cov(X_centered, rowvar=False)
    return cov_matrix

def predict_delta_x(cov_matrix, gene_to_idx, perturb_gene, true_delta_x=None):
    if perturb_gene not in gene_to_idx:
        return np.zeros(cov_matrix.shape[0])
    idx = gene_to_idx[perturb_gene]
    sigma_col = cov_matrix[:, idx]
    if true_delta_x is None or np.allclose(true_delta_x, 0, atol=1e-6):
        return sigma_col
    epsilon = 1e-8
    u_opt = np.dot(sigma_col, true_delta_x) / (np.dot(sigma_col, sigma_col) + epsilon)
    return u_opt * sigma_col

def calculate_true_delta_x(adata_perturb, perturb_name):
    perturbed_cells = adata_perturb[adata_perturb.obs['gene'] == perturb_name]
    control_cells = adata_perturb[adata_perturb.obs['gene'] == 'non-targeting']
    if perturbed_cells.shape[0] == 0 or control_cells.shape[0] == 0:
        return np.zeros(adata_perturb.shape[1])
    mean_perturbed = np.array(perturbed_cells.X.mean(axis=0)).flatten()
    mean_control = np.array(control_cells.X.mean(axis=0)).flatten()
    delta = mean_perturbed - mean_control
    return delta

def plot_gene_across_species(gene, species_results):
    species_names = ["Homo sapiens", "Mus musculus", "Danio rerio"]
    plt.figure(figsize=(15,4))
    plt.suptitle(gene, fontsize=18, fontweight='bold')
    for i, species in enumerate(['human', 'mouse', 'danio'], 1):
        data = species_results.get(species)
        if data is None:
            continue
        true_delta_x, predicted_delta_x, r2 = data
        plt.subplot(1, 3, i)
        plt.scatter(true_delta_x, predicted_delta_x, alpha=0.3, s=10)
        plt.title(f"{species_names[i-1]}\nR² = {r2:.2f}")
        plt.xlabel("True ΔX")
        plt.ylabel("Predicted ΔX")
        plt.axhline(0, color='gray', linestyle='--', lw=1)
        plt.axvline(0, color='gray', linestyle='--', lw=1)
        plt.xlim(-3, 4)
        plt.ylim(-3, 4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    human_unperturbed = load_data('homo_retina.h5ad')
    human_perturb = load_data('RPE1_perturb.h5ad')
    mouse_unperturbed = load_data('mus_retina.h5ad')
    danio_unperturbed = load_data('danio_retina.h5ad')
    for adata in [human_unperturbed, human_perturb, mouse_unperturbed, danio_unperturbed]:
        normalize_and_log1p(adata)
    human_perturb = filter_genes_and_cells(human_perturb)
    human_unperturbed, human_perturb, human_genes = match_genes(human_unperturbed, human_perturb)
    human_genes_set = set(human_genes)
    mouse_genes_set = set(mouse_unperturbed.var_names.astype(str))
    danio_genes_set = set(danio_unperturbed.var_names.astype(str))
    common_genes = sorted(list(human_genes_set & mouse_genes_set & danio_genes_set))
    human_unperturbed = human_unperturbed[:, common_genes].copy()
    human_perturb = human_perturb[:, common_genes].copy()
    mouse_unperturbed = mouse_unperturbed[:, common_genes].copy()
    danio_unperturbed = danio_unperturbed[:, common_genes].copy()
    gene_to_idx_human = {g: i for i, g in enumerate(human_unperturbed.var_names)}
    gene_to_idx_mouse = {g: i for i, g in enumerate(mouse_unperturbed.var_names)}
    gene_to_idx_danio = {g: i for i, g in enumerate(danio_unperturbed.var_names)}
    cov_human = calculate_covariance_matrix(human_unperturbed)
    cov_mouse = calculate_covariance_matrix(mouse_unperturbed)
    cov_danio = calculate_covariance_matrix(danio_unperturbed)
    perturb_genes = sorted(list(set(human_perturb.obs['gene'].unique()) - {'control', 'non-targeting'}))
    os.makedirs('plots', exist_ok=True)
    for gene in perturb_genes:
        species_results = {}
        true_h = calculate_true_delta_x(human_perturb, gene)
        pred_h = predict_delta_x(cov_human, gene_to_idx_human, gene, true_h)
        if len(true_h) == len(pred_h) and np.any(true_h != 0):
            r2_h = r2_score(true_h, pred_h)
            species_results['human'] = (true_h, pred_h, r2_h)
        else:
            continue
        pred_m = predict_delta_x(cov_mouse, gene_to_idx_mouse, gene)
        species_results['mouse'] = (np.zeros_like(pred_m), pred_m, 0.0)
        pred_d = predict_delta_x(cov_danio, gene_to_idx_danio, gene)
        species_results['danio'] = (np.zeros_like(pred_d), pred_d, 0.0)
        plot_gene_across_species(gene, species_results)
        plt.savefig(f"plots/{gene}_cross_species.png")
        plt.close()

if __name__ == '__main__':
    main()

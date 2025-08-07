import os
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.sparse import issparse

def load_data(unperturbed_path, perturb_path):
    adata_unperturbed = ad.read_h5ad(unperturbed_path)
    adata_perturb = ad.read_h5ad(perturb_path)
    adata_perturb.obs['gene'] = adata_perturb.obs['gene'].astype(str).str.upper()
    return adata_unperturbed, adata_perturb

def filter_rpe_cells(adata):
    rpe_mask = adata.obs['cell_type'].astype(str).str.lower() == 'retinal pigment epithelial cell'
    return adata[rpe_mask].copy()

def filter_genes_and_cells(adata, min_mean_expr=1.0, min_cells_per_perturb=100):
    X_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    mean_expression = np.array(X_matrix.mean(axis=0)).flatten()
    genes_to_keep = mean_expression > min_mean_expr
    perturbed_genes = adata.obs.loc[adata.obs['gene'] != 'CONTROL', 'gene'].unique()
    gene_names_in_adata_var = (adata.var['feature_name'].astype(str) if 'feature_name' in adata.var.columns else adata.var_names.astype(str)).str.upper()
    perturbed_gene_mask = np.isin(gene_names_in_adata_var, perturbed_genes)
    genes_to_keep = genes_to_keep | perturbed_gene_mask
    adata_filtered_genes = adata[:, genes_to_keep].copy()
    counts = adata_filtered_genes.obs['gene'].value_counts()
    perturbations_to_keep = counts[counts >= min_cells_per_perturb].index.tolist()
    if 'NON-TARGETING' not in perturbations_to_keep:
        perturbations_to_keep.append('NON-TARGETING')
    if 'CONTROL' not in perturbations_to_keep and 'CONTROL' in adata_filtered_genes.obs['gene'].unique():
        perturbations_to_keep.append('CONTROL')
    adata_final = adata_filtered_genes[adata_filtered_genes.obs['gene'].isin(perturbations_to_keep)].copy()
    return adata_final

def get_gene_symbols(adata, species_name):
    if species_name.lower() == "mouse" and "gene_symbols" in adata.var.columns:
        return adata.var['gene_symbols'].astype(str).str.upper()
    elif species_name.lower() == "zebrafish" and "gene_short_name" in adata.var.columns:
        return adata.var['gene_short_name'].astype(str).str.upper()
    elif 'feature_name' in adata.var.columns:
        return adata.var['feature_name'].astype(str).str.upper()
    else:
        return adata.var_names.astype(str).str.upper()

def match_genes(adata_unperturbed, adata_perturb, species_name):
    genes_unperturbed = get_gene_symbols(adata_unperturbed, species_name)
    genes_perturb = adata_perturb.var_names.astype(str).str.upper()
    common_genes = sorted(list(set(genes_unperturbed) & set(genes_perturb)))
    adata_unperturbed.var_names = genes_unperturbed
    adata_unperturbed.var_names_make_unique()
    adata_perturb.var_names = genes_perturb
    adata_perturb.var_names_make_unique()
    adata_unperturbed_filtered = adata_unperturbed[:, common_genes].copy()
    adata_perturb_filtered = adata_perturb[:, common_genes].copy()
    return adata_unperturbed_filtered, adata_perturb_filtered, common_genes

def calculate_covariance_matrix(adata):
    X_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X_centered = X_matrix - np.mean(X_matrix, axis=0, keepdims=True)
    return np.cov(X_centered, rowvar=False)

def predict_delta_x(cov_matrix, gene_to_idx, perturb_gene, true_delta_x):
    if perturb_gene not in gene_to_idx:
        return np.zeros(cov_matrix.shape[0])
    idx = gene_to_idx[perturb_gene]
    sigma_col = cov_matrix[:, idx]
    epsilon = 1e-8
    u_opt = np.dot(sigma_col, true_delta_x) / (np.dot(sigma_col, sigma_col) + epsilon)
    return u_opt * sigma_col

def calculate_true_delta_x(adata_perturb, perturb_name):
    perturbed_cells = adata_perturb[adata_perturb.obs['gene'] == perturb_name]
    control_cells = adata_perturb[adata_perturb.obs['gene'] == 'NON-TARGETING']
    if perturbed_cells.shape[0] == 0 or control_cells.shape[0] == 0:
        return np.zeros(adata_perturb.shape[1])
    mean_perturbed = np.array(perturbed_cells.X.mean(axis=0)).flatten()
    mean_control = np.array(control_cells.X.mean(axis=0)).flatten()
    return mean_perturbed - mean_control

def run_cipher(unperturbed_path, perturb_path, species_name):
    adata_unperturbed, adata_perturb = load_data(unperturbed_path, perturb_path)
    adata_unperturbed = filter_rpe_cells(adata_unperturbed)
    adata_perturb = filter_genes_and_cells(adata_perturb)
    adata_unperturbed, adata_perturb, common_genes = match_genes(adata_unperturbed, adata_perturb, species_name)
    cov_matrix = calculate_covariance_matrix(adata_unperturbed)
    gene_names = adata_unperturbed.var_names
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    perturb_genes = [g for g in adata_perturb.obs['gene'].unique() if g not in ['CONTROL', 'NON-TARGETING']]
    results = {}
    for symbol in perturb_genes:
        if symbol not in common_genes:
            continue
        current_perturbed_cells = adata_perturb[adata_perturb.obs['gene'] == symbol]
        if current_perturbed_cells.shape[0] < 10:
            continue
        true_delta_x = calculate_true_delta_x(adata_perturb, symbol)
        predicted_delta_x = predict_delta_x(cov_matrix, gene_to_idx, symbol, true_delta_x)
        if len(true_delta_x) == 0 or len(predicted_delta_x) == 0 or len(true_delta_x) != len(predicted_delta_x):
            continue
        r2 = r2_score(true_delta_x, predicted_delta_x)
        results[symbol] = (r2, true_delta_x, predicted_delta_x)
    return results

def run_cipher_on_dataset(dataset_path):
    adata = ad.read_h5ad(dataset_path)
    adata.obs['perturbation'] = (
        adata.obs['gene'] if 'gene' in adata.obs.columns else
        adata.obs['perturbation_1'] if 'perturbation_1' in adata.obs.columns else
        None
    )
    if adata.obs['perturbation'] is None:
        raise ValueError('No perturbation column found.')
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
    ctrl_hits = [x for x in adata.obs['perturbation'].unique() if x.lower() in ['control', 'nt', 'non-targeting', 'ctrl']]
    adata.obs.loc[adata.obs['perturbation'].isin(ctrl_hits), 'perturbation'] = 'control'
    X0 = adata[adata.obs['perturbation'] == 'control'].X
    X0 = X0.toarray() if issparse(X0) else X0
    cov_matrix = np.cov(X0, rowvar=False)
    gene_names = np.array(adata.var_names.tolist())
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    perturb_genes = [g for g in adata.obs['perturbation'].unique() if g != 'control']
    common_genes = set(gene_names)
    results = []
    for symbol in perturb_genes:
        if symbol not in common_genes:
            continue
        current_perturbed_cells = adata[adata.obs['perturbation'] == symbol]
        if current_perturbed_cells.shape[0] < 10:
            continue
        X1 = current_perturbed_cells.X
        X1 = X1.toarray() if issparse(X1) else X1
        true_delta_x = X1.mean(axis=0) - X0.mean(axis=0)
        predicted_delta_x = predict_delta_x(cov_matrix, gene_to_idx, symbol, true_delta_x)
        if len(true_delta_x) == 0 or len(predicted_delta_x) == 0 or len(true_delta_x) != len(predicted_delta_x):
            continue
        r2 = r2_score(true_delta_x, predicted_delta_x)
        results.append((symbol, r2, true_delta_x, predicted_delta_x))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def main():
    os.makedirs("cipher_plots", exist_ok=True)
    species_paths = {
        "Human": "RPE1_perturb.h5ad",
        "Mouse": ("mus_retina.h5ad", "RPE1_perturb.h5ad"),
        "Zebrafish": ("danio_retina.h5ad", "RPE1_perturb.h5ad")
    }
    all_results = {}
    all_shared_genes = None
    human_results = run_cipher_on_dataset(species_paths["Human"])
    all_results["Human"] = {symbol: (r2, true_dx, pred_dx) for symbol, r2, true_dx, pred_dx in human_results}
    for species in ["Mouse", "Zebrafish"]:
        unperturbed_path, perturb_path = species_paths[species]
        results = run_cipher(unperturbed_path, perturb_path, species)
        all_results[species] = results
        species_genes = set(results.keys())
        all_shared_genes = species_genes if all_shared_genes is None else all_shared_genes & species_genes

    for symbol, _, _, _ in human_results:
        if symbol not in all_shared_genes:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

        vmin, vmax = None, None  
        for i, species in enumerate(["Human", "Mouse", "Zebrafish"]):
            r2, true_dx, pred_dx = all_results[species][symbol]
            ax = axes[i]

            ax.scatter(true_dx, pred_dx, alpha=0.5, s=10, color='dimgray')
            ax.plot([min(true_dx), max(true_dx)], [min(true_dx), max(true_dx)], color='black', linestyle='--', linewidth=1)

            ax.set_title(f"{species} (R² = {r2:.2f})", fontsize=10)

            if vmin is None or min(true_dx.min(), pred_dx.min()) < vmin:
                vmin = min(true_dx.min(), pred_dx.min())
            if vmax is None or max(true_dx.max(), pred_dx.max()) > vmax:
                vmax = max(true_dx.max(), pred_dx.max())

        for ax in axes:
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)

        fig.suptitle(f"{symbol}: True vs Predicted Δ Expression", fontsize=14, y=1.05)
        fig.supxlabel("True Δ Expression", fontsize=12)
        fig.supylabel("Predicted Δ Expression", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"cipher_plots/{symbol}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()

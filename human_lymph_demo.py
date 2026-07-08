from pathlib import Path
import random

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import diags
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_scipy_sparse_matrix


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(__file__).resolve().parent / "continuous_edge_graph_demo"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parent / "results_demo"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights_dir_demo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_SEED = 42

def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dense_array(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def normalize_coords(coords):
    coords = np.asarray(coords, dtype=np.float32)
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span == 0] = 1.0
    return (coords - coords.min(axis=0)) / span


def compute_local_density(coords, k=50):
    n_neighbors = min(int(k), coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    if distances.shape[1] <= 1:
        return np.ones(coords.shape[0], dtype=np.float32)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    return 1.0 / (mean_dist + 1e-6)


def sparse_distance_to_exp_similarity(distance_graph):
    sim_graph = distance_graph.tocsr().copy().astype(np.float32)
    if sim_graph.nnz == 0:
        return sim_graph
    for row_idx in range(sim_graph.shape[0]):
        start = sim_graph.indptr[row_idx]
        end = sim_graph.indptr[row_idx + 1]
        if start == end:
            continue
        row_dist = sim_graph.data[start:end]
        scale = np.mean(row_dist)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        sim_graph.data[start:end] = np.exp(-row_dist / (scale + 1e-6)).astype(np.float32)
    return sim_graph


def sparse_cosine_distance_to_similarity(distance_graph):
    sim_graph = distance_graph.copy().astype(np.float32)
    if sim_graph.nnz == 0:
        return sim_graph
    sim_graph.data = np.clip(1.0 - sim_graph.data, 0.0, 1.0).astype(np.float32)
    return sim_graph


def row_normalize_sparse(mat):
    row_sum = np.asarray(mat.sum(axis=1)).reshape(-1)
    row_sum[row_sum <= 0] = 1.0
    return diags(1.0 / row_sum).dot(mat)


def create_fused_adjacency_graph(
    x,
    coords,
    y=None,
    k_spatial=4,
    k_expression=20,
    adaptive_range=0.2,
    pca_dim=30,
    overlap_lambda=0.2,
    row_normalize=False,
):
    x = np.asarray(x, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    n_pcs = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
    if n_pcs < 1:
        raise ValueError("PCA dimension must be at least 1.")
    x_pca = PCA(n_components=n_pcs, random_state=42).fit_transform(x)

    spatial_dist = kneighbors_graph(
        coords,
        n_neighbors=min(int(k_spatial), coords.shape[0] - 1),
        mode="distance",
        metric="euclidean",
        include_self=False,
    ).tocsr()
    expression_dist = kneighbors_graph(
        x_pca,
        n_neighbors=min(int(k_expression), x_pca.shape[0] - 1),
        mode="distance",
        metric="cosine",
        include_self=False,
    ).tocsr()

    spatial_weight = sparse_distance_to_exp_similarity(spatial_dist)
    expression_weight = sparse_cosine_distance_to_similarity(expression_dist)

    base_alpha = k_spatial / (k_spatial + k_expression + 1e-6)
    local_density = compute_local_density(coords, k=50)
    dens_norm = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-6)

    half_range = adaptive_range / 3.0
    alpha_min = np.clip(base_alpha - half_range, 0.05, 0.95)
    alpha_max = np.clip(base_alpha + half_range, 0.05, 0.95)
    alpha_spatial = alpha_min + (alpha_max - alpha_min) * dens_norm

    base_adj = diags(alpha_spatial).dot(spatial_weight) + diags(1.0 - alpha_spatial).dot(expression_weight)
    spatial_binary = spatial_weight.copy()
    spatial_binary.data = np.ones_like(spatial_binary.data, dtype=np.float32)
    expression_binary = expression_weight.copy()
    expression_binary.data = np.ones_like(expression_binary.data, dtype=np.float32)
    overlap_mask = spatial_binary.multiply(expression_binary).astype(bool)

    overlap_lambda = float(overlap_lambda)
    final_adj = base_adj.copy().astype(np.float32)
    overlap_rows, overlap_cols = overlap_mask.nonzero()
    if len(overlap_rows) > 0:
        overlap_base = base_adj[overlap_rows, overlap_cols].A1
        overlap_values = overlap_base + overlap_lambda * (1.0 - overlap_base)
        final_adj[overlap_rows, overlap_cols] = overlap_values.astype(np.float32)

    overlap_boost = final_adj - base_adj
    final_adj.data = np.clip(final_adj.data, 0.0, 1.0).astype(np.float32)
    if row_normalize:
        final_adj = row_normalize_sparse(final_adj).astype(np.float32)

    edge_index, edge_attr = from_scipy_sparse_matrix(final_adj.tocsr())
    data = Data(
        x=torch.FloatTensor(x),
        edge_index=edge_index,
        edge_attr=edge_attr.float().view(-1, 1),
        pos=torch.FloatTensor(coords),
        y=torch.FloatTensor(y) if y is not None else None,
    )

    return data, {
        "final_adj": final_adj.tocsr(),
        "spatial_weight": spatial_weight.tocsr(),
        "expression_weight": expression_weight.tocsr(),
        "overlap_boost": overlap_boost.tocsr(),
        "overlap_lambda": overlap_lambda,
        "alpha_spatial": alpha_spatial,
    }


def prepare_human_lymph_data():
    rna_raw = {}
    pro_raw = {}
    for name in ["A", "B"]:
        rna = ad.read_h5ad(PATHS[name]["rna"])
        pro = ad.read_h5ad(PATHS[name]["pro"])
        if not rna.var_names.is_unique:
            rna.var_names_make_unique()
        if not pro.var_names.is_unique:
            pro.var_names_make_unique()
        rna_raw[name] = rna
        pro_raw[name] = pro

    common_genes = sorted(set(rna_raw["A"].var_names) & set(rna_raw["B"].var_names))
    adata_hvg_ref = rna_raw["A"][:, common_genes].copy()
    if "highly_variable" in adata_hvg_ref.var:
        selected_hvg = adata_hvg_ref.var_names[adata_hvg_ref.var["highly_variable"]].tolist()
    else:
        selected_hvg = common_genes[: CONFIG["A"]["n_hvg"]]
    if len(selected_hvg) == 0:
        selected_hvg = common_genes[: CONFIG["A"]["n_hvg"]]
    common_proteins = sorted(set(pro_raw["A"].var_names) & set(pro_raw["B"].var_names))

    datasets = {}
    for name in ["A", "B"]:
        rna = rna_raw[name][:, selected_hvg].copy()
        pro = pro_raw[name][:, common_proteins].copy()
        x = dense_array(rna.X).astype(np.float32)
        y = dense_array(pro.X).astype(np.float32)
        coords = normalize_coords(rna_raw[name].obsm["spatial"])
        datasets[name] = {"X": x, "Y": y, "coords": coords}
    return datasets, rna_raw, common_proteins


def save_edge_table(name, data):
    edge_index = data.edge_index.detach().cpu().numpy()
    edge_attr = data.edge_attr.detach().cpu().numpy().reshape(-1)
    edge_df = pd.DataFrame(
        {
            "source": edge_index[0],
            "target": edge_index[1],
            "edge_weight": edge_attr,
        }
    )
    edge_df.to_csv(OUTDIR / f"{name}_continuous_fused_edges.csv", index=False)


class EdgeWeightUpdater(nn.Module):
    """
    Learn an edge-specific weight from source node features, target node features,
    and the current edge attribute.
    """

    def __init__(self, node_dim, edge_dim=1, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        hs = h[src]
        hd = h[dst]

        if edge_attr is None:
            edge_attr = torch.ones((hs.size(0), 1), device=h.device, dtype=h.dtype)
        elif edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        edge_input = torch.cat([hs, hd, edge_attr], dim=-1)
        return F.softplus(self.mlp(edge_input))


class ProteinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.4):
        super().__init__()

        self.eup1 = EdgeWeightUpdater(node_dim=input_dim, edge_dim=1, hidden=32)
        self.eup2 = EdgeWeightUpdater(node_dim=hidden_dim * heads, edge_dim=1, hidden=32)
        self.eup3 = EdgeWeightUpdater(node_dim=hidden_dim * heads, edge_dim=1, hidden=32)

        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1, fill_value=1.0)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
            fill_value=1.0,
        )
        self.norm2 = nn.LayerNorm(hidden_dim * heads)

        self.gat3 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            edge_dim=1,
            fill_value=1.0,
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.skip_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # if edge_attr is not None and edge_attr.dim() == 1:
        #     edge_attr = edge_attr.view(-1, 1)

        e1 = self.eup1(x, edge_index, edge_attr)
        h = self.gat1(x, edge_index, edge_attr=e1)
        h = F.gelu(self.norm1(h))

        e2 = self.eup2(h, edge_index, e1)
        h = self.gat2(h, edge_index, edge_attr=e2)
        h = F.gelu(self.norm2(h))

        e3 = self.eup3(h, edge_index, e2)
        h_gat = self.gat3(h, edge_index, edge_attr=e3)
        h_gat = F.gelu(self.norm3(h_gat))

        skip = torch.sigmoid(self.skip_logit)
        h_combined = torch.cat([skip * x, (1.0 - skip) * h_gat], dim=1)
        return self.fusion(h_combined)


def get_metrics(true, pred):
    pearsons, spearmans, rmses = [], [], []
    for i in range(true.shape[1]):
        true_i = true[:, i]
        pred_i = pred[:, i]
        if np.std(true_i) == 0 or np.std(pred_i) == 0:
            pearsons.append(np.nan)
            spearmans.append(np.nan)
            rmses.append(np.nan)
            continue
        pearsons.append(pearsonr(true_i, pred_i)[0])
        spearmans.append(spearmanr(true_i, pred_i).correlation)
        rmses.append(np.sqrt(mean_squared_error(true_i, pred_i)))
    return pearsons, spearmans, rmses


def plot_metric_boxplot(metrics_df, out_path):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    plot_df = metrics_df.melt(var_name="Metric", value_name="Value")
    ax = sns.boxplot(x="Metric", y="Value", data=plot_df, palette="Set2", width=0.5)
    sns.stripplot(x="Metric", y="Value", data=plot_df, color=".3", size=3, alpha=0.4, jitter=True)

    for i, col in enumerate(["Pearson", "Spearman", "RMSE"]):
        median_value = metrics_df[col].median()
        ax.text(
            i,
            median_value + 0.05,
            f"median: {median_value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="red",
        )

    plt.title("Evaluation Metrics for Slice B (A -> B Transfer, continuous edge weights)", fontsize=15)
    plt.ylim(metrics_df.values.min() - 0.2, metrics_df.values.max() + 0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
CONFIG = {
    "A": {
        "k_spatial": 4,
        "k_expression": 20,
        "pca_dim": 10,
        "adaptive_range": 0.2,
        "overlap_lambda": 0.2,
        "n_hvg": 4000,
    },
    "B": {
        "k_spatial": 4,
        "k_expression": 20,
        "pca_dim": 30,
        "adaptive_range": 0.2,
        "overlap_lambda": 0.2,
        "n_hvg": 4000,
    },
}

PATHS = {
    "A": {
        "rna": PROJECT_ROOT / "data" / "human_lymph" / "GSM8195494_A1_LN.h5ad",
        "pro": PROJECT_ROOT / "data" / "human_lymph" / "GSM8195498_A1_LN_Protein.h5ad",
    },
    "B": {
        "rna": PROJECT_ROOT / "data" / "human_lymph" / "GSM8195496_D1_LN.h5ad",
        "pro": PROJECT_ROOT / "data" / "human_lymph" / "GSM8195500_D1_LN_Protein.h5ad",
    },
}

def main():
    set_seed(GLOBAL_SEED)
    datasets, rna_raw, common_proteins = prepare_human_lymph_data()
    graph_data = {}
    for name in ["A", "B"]:
        data, aux = create_fused_adjacency_graph(
            datasets[name]["X"],
            datasets[name]["coords"],
            datasets[name]["Y"],
            **{
                key: CONFIG[name][key]
                for key in [
                    "k_spatial",
                    "k_expression",
                    "pca_dim",
                    "adaptive_range",
                    "overlap_lambda",
                ]
            },
        )
        save_edge_table(name, data)
        graph_data[name] = data

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    data_a = graph_data["A"].to(device)
    data_b = graph_data["B"].to(device)

    model = ProteinPredictor(
        input_dim=data_a.x.shape[1],
        hidden_dim=512,
        output_dim=data_a.y.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.MSELoss()

    split_rng = np.random.default_rng(GLOBAL_SEED)
    indices = split_rng.permutation(data_a.x.shape[0])
    train_idx = torch.from_numpy(indices[: int(0.8 * len(indices))]).to(device)
    val_idx = torch.from_numpy(indices[int(0.8 * len(indices)) :]).to(device)

    best_val_loss = float("inf")
    patience, counter = 30, 0
    best_model_path = WEIGHTS_DIR / "best_model_A_to_B_continuous_edge.pt"

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        out = model(data_a)
        loss = criterion(out[train_idx], data_a.y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(data_a)[val_idx], data_a.y[val_idx])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
        print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        protein_pred_b = model(data_b).cpu().numpy()
        y_b_true = data_b.y.cpu().numpy()

    pearson_rs, spearman_corrs, rmses = get_metrics(y_b_true, protein_pred_b)
    metrics_df = pd.DataFrame(
        {
            "Pearson": pearson_rs,
            "Spearman": spearman_corrs,
            "RMSE": rmses,
        },
        index=common_proteins,
    )
    metrics_plot_df = metrics_df.dropna()

    print("\nFinal Evaluation on Slice B:")
    print(metrics_plot_df.median())

    plot_metric_boxplot(
        metrics_plot_df,
        RESULTS_DIR / "metrics_combined_A_to_B_continuous_edge.png",
    )

    pd.DataFrame(protein_pred_b, index=rna_raw["B"].obs_names, columns=common_proteins).to_csv(
        RESULTS_DIR / "predicted_protein_B_continuous_edge.csv"
    )
    pd.DataFrame(y_b_true, index=rna_raw["B"].obs_names, columns=common_proteins).to_csv(
        RESULTS_DIR / "true_protein_B_continuous_edge.csv"
    )
    metrics_df.to_csv(RESULTS_DIR / "evaluation_metrics_B_continuous_edge.csv")

    print("=" * 30)
    print(f"Saved edge tables and summary to: {OUTDIR}")
    print(f"Saved model and prediction outputs to: {RESULTS_DIR}")
    print(f"Best model: {best_model_path}")
    print("=" * 30)


if __name__ == "__main__":
    main()

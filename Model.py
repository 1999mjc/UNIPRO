import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EdgeWeightUpdater(nn.Module):
    """
    Learn an edge-specific weight from source node features, target node features,
    and the current edge attribute.
    """

    def __init__(self, node_dim, edge_dim=EDGE_DIM, hidden=EDGE_HIDDEN):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
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
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        heads=HEADS,
        dropout=DROPOUT,
        edge_dim=EDGE_DIM,
        edge_hidden=EDGE_HIDDEN,
        final_heads=FINAL_HEADS,
        gat_fill_value=GAT_FILL_VALUE,
        skip_logit_init=SKIP_LOGIT_INIT,
        fusion_hidden_dim=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.edge_hidden = edge_hidden
        self.final_heads = final_heads
        self.gat_fill_value = gat_fill_value
        self.skip_logit_init = skip_logit_init
        self.fusion_hidden_dim = fusion_hidden_dim if fusion_hidden_dim is not None else hidden_dim

        hidden_multi_dim = hidden_dim * heads
        hidden_final_dim = hidden_dim * final_heads

        self.eup1 = EdgeWeightUpdater(node_dim=input_dim, edge_dim=edge_dim, hidden=edge_hidden)
        self.eup2 = EdgeWeightUpdater(node_dim=hidden_multi_dim, edge_dim=edge_dim, hidden=edge_hidden)
        self.eup3 = EdgeWeightUpdater(node_dim=hidden_multi_dim, edge_dim=edge_dim, hidden=edge_hidden)

        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            fill_value=gat_fill_value,
        )
        self.norm1 = nn.LayerNorm(hidden_multi_dim)

        self.gat2 = GATConv(
            hidden_multi_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            fill_value=gat_fill_value,
        )
        self.norm2 = nn.LayerNorm(hidden_multi_dim)

        self.gat3 = GATConv(
            hidden_multi_dim,
            hidden_dim,
            heads=final_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            fill_value=gat_fill_value,
        )
        self.norm3 = nn.LayerNorm(hidden_final_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_final_dim + input_dim, self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden_dim, output_dim),
        )

        self.skip_logit = nn.Parameter(torch.tensor(float(skip_logit_init)))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

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

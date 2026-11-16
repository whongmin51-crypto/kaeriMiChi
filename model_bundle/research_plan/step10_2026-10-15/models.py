"""Step10 source-traced CGCNN/GANN shared-monolayer paired predictors.

Sparse real edges are the only edges that enter BatchNorm and aggregation.
The GANN update follows syd-c/GANN cgcnn/model.py, not a generic GAT layer.
The optional multiple attention heads and paired numeric fusion are project
adaptations frozen before scientific fitting; see 模型来源与参数说明.md.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODEL_IMPLEMENTATION_VERSION = "step10-paired-source-traced-v1"
TASKS = ("classifier", "residual", "direct_gap")


def _batch_norm(module: nn.BatchNorm1d, x: Tensor) -> Tensor:
    """Use running moments for a singleton; do not fabricate repeated atoms."""
    if module.training and x.shape[0] < 2:
        return F.batch_norm(x, module.running_mean, module.running_var,
                            module.weight, module.bias, training=False,
                            momentum=0.0, eps=module.eps)
    return module(x)


def segment_softmax(scores: Tensor, center: Tensor, n_nodes: int) -> Tensor:
    """Softmax over all actual periodic-image neighbors of each center."""
    if scores.ndim != 2:
        raise ValueError("scores must be [n_edges, n_heads]")
    index = center[:, None].expand_as(scores)
    maxima = scores.new_full((n_nodes, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, index, scores.detach(), reduce="amax", include_self=True)
    numerators = (scores - maxima[center]).exp()
    denominators = scores.new_zeros((n_nodes, scores.shape[1]))
    denominators.index_add_(0, center, numerators)
    return numerators / denominators[center].clamp_min(torch.finfo(scores.dtype).tiny)


class CGCNNConv(nn.Module):
    """The original gated residual CGCNN update on a sparse valid-edge list."""
    def __init__(self, hidden_dim: int, n_rbf: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_full = nn.Linear(2 * hidden_dim + n_rbf, 2 * hidden_dim)
        self.bn1 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, node: Tensor, edge_index: Tensor, edge_rbf: Tensor) -> Tensor:
        center, neighbor = edge_index
        # No padded slots exist in this representation, including before BN1.
        z = torch.cat((node[center], node[neighbor], edge_rbf), dim=-1)
        gate, core = _batch_norm(self.bn1, self.fc_full(z)).chunk(2, dim=-1)
        messages = torch.sigmoid(gate) * F.softplus(core)
        aggregated = node.new_zeros(node.shape)
        aggregated.index_add_(0, center, messages)
        return F.softplus(node + _batch_norm(self.bn2, aggregated))


class GANNConv(nn.Module):
    """Author GANN concat/BN/attention/first-half/BN/sigmoid update.

    For heads=1 and attention_dropout=0 this is the exact author convolution
    math on a complete neighbor list, modulo sparse summation round-off.
    H>1 adds H independent score vectors and averages their aggregations;
    the author edge projection and node output remain shared.
    """
    def __init__(self, hidden_dim: int, n_rbf: int = 32,
                 heads: int = 1, attention_dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = int(heads)
        if self.heads < 1:
            raise ValueError("attention_heads must be positive")
        self.fc_full = nn.Linear(2 * hidden_dim + n_rbf, 2 * hidden_dim)
        self.attn_fc = nn.Linear(2 * hidden_dim, self.heads, bias=False)
        self.bn1 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.heads)
        self.attention_dropout = float(attention_dropout)
        self.capture_attention = False
        self.last_attention = None
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_full.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def forward(self, node: Tensor, edge_index: Tensor, edge_rbf: Tensor) -> Tensor:
        center, neighbor = edge_index
        z = torch.cat((node[center], node[neighbor], edge_rbf), dim=-1)
        transformed = _batch_norm(self.bn1, self.fc_full(z))
        # The source uses the first h channels as values. The second half is
        # still used by attn_fc; replacing values by neighbor node or CGCNN's
        # gated message would change the published implementation.
        values = transformed[:, :self.hidden_dim]
        logits = F.leaky_relu(_batch_norm(self.bn3, self.attn_fc(transformed)),
                              negative_slope=0.01)
        attention = segment_softmax(logits, center, node.shape[0])
        if self.capture_attention:
            self.last_attention = attention.detach()
        weights = F.dropout(attention, p=self.attention_dropout, training=self.training)
        # Averaging heads before index_add is algebraically identical to
        # averaging H node aggregations and avoids an E x H x h tensor.
        messages = weights.mean(dim=1, keepdim=True) * values
        aggregated = node.new_zeros(node.shape)
        aggregated.index_add_(0, center, messages)
        return torch.sigmoid(_batch_norm(self.bn2, aggregated))


class MonolayerEncoder(nn.Module):
    def __init__(self, family: str, params: Mapping, n_rbf: int = 32):
        super().__init__()
        self.family = family.upper()
        hidden = int(params["hidden_dim"])
        self.hidden_dim = hidden
        if params.get("pooling", "mean") != "mean":
            raise ValueError("The fixed protocol uses mean pooling")
        # Linear(one_hot(Z-1,118)) matches the frozen fixed elemental vocabulary.
        self.embedding = nn.Linear(118, hidden)
        if self.family == "CGCNN":
            make_conv = lambda: CGCNNConv(hidden, n_rbf)
        elif self.family == "GANN":
            make_conv = lambda: GANNConv(hidden, n_rbf,
                heads=int(params.get("attention_heads", 1)),
                attention_dropout=float(params.get("attention_dropout", 0.0)))
        else:
            raise ValueError(f"Unknown graph family {family!r}")
        self.convs = nn.ModuleList(make_conv() for _ in range(int(params["message_layers"])))
        self.conv_to_fc = nn.Linear(hidden, hidden)

    def forward(self, graph: Mapping) -> Tensor:
        atomic_numbers = graph["atomic_numbers"]
        node = self.embedding(F.one_hot(atomic_numbers - 1, num_classes=118).to(
            dtype=self.embedding.weight.dtype))
        for conv in self.convs:
            node = conv(node, graph["edge_index"], graph["edge_rbf"])
        group = graph["node_graph_index"]
        n_graphs = int(graph["n_graphs"])
        pooled = node.new_zeros((n_graphs, self.hidden_dim))
        pooled.index_add_(0, group, node)
        count = torch.bincount(group, minlength=n_graphs).to(node.dtype)
        if torch.any(count == 0):
            raise ValueError("Empty graphs are not admitted")
        pooled = pooled / count[:, None]
        return F.softplus(self.conv_to_fc(F.softplus(pooled)))


def merge_graph_packs(lower: Mapping, upper: Mapping) -> dict:
    """One shared encoder invocation, one set of batch moments for both layers.

    The caller may deduplicate graphs independently within each layer side and
    pass sample_index[B]. Unique-graph atoms/edges determine batch statistics.
    No embedding persists across batches, optimizer steps, folds or fit scopes.
    """
    offset = lower["atomic_numbers"].shape[0]
    return {
        "atomic_numbers": torch.cat((lower["atomic_numbers"], upper["atomic_numbers"])),
        "edge_index": torch.cat((lower["edge_index"], upper["edge_index"] + offset), dim=1),
        "edge_rbf": torch.cat((lower["edge_rbf"], upper["edge_rbf"])),
        "node_graph_index": torch.cat((lower["node_graph_index"],
            upper["node_graph_index"] + int(lower["n_graphs"]))),
        "n_graphs": int(lower["n_graphs"]) + int(upper["n_graphs"]),
    }


class PairedGraphPredictor(nn.Module):
    def __init__(self, family: str, params: Mapping, n_numeric: int = 1540,
                 tasks: Sequence[str] = ("classifier",)):
        super().__init__()
        self.family = family.upper()
        self.params = dict(params)
        self.tasks = tuple(tasks)
        if not self.tasks or set(self.tasks) - set(TASKS):
            raise ValueError(f"tasks must be a nonempty subset of {TASKS}")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Duplicate task heads")
        if not params.get("shared_monolayer_encoder", True):
            raise ValueError("The protocol requires shared monolayer weights")
        self.n_numeric = int(n_numeric)
        self.encoder = MonolayerEncoder(self.family, params)
        # The physical lower and upper order is supplied by Step08; A/B labels
        # do not decide this order. Signed/absolute property differences and
        # layer-facing/stacking conditions are already in the numeric vector.
        in_dim = 2 * self.encoder.hidden_dim + self.n_numeric
        layers = []
        for width in params["fusion_hidden_dims"]:
            width = int(width)
            layers.extend((nn.Linear(in_dim, width), nn.Softplus(),
                           nn.Dropout(float(params.get("dropout", 0.0)))))
            in_dim = width
        self.fusion = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({task: nn.Linear(in_dim, 1) for task in self.tasks})

    def encode_pair(self, batch: Mapping) -> Tensor:
        lo, up = batch["lower"], batch["upper"]
        embeddings = self.encode_graph(merge_graph_packs(lo, up))
        n_lower = int(lo["n_graphs"])
        lower, upper = embeddings[:n_lower], embeddings[n_lower:]
        if "sample_index" in lo:
            lower = lower.index_select(0, lo["sample_index"])
        if "sample_index" in up:
            upper = upper.index_select(0, up["sample_index"])
        numeric = batch["numeric"]
        if numeric.ndim != 2 or numeric.shape[1] != self.n_numeric:
            raise ValueError(f"numeric must have shape [B,{self.n_numeric}]")
        if lower.shape[0] != numeric.shape[0] or upper.shape[0] != numeric.shape[0]:
            raise ValueError("sample_index must match numeric rows")
        return self.fusion(torch.cat((lower, upper, numeric), dim=1))

    def encode_graph(self, packed_graph: Mapping) -> Tensor:
        """Encode unique graphs; callers may cache only in eval/no_grad mode."""
        return self.encoder(packed_graph)

    def fuse_embeddings(self, lower: Tensor, upper: Tensor, numeric: Tensor) -> dict[str, Tensor]:
        """Evaluate fusion from same-checkpoint eval embeddings."""
        if numeric.ndim != 2 or numeric.shape[1] != self.n_numeric:
            raise ValueError(f"numeric must have shape [B,{self.n_numeric}]")
        fused = self.fusion(torch.cat((lower, upper, numeric), dim=1))
        return {name: head(fused).squeeze(-1) for name, head in self.heads.items()}

    def forward(self, batch: Mapping) -> dict[str, Tensor]:
        fused = self.encode_pair(batch)
        # Raw logits/targets: probability calibration and gap nonnegativity
        # belong to the downstream frozen metric/routing layer, not here.
        return {name: head(fused).squeeze(-1) for name, head in self.heads.items()}


def build_model(family: str, params: Mapping, n_numeric: int = 1540,
                tasks: Sequence[str] = ("classifier",)) -> PairedGraphPredictor:
    return PairedGraphPredictor(family, params, n_numeric=n_numeric, tasks=tasks)


def parameter_count(model: nn.Module) -> dict:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def pack_from_padded(atom_fea: Tensor, neighbor_rbf: Tensor, neighbor_index: Tensor,
                     neighbor_mask: Tensor, node_graph_index: Tensor,
                     n_graphs: int) -> dict:
    """Adapter to demonstrate that invalid slots never reach edge BN.

    atom_fea is the frozen 118-element one-hot, not learned continuous atoms.
    Filtering happens before indexing, so even invalid out-of-range padding
    indices cannot enter message construction or BatchNorm statistics.
    """
    center = torch.arange(atom_fea.shape[0], device=atom_fea.device)[:, None].expand_as(neighbor_index)
    mask = neighbor_mask.bool()
    return {"atomic_numbers": atom_fea.argmax(dim=-1).long() + 1,
            "edge_index": torch.stack((center[mask], neighbor_index[mask])),
            "edge_rbf": neighbor_rbf[mask], "node_graph_index": node_graph_index,
            "n_graphs": int(n_graphs)}

"""
models/causal_graph_transformer.py - Optimized Causal Graph Transformer with Time2Vec

Handles variable length sequences with ZERO feature leakage.
Labels are organized by package, but processing is optimized by position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ════════════════════════════════════════════════════════════════════════════════
# TIME2VEC MODULE
# ════════════════════════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    """
    Time2Vec: Learnable time representation.
    
    t2v(τ)[i] = ω_i × τ + φ_i,           if i = 0  (linear term)
    t2v(τ)[i] = sin(ω_i × τ + φ_i),      if i > 0  (periodic terms)
    
    Reference: "Time2Vec: Learning to Time" (Kazemi et al., 2019)
    """
    
    def __init__(self, input_dim: int = 1, embed_dim: int = 16, 
                 activation: str = 'sin'):
        """
        Args:
            input_dim: Number of input time features
            embed_dim: Output embedding dimension (must be >= 1)
            activation: 'sin' or 'cos' for periodic terms
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.activation = torch.sin if activation == 'sin' else torch.cos
        
        # Linear term parameters (first dimension)
        self.W_linear = nn.Parameter(torch.randn(input_dim, 1))
        self.b_linear = nn.Parameter(torch.randn(1))
        
        # Periodic term parameters (remaining dimensions)
        if embed_dim > 1:
            self.W_periodic = nn.Parameter(torch.randn(input_dim, embed_dim - 1))
            self.b_periodic = nn.Parameter(torch.randn(embed_dim - 1))
        else:
            self.register_parameter('W_periodic', None)
            self.register_parameter('b_periodic', None)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for stable training."""
        with torch.no_grad():
            # Linear term: small values
            nn.init.uniform_(self.W_linear, -0.1, 0.1)
            nn.init.zeros_(self.b_linear)
            
            # Periodic terms: initialize frequencies for different time scales
            if self.W_periodic is not None:
                num_periodic = self.embed_dim - 1
                # Log-spaced frequencies from hourly to yearly
                frequencies = torch.logspace(
                    math.log10(2 * math.pi / 24),      # Hourly
                    math.log10(2 * math.pi / 8760),   # Yearly
                    num_periodic
                )
                self.W_periodic.data = frequencies.unsqueeze(0).expand(self.input_dim, -1).clone()
                nn.init.uniform_(self.b_periodic, -math.pi, math.pi)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [..., input_dim]
        
        Returns:
            Time embeddings [..., embed_dim]
        """
        original_shape = t.shape[:-1]
        t = t.reshape(-1, self.input_dim)  # [batch, input_dim]
        
        # Linear term: [batch, 1]
        linear = torch.matmul(t, self.W_linear) + self.b_linear
        
        if self.W_periodic is not None:
            # Periodic terms: [batch, embed_dim - 1]
            periodic = self.activation(torch.matmul(t, self.W_periodic) + self.b_periodic)
            # Combine: [batch, embed_dim]
            output = torch.cat([linear, periodic], dim=-1)
        else:
            output = linear
        
        # Reshape back
        return output.reshape(*original_shape, self.embed_dim)


class MultiScaleTime2Vec(nn.Module):
    """
    Multi-scale Time2Vec for processing multiple time features.
    
    Input features (from preprocessor):
    - hour (0-24): Hour of day as float
    - day_of_week (0-6): Day of week
    - day_of_month (1-31): Day of month
    - month (1-12): Month of year
    - elapsed_hours: Hours since journey start (scaled)
    - time_delta: Time until plan OR time vs plan (scaled)
    """
    
    def __init__(self, embed_dim: int = 32, dropout: float = 0.1,
                 use_raw_cyclical_hybrid: bool = True):
        """
        Args:
            embed_dim: Final output embedding dimension
            dropout: Dropout rate
            use_raw_cyclical_hybrid: If True, also include raw cyclical features
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_hybrid = use_raw_cyclical_hybrid
        
        # Individual Time2Vec for each time aspect
        self.hour_t2v = Time2Vec(input_dim=1, embed_dim=8)      # Hour of day
        self.dow_t2v = Time2Vec(input_dim=1, embed_dim=4)       # Day of week
        self.dom_t2v = Time2Vec(input_dim=1, embed_dim=4)       # Day of month
        self.month_t2v = Time2Vec(input_dim=1, embed_dim=4)     # Month
        self.elapsed_t2v = Time2Vec(input_dim=1, embed_dim=8)   # Elapsed time
        self.delta_t2v = Time2Vec(input_dim=1, embed_dim=8)     # Time delta
        
        # Calculate input dimension for projection
        t2v_dim = 8 + 4 + 4 + 4 + 8 + 8  # 36
        cyclical_dim = 8 if use_raw_cyclical_hybrid else 0  # sin/cos for hour, dow, dom, month
        total_input = t2v_dim + cyclical_dim
        
        # Project to final dimension
        self.proj = nn.Sequential(
            nn.Linear(total_input, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_features: [..., 6] containing:
                - hour (0-24)
                - day_of_week (0-6)
                - day_of_month (1-31)
                - month (1-12)
                - elapsed (scaled)
                - time_delta (scaled)
        
        Returns:
            Time embedding [..., embed_dim]
        """
        # Split input features
        hour = time_features[..., 0:1]
        dow = time_features[..., 1:2]
        dom = time_features[..., 2:3]
        month = time_features[..., 3:4]
        elapsed = time_features[..., 4:5]
        delta = time_features[..., 5:6]
        
        # Apply Time2Vec to each component
        h_emb = self.hour_t2v(hour)         # [..., 8]
        d_emb = self.dow_t2v(dow)           # [..., 4]
        dm_emb = self.dom_t2v(dom)          # [..., 4]
        m_emb = self.month_t2v(month)       # [..., 4]
        e_emb = self.elapsed_t2v(elapsed)   # [..., 8]
        td_emb = self.delta_t2v(delta)      # [..., 8]
        
        embeddings = [h_emb, d_emb, dm_emb, m_emb, e_emb, td_emb]
        
        # Optionally add raw cyclical features as baseline
        if self.use_hybrid:
            hour_rad = hour * (2 * math.pi / 24)
            dow_rad = dow * (2 * math.pi / 7)
            dom_rad = dom * (2 * math.pi / 31)
            month_rad = month * (2 * math.pi / 12)
            
            cyclical = torch.cat([
                torch.sin(hour_rad),
                torch.cos(hour_rad),
                torch.sin(dow_rad),
                torch.cos(dow_rad),
                torch.sin(dom_rad),
                torch.cos(dom_rad),
                torch.sin(month_rad),
                torch.cos(month_rad),
            ], dim=-1)
            embeddings.append(cyclical)
        
        combined = torch.cat(embeddings, dim=-1)
        return self.proj(combined)


class EdgeTime2Vec(nn.Module):
    """
    Time2Vec for edge features.
    
    Input features (from preprocessor edge_features):
    - distance_scaled
    - has_distance
    - same_location
    - cross_region
    - hour (0-24)
    - dow (0-6)
    - dom (1-31)
    - month (1-12)
    """
    
    def __init__(self, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Time2Vec for time components
        self.hour_t2v = Time2Vec(input_dim=1, embed_dim=8)
        self.dow_t2v = Time2Vec(input_dim=1, embed_dim=4)
        self.dom_t2v = Time2Vec(input_dim=1, embed_dim=4)
        self.month_t2v = Time2Vec(input_dim=1, embed_dim=4)
        
        # Non-time features: distance_scaled, has_distance, same_location, cross_region = 4
        # Time2Vec: 8 + 4 + 4 + 4 = 20
        # Cyclical: 8
        total_input = 4 + 20 + 8
        
        self.proj = nn.Sequential(
            nn.Linear(total_input, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: [E, 8] - distance, has_dist, same_loc, cross_region, hour, dow, dom, month
        
        Returns:
            Edge embedding [E, embed_dim]
        """
        # Non-time features
        non_time = edge_features[..., :4]  # [E, 4]
        
        # Time features
        hour = edge_features[..., 4:5]
        dow = edge_features[..., 5:6]
        dom = edge_features[..., 6:7]
        month = edge_features[..., 7:8]
        
        # Time2Vec embeddings
        h_emb = self.hour_t2v(hour)
        d_emb = self.dow_t2v(dow)
        dm_emb = self.dom_t2v(dom)
        m_emb = self.month_t2v(month)
        
        # Cyclical features
        hour_rad = hour * (2 * math.pi / 24)
        dow_rad = dow * (2 * math.pi / 7)
        dom_rad = dom * (2 * math.pi / 31)
        month_rad = month * (2 * math.pi / 12)
        
        cyclical = torch.cat([
            torch.sin(hour_rad), torch.cos(hour_rad),
            torch.sin(dow_rad), torch.cos(dow_rad),
            torch.sin(dom_rad), torch.cos(dom_rad),
            torch.sin(month_rad), torch.cos(month_rad),
        ], dim=-1)
        
        combined = torch.cat([non_time, h_emb, d_emb, dm_emb, m_emb, cyclical], dim=-1)
        return self.proj(combined)


# ════════════════════════════════════════════════════════════════════════════════
# MULTI-EMBEDDING MODULE
# ════════════════════════════════════════════════════════════════════════════════

class MultiEmbedding(nn.Module):
    """Module for handling multiple categorical embeddings."""
    
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int, 
                 feature_names: List[str], dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.feature_names = feature_names
        self.embeddings = nn.ModuleDict()
        
        for name in feature_names:
            vocab_size = vocab_sizes.get(name, 100)
            self.embeddings[name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0
            )
        
        self.dropout = nn.Dropout(dropout)
        self._init_embeddings()
    
    def _init_embeddings(self):
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0, std=0.02)
            with torch.no_grad():
                emb.weight[0].fill_(0)
    
    def forward(self, indices: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for name in self.feature_names:
            if name in indices and name in self.embeddings:
                emb = self.embeddings[name](indices[name])
                emb = self.dropout(emb)
                embeddings.append(emb)
        
        if not embeddings:
            raise ValueError(f"No embeddings found. Expected: {self.feature_names}, Got: {list(indices.keys())}")
        
        return torch.cat(embeddings, dim=-1)
    
    def get_output_dim(self) -> int:
        return len(self.feature_names) * self.embed_dim


# ════════════════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ════════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Add positional encoding based on positions tensor."""
        positions = positions.clamp(0, self.pe.size(0) - 1)
        return x + self.pe[positions]


# ════════════════════════════════════════════════════════════════════════════════
# GRAPH TRANSFORMER LAYER
# ════════════════════════════════════════════════════════════════════════════════

class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with edge features."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 edge_dim: Optional[int] = None):
        super().__init__()
        
        self.transformer_conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True,
            concat=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Graph attention
        h = self.transformer_conv(x, edge_index, edge_attr)
        x = self.norm1(x + h)
        
        # FFN
        h = self.ffn(x)
        x = self.norm2(x + h)
        
        return x


# ════════════════════════════════════════════════════════════════════════════════
# CAUSAL GRAPH TRANSFORMER
# ════════════════════════════════════════════════════════════════════════════════

class CausalGraphTransformer(nn.Module):
    """
    Optimized Causal Graph Transformer for Transit Time Prediction with Time2Vec.
    
    ════════════════════════════════════════════════════════════════════════════
    TIME2VEC INTEGRATION
    ════════════════════════════════════════════════════════════════════════════
    
    Uses learnable Time2Vec representations for:
    - Observable time features: plan time, elapsed time, time until plan
    - Realized time features: actual event time, elapsed, time vs plan
    - Edge time features: source event time
    
    ════════════════════════════════════════════════════════════════════════════
    CAUSAL MASKING - ZERO FEATURE LEAKAGE
    ════════════════════════════════════════════════════════════════════════════
    
    For predicting edge (source → target):
    - Nodes at position ≤ source position: use OBSERVABLE + REALIZED features
    - Nodes at position > source position: use OBSERVABLE + ZEROS (realized masked)
    
    ════════════════════════════════════════════════════════════════════════════
    OPTIMIZATION - PROCESS BY POSITION
    ════════════════════════════════════════════════════════════════════════════
    
    Edges with same source position share the same causal mask.
    Speedup: E/P ≈ (batch_size × avg_edges_per_package) / max_seq_len ≈ 100-1000x
    
    ════════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        embed_dim: int = 32,
        time2vec_dim: int = 32,
        output_dim: int = 1,
        use_edge_features: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.use_edge_features = use_edge_features
        self.time2vec_dim = time2vec_dim
        
        # Store dimensions from preprocessor
        self.observable_time_dim = feature_dims.get('observable_time_dim', 6)
        self.observable_other_dim = feature_dims.get('observable_other_dim', 3)
        self.realized_time_dim = feature_dims.get('realized_time_dim', 6)
        self.realized_other_dim = feature_dims.get('realized_other_dim', 20)
        self.edge_dim = feature_dims.get('edge_dim', 8)
        self.package_dim = feature_dims.get('package_dim', 4)
        
        # ════════════════════════════════════════════════════════════════
        # TIME2VEC MODULES
        # ════════════════════════════════════════════════════════════════
        self.observable_time2vec = MultiScaleTime2Vec(
            embed_dim=time2vec_dim,
            dropout=dropout,
            use_raw_cyclical_hybrid=True
        )
        
        self.realized_time2vec = MultiScaleTime2Vec(
            embed_dim=time2vec_dim,
            dropout=dropout,
            use_raw_cyclical_hybrid=True
        )
        
        self.edge_time2vec = EdgeTime2Vec(
            embed_dim=time2vec_dim,
            dropout=dropout
        )
        
        # ════════════════════════════════════════════════════════════════
        # NODE CATEGORICAL EMBEDDINGS
        # ════════════════════════════════════════════════════════════════
        node_categorical_features = [
            'event_type', 'location', 'postal', 'region',
            'carrier', 'leg_type', 'ship_method'
        ]
        self.node_categorical_features = [
            f for f in node_categorical_features if f in vocab_sizes
        ]
        
        self.node_embedding = MultiEmbedding(
            vocab_sizes=vocab_sizes,
            embed_dim=embed_dim,
            feature_names=self.node_categorical_features,
            dropout=dropout
        )
        
        # ════════════════════════════════════════════════════════════════
        # PACKAGE POSTAL EMBEDDINGS
        # ════════════════════════════════════════════════════════════════
        self.postal_embedding = nn.Embedding(
            num_embeddings=vocab_sizes.get('postal', 1000),
            embedding_dim=embed_dim,
            padding_idx=0
        )
        nn.init.normal_(self.postal_embedding.weight, mean=0, std=0.02)
        with torch.no_grad():
            self.postal_embedding.weight[0].fill_(0)
        
        # ════════════════════════════════════════════════════════════════
        # CALCULATE INPUT DIMENSIONS
        # ════════════════════════════════════════════════════════════════
        node_cat_dim = self.node_embedding.get_output_dim()
        package_postal_dim = embed_dim * 2  # source + dest postal
        
        # Observable: Time2Vec + other continuous + categorical + package
        observable_input_dim = (
            time2vec_dim +             # Time2Vec embedding
            self.observable_other_dim + # other continuous (is_delivery, position, has_plan)
            node_cat_dim +             # categorical embeddings
            package_postal_dim +       # package postal embeddings
            self.package_dim           # package features
        )
        
        # Realized: Time2Vec + other continuous
        realized_input_dim = (
            time2vec_dim +             # Time2Vec embedding
            self.realized_other_dim    # other continuous (time_since_prev, dwelling, missort, problems)
        )
        
        print(f"\n{'='*70}")
        print(f"CAUSAL GRAPH TRANSFORMER (with Time2Vec)")
        print(f"{'='*70}")
        print(f"Time2Vec dimension: {time2vec_dim}")
        print(f"Observable input: {observable_input_dim}")
        print(f"  - Time2Vec: {time2vec_dim}")
        print(f"  - Other continuous: {self.observable_other_dim}")
        print(f"  - Categorical ({len(self.node_categorical_features)}): {node_cat_dim}")
        print(f"  - Package postal: {package_postal_dim}")
        print(f"  - Package features: {self.package_dim}")
        print(f"Realized input: {realized_input_dim}")
        print(f"  - Time2Vec: {time2vec_dim}")
        print(f"  - Other continuous: {self.realized_other_dim}")
        print(f"Edge input: {time2vec_dim} (via EdgeTime2Vec)")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Layers: {num_layers}, Heads: {num_heads}")
        print(f"{'='*70}")
        
        # ════════════════════════════════════════════════════════════════
        # INPUT PROJECTIONS
        # ════════════════════════════════════════════════════════════════
        
        # Project observable features → hidden_dim
        self.observable_proj = nn.Sequential(
            nn.Linear(observable_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Project realized features → hidden_dim
        self.realized_proj = nn.Sequential(
            nn.Linear(realized_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Combine observable + masked realized → hidden_dim
        self.combine_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ════════════════════════════════════════════════════════════════
        # POSITIONAL ENCODING
        # ════════════════════════════════════════════════════════════════
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=200)
        
        # ════════════════════════════════════════════════════════════════
        # TRANSFORMER LAYERS
        # ════════════════════════════════════════════════════════════════
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                edge_dim=time2vec_dim if use_edge_features else None
            )
            for _ in range(num_layers)
        ])
        
        # ════════════════════════════════════════════════════════════════
        # OUTPUT HEAD
        # ════════════════════════════════════════════════════════════════
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + time2vec_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._init_weights()
        
        params = self.get_num_parameters()
        print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
        print(f"{'='*70}\n")
    
    def _init_weights(self):
        """Initialize linear layer weights."""
        for module in [self.observable_proj, self.realized_proj, 
                       self.combine_proj, self.output_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def _compute_node_positions(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute position of each node within its graph.
        
        Example:
            batch = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]  (3 packages)
            returns [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]  (positions within each)
        """
        device = batch.device
        num_nodes = batch.size(0)
        positions = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        
        offset = 0
        for count in counts:
            positions[offset:offset + count] = torch.arange(count, device=device)
            offset += count
        
        return positions
    
    def _group_edges_by_source_position(
        self, 
        edge_index: torch.Tensor, 
        positions: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Group edge indices by their source node's position.
        """
        source_nodes = edge_index[0]
        source_positions = positions[source_nodes]
        
        unique_positions = torch.unique(source_positions)
        
        position_to_edges = {}
        for pos in unique_positions.tolist():
            mask = (source_positions == pos)
            position_to_edges[pos] = torch.where(mask)[0]
        
        return position_to_edges
    
    def _build_observable_features(self, data) -> torch.Tensor:
        """
        Build observable features for all nodes using Time2Vec.
        
        Observable features are ALWAYS available (known before event happens):
        - Time features: processed through Time2Vec
        - Other continuous: is_delivery, position, has_plan
        - Categorical: event type, location, carrier, etc.
        - Package: origin/destination postal, weight, etc.
        """
        batch = data.batch
        
        # 1. Time features through Time2Vec [N, time2vec_dim]
        time_emb = self.observable_time2vec(data.node_observable_time)
        
        # 2. Other continuous features [N, observable_other_dim]
        other_continuous = data.node_observable_other
        
        # 3. Categorical embeddings [N, cat_dim]
        node_cat_indices = {}
        for name in self.node_categorical_features:
            attr_name = f"{name}_idx"
            if hasattr(data, attr_name):
                node_cat_indices[name] = getattr(data, attr_name)
        
        node_cat_emb = self.node_embedding(node_cat_indices)
        
        # 4. Package postal embeddings [N, 2*embed_dim]
        source_postal_emb = self.postal_embedding(data.source_postal_idx)
        dest_postal_emb = self.postal_embedding(data.dest_postal_idx)
        package_postal_emb = torch.cat([
            source_postal_emb[batch],
            dest_postal_emb[batch]
        ], dim=-1)
        
        # 5. Package features [N, package_dim]
        package_features = data.package_features[batch]
        
        # Combine all observable features
        observable_combined = torch.cat([
            time_emb,             # [N, time2vec_dim]
            other_continuous,     # [N, obs_other_dim]
            node_cat_emb,         # [N, cat_dim]
            package_postal_emb,   # [N, 2*embed]
            package_features      # [N, pkg_dim]
        ], dim=-1)
        
        return self.observable_proj(observable_combined)
    
    def _build_realized_features(self, data) -> torch.Tensor:
        """
        Build realized features for all nodes using Time2Vec.
        
        Realized features (known AFTER event happens):
        - Time features: actual event time processed through Time2Vec
        - Other continuous: time_since_prev, dwelling, missort, problems
        """
        # 1. Time features through Time2Vec [N, time2vec_dim]
        time_emb = self.realized_time2vec(data.node_realized_time)
        
        # 2. Other continuous features [N, realized_other_dim]
        other_continuous = data.node_realized_other
        
        # Combine realized features
        realized_combined = torch.cat([
            time_emb,
            other_continuous
        ], dim=-1)
        
        return self.realized_proj(realized_combined)
    
    def _apply_causal_mask(
        self, 
        observable_hidden: torch.Tensor,
        realized_hidden: torch.Tensor, 
        positions: torch.Tensor,
        current_position: int
    ) -> torch.Tensor:
        """
        Apply causal mask and combine features.
        
        For position P:
        - Nodes at position ≤ P: KEEP realized (events have happened)
        - Nodes at position > P: ZERO realized (future events)
        """
        causal_mask = (positions <= current_position)
        mask_expanded = causal_mask.unsqueeze(-1).float()
        realized_masked = realized_hidden * mask_expanded
        combined = torch.cat([observable_hidden, realized_masked], dim=-1)
        
        return self.combine_proj(combined)
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with causal masking and Time2Vec.
        
        Args:
            data: PyG Batch with:
                - node_observable_time: [N, 6] - time features for Time2Vec
                - node_observable_other: [N, 3] - other observable features
                - node_realized_time: [N, 6] - realized time features
                - node_realized_other: [N, realized_other_dim] - other realized features
                - edge_features: [E, 8]
                - edge_index: [2, E]
                - batch: [N]
                - edge_labels: [E] - targets
        
        Returns:
            predictions: [E, output_dim] - aligned with edge_labels by index
        """
        device = data.node_observable_time.device
        batch = data.batch
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Handle empty batch
        if num_edges == 0:
            return torch.zeros((0, self.output_dim), device=device)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: PRECOMPUTE (once for entire batch)
        # ════════════════════════════════════════════════════════════════
        
        # Observable features with Time2Vec
        observable_hidden = self._build_observable_features(data)
        
        # Realized features with Time2Vec
        realized_hidden = self._build_realized_features(data)
        
        # Edge features with Time2Vec
        edge_hidden = self.edge_time2vec(data.edge_features)
        
        # Compute node positions
        positions = self._compute_node_positions(batch)
        
        # Group edges by source position
        position_to_edges = self._group_edges_by_source_position(
            edge_index, positions
        )
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: ALLOCATE OUTPUT
        # ════════════════════════════════════════════════════════════════
        
        predictions = torch.zeros(num_edges, self.output_dim, device=device)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: PROCESS EACH UNIQUE POSITION
        # ════════════════════════════════════════════════════════════════
        
        for pos in sorted(position_to_edges.keys()):
            edge_indices = position_to_edges[pos]
            
            # 3a. Apply causal mask
            node_hidden = self._apply_causal_mask(
                observable_hidden, 
                realized_hidden, 
                positions, 
                current_position=pos
            )
            
            # 3b. Add positional encoding
            node_hidden = self.pos_encoding(node_hidden, positions)
            
            # 3c. Run transformer layers
            for layer in self.layers:
                node_hidden = layer(
                    node_hidden, 
                    edge_index, 
                    edge_hidden if self.use_edge_features else None
                )
            
            # 3d. Batch predict for ALL edges at this position
            source_indices = edge_index[0, edge_indices]
            source_hidden = node_hidden[source_indices]
            edges_hidden = edge_hidden[edge_indices]
            
            combined = torch.cat([source_hidden, edges_hidden], dim=-1)
            preds = self.output_head(combined)
            
            # 3e. Store at original indices
            predictions[edge_indices] = preds
        
        return predictions
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
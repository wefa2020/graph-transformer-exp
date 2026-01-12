"""
models/causal_graph_transformer.py - Optimized Causal Graph Transformer

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


class CausalGraphTransformer(nn.Module):
    """
    Optimized Causal Graph Transformer for Transit Time Prediction.
    
    ════════════════════════════════════════════════════════════════════════════
    CAUSAL MASKING - ZERO FEATURE LEAKAGE
    ════════════════════════════════════════════════════════════════════════════
    
    For predicting edge (source → target):
    - Nodes at position ≤ source position: use OBSERVABLE + REALIZED features
    - Nodes at position > source position: use OBSERVABLE + ZEROS (realized masked)
    
    This ensures no information from future events leaks into predictions.
    
    ════════════════════════════════════════════════════════════════════════════
    OPTIMIZATION - PROCESS BY POSITION
    ════════════════════════════════════════════════════════════════════════════
    
    Instead of running transformer once per edge (O(E) passes),
    we run once per unique source position (O(P) passes where P << E).
    
    Edges with same source position share the same causal mask!
    
    Speedup: E/P ≈ (batch_size × avg_edges_per_package) / max_seq_len ≈ 100-1000x
    
    ════════════════════════════════════════════════════════════════════════════
    VARIABLE LENGTH HANDLING
    ════════════════════════════════════════════════════════════════════════════
    
    Each package can have different number of events. Handled automatically:
    - Position tensor tracks each node's position within its package
    - Edges grouped by source position (not by package)
    - Labels stored by package, predictions stored at original edge indices
    
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
        output_dim: int = 1,
        use_edge_features: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.use_edge_features = use_edge_features
        
        # Store dimensions
        self.observable_dim = feature_dims.get('observable_dim', 11)
        self.realized_dim = feature_dims.get('realized_dim', 20)
        self.edge_dim = feature_dims.get('edge_dim', 8)
        self.package_dim = feature_dims.get('package_dim', 4)
        
        # ════════════════════════════════════════════════════════════════
        # NODE CATEGORICAL EMBEDDINGS
        # ════════════════════════════════════════════════════════════════
        node_categorical_features = [
            'event_type', 'location', 'postal', 'region',
            'carrier', 'leg_type', 'ship_method'
        ]
        # Filter to only features that exist in vocab_sizes
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
        
        observable_input_dim = (
            self.observable_dim +      # continuous observable features
            node_cat_dim +             # categorical embeddings
            package_postal_dim +       # package postal embeddings
            self.package_dim           # package features
        )
        
        print(f"\n{'='*70}")
        print(f"CAUSAL GRAPH TRANSFORMER (Optimized)")
        print(f"{'='*70}")
        print(f"Observable input: {observable_input_dim}")
        print(f"  - Continuous: {self.observable_dim}")
        print(f"  - Categorical ({len(self.node_categorical_features)}): {node_cat_dim}")
        print(f"  - Package postal: {package_postal_dim}")
        print(f"  - Package features: {self.package_dim}")
        print(f"Realized input: {self.realized_dim}")
        print(f"Edge input: {self.edge_dim}")
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
            nn.Linear(self.realized_dim, hidden_dim),
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
        
        # Project edge features → hidden_dim
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_dim, hidden_dim),
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
                edge_dim=hidden_dim if use_edge_features else None
            )
            for _ in range(num_layers)
        ])
        
        # ════════════════════════════════════════════════════════════════
        # OUTPUT HEAD
        # ════════════════════════════════════════════════════════════════
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
                       self.combine_proj, self.edge_proj, self.output_head]:
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
        
        # Get counts per graph using unique_consecutive (assumes sorted batch)
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
        
        This is the KEY OPTIMIZATION: edges with same source position
        share the same causal mask.
        
        Example:
            Edge sources at positions: [0, 1, 0, 1, 2, 3, 0]
            Returns: {
                0: tensor([0, 2, 6]),  # edges with source at position 0
                1: tensor([1, 3]),     # edges with source at position 1
                2: tensor([4]),
                3: tensor([5])
            }
        """
        source_nodes = edge_index[0]
        source_positions = positions[source_nodes]  # [E]
        
        unique_positions = torch.unique(source_positions)
        
        position_to_edges = {}
        for pos in unique_positions.tolist():
            mask = (source_positions == pos)
            position_to_edges[pos] = torch.where(mask)[0]
        
        return position_to_edges
    
    def _build_observable_features(self, data) -> torch.Tensor:
        """
        Build observable features for all nodes.
        
        Observable features are ALWAYS available (known before event happens):
        - Continuous: time encodings, position, coordinates
        - Categorical: event type, location, carrier, etc.
        - Package: origin/destination, weight, etc.
        """
        batch = data.batch
        
        # 1. Continuous observable features [N, observable_dim]
        node_observable = data.node_observable
        
        # 2. Categorical embeddings [N, cat_dim]
        node_cat_indices = {}
        for name in self.node_categorical_features:
            attr_name = f"{name}_idx"
            if hasattr(data, attr_name):
                node_cat_indices[name] = getattr(data, attr_name)
        
        node_cat_emb = self.node_embedding(node_cat_indices)
        
        # 3. Package postal embeddings [N, 2*embed_dim]
        # These are package-level, expanded to all nodes
        source_postal_emb = self.postal_embedding(data.source_postal_idx)  # [G, embed]
        dest_postal_emb = self.postal_embedding(data.dest_postal_idx)      # [G, embed]
        package_postal_emb = torch.cat([
            source_postal_emb[batch],  # [N, embed]
            dest_postal_emb[batch]     # [N, embed]
        ], dim=-1)
        
        # 4. Package features [N, package_dim]
        package_features = data.package_features[batch]
        
        # Combine all observable features
        observable_combined = torch.cat([
            node_observable,      # [N, obs_dim]
            node_cat_emb,         # [N, cat_dim]
            package_postal_emb,   # [N, 2*embed]
            package_features      # [N, pkg_dim]
        ], dim=-1)
        
        return self.observable_proj(observable_combined)  # [N, hidden_dim]
    
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
        
        Args:
            observable_hidden: [N, H] - always available
            realized_hidden: [N, H] - to be masked
            positions: [N] - position of each node
            current_position: P - source position we're predicting from
        """
        # Mask: True = keep realized, False = zero realized
        causal_mask = (positions <= current_position)  # [N]
        
        # Expand mask for element-wise multiplication
        mask_expanded = causal_mask.unsqueeze(-1).float()  # [N, 1]
        
        # Zero out future nodes' realized features
        realized_masked = realized_hidden * mask_expanded  # [N, H]
        
        # Combine observable (always available) with masked realized
        combined = torch.cat([observable_hidden, realized_masked], dim=-1)  # [N, 2H]
        
        return self.combine_proj(combined)  # [N, H]
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with causal masking - optimized by position.
        
        ════════════════════════════════════════════════════════════════════
        ALGORITHM
        ════════════════════════════════════════════════════════════════════
        
        1. PRECOMPUTE (once for batch):
           - Observable features for all nodes
           - Realized features for all nodes (before masking)
           - Edge features
           - Node positions
           - Group edges by source position
        
        2. FOR EACH UNIQUE POSITION P (not each edge!):
           a. Build causal mask: nodes ≤ P keep realized, others get zeros
           b. Combine observable + masked realized
           c. Run transformer layers
           d. Predict for ALL edges with source at position P
           e. Store predictions at original edge indices
        
        3. RETURN predictions aligned with data.edge_labels
        
        ════════════════════════════════════════════════════════════════════
        
        Args:
            data: PyG Batch with:
                - node_observable: [N, obs_dim]
                - node_realized: [N, real_dim]
                - edge_features: [E, edge_dim]
                - edge_index: [2, E]
                - batch: [N]
                - edge_labels: [E] - targets (for loss computation externally)
        
        Returns:
            predictions: [E, output_dim] - aligned with edge_labels by index
        """
        device = data.node_observable.device
        batch = data.batch
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Handle empty batch
        if num_edges == 0:
            return torch.zeros((0, self.output_dim), device=device)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: PRECOMPUTE (once for entire batch)
        # ════════════════════════════════════════════════════════════════
        
        # Observable features - always available for all nodes
        observable_hidden = self._build_observable_features(data)  # [N, H]
        
        # Realized features - will be masked based on causal position
        realized_hidden = self.realized_proj(data.node_realized)   # [N, H]
        
        # Edge features
        edge_hidden = self.edge_proj(data.edge_features)           # [E, H]
        
        # Compute node positions within each graph
        positions = self._compute_node_positions(batch)             # [N]
        
        # Group edges by source position (KEY OPTIMIZATION)
        position_to_edges = self._group_edges_by_source_position(
            edge_index, positions
        )
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: ALLOCATE OUTPUT (same order as edge_labels)
        # ════════════════════════════════════════════════════════════════
        
        predictions = torch.zeros(num_edges, self.output_dim, device=device)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: PROCESS EACH UNIQUE POSITION
        # ════════════════════════════════════════════════════════════════
        
        for pos in sorted(position_to_edges.keys()):
            edge_indices = position_to_edges[pos]  # Tensor of edge indices
            
            # ────────────────────────────────────────────────────────────
            # 3a. Apply causal mask for this position
            # ────────────────────────────────────────────────────────────
            node_hidden = self._apply_causal_mask(
                observable_hidden, 
                realized_hidden, 
                positions, 
                current_position=pos
            )  # [N, H]
            
            # ────────────────────────────────────────────────────────────
            # 3b. Add positional encoding
            # ────────────────────────────────────────────────────────────
            node_hidden = self.pos_encoding(node_hidden, positions)  # [N, H]
            
            # ────────────────────────────────────────────────────────────
            # 3c. Run transformer layers
            # ────────────────────────────────────────────────────────────
            for layer in self.layers:
                node_hidden = layer(
                    node_hidden, 
                    edge_index, 
                    edge_hidden if self.use_edge_features else None
                )
            
            # ────────────────────────────────────────────────────────────
            # 3d. Batch predict for ALL edges at this position
            # ────────────────────────────────────────────────────────────
            source_indices = edge_index[0, edge_indices]  # [num_edges_at_pos]
            
            source_hidden = node_hidden[source_indices]   # [num_edges_at_pos, H]
            edges_hidden = edge_hidden[edge_indices]      # [num_edges_at_pos, H]
            
            combined = torch.cat([source_hidden, edges_hidden], dim=-1)  # [*, 2H]
            preds = self.output_head(combined)                            # [*, output_dim]
            
            # ────────────────────────────────────────────────────────────
            # 3e. Store at ORIGINAL indices (aligns with edge_labels)
            # ────────────────────────────────────────────────────────────
            predictions[edge_indices] = preds
        
        return predictions  # [E, output_dim] - aligned with data.edge_labels
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

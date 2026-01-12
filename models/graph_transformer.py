"""
models/causal_graph_transformer.py - Causal Graph Transformer with ZERO Feature Leakage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import math
from typing import Dict, List, Optional
import numpy as np


class MultiEmbedding(nn.Module):
    """Module for handling multiple categorical embeddings"""
    
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int, 
                 feature_names: List[str], dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.feature_names = feature_names
        
        self.embeddings = nn.ModuleDict()
        
        for name in feature_names:
            base_name = self._get_base_vocab_name(name, vocab_sizes)
            vocab_size = vocab_sizes.get(base_name, 100)
            
            self.embeddings[name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0
            )
        
        self.dropout = nn.Dropout(dropout)
        self._init_embeddings()
    
    def _get_base_vocab_name(self, name: str, vocab_sizes: Dict[str, int]) -> str:
        if name in vocab_sizes:
            return name
        
        mappings = {
            'event_type': 'event_type',
            'location': 'location',
            'region': 'region',
            'carrier': 'carrier',
            'leg_type': 'leg_type',
            'ship_method': 'ship_method',
            'postal': 'postal',
        }
        
        return mappings.get(name, name)
    
    def _init_embeddings(self):
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0, std=0.02)
            with torch.no_grad():
                emb.weight[0].fill_(0)
    
    def forward(self, indices: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for name in self.feature_names:
            if name in indices:
                emb = self.embeddings[name](indices[name])
                emb = self.dropout(emb)
                embeddings.append(emb)
        return torch.cat(embeddings, dim=-1)
    
    def get_output_dim(self) -> int:
        return len(self.feature_names) * self.embed_dim


class PositionalEncoding(nn.Module):
    """Positional encoding for sequential data"""
    
    def __init__(self, d_model: int, max_len: int = 100):
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
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        _, counts = torch.unique(batch, return_counts=True)
        positions = torch.cat([torch.arange(c, device=x.device) for c in counts])
        return x + self.pe[positions]


class GraphTransformerLayer(nn.Module):
    """Standard Graph Transformer layer - attention follows edge_index"""
    
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
        h = self.transformer_conv(x, edge_index, edge_attr)
        x = self.norm1(x + h)
        h = self.ffn(x)
        x = self.norm2(x + h)
        return x


class GraphTransformerWithEmbeddings(nn.Module):
    """
    Causal Graph Transformer with ZERO Feature Leakage.
    
    For each edge prediction (source → target):
    - Nodes 0 to source: use observable + realized features
    - Nodes source+1 to N-1: use observable + ZEROS (realized zeroed out)
    
    This guarantees no information from future events leaks into predictions.
    """
    
    def __init__(self, config, vocab_sizes: Dict[str, int], feature_dims: Dict = None):
        super().__init__()
        
        self.config = config
        self.vocab_sizes = vocab_sizes
        
        # Get feature dimensions
        if feature_dims is None:
            observable_dim = getattr(config, 'observable_dim', 11)
            realized_dim = getattr(config, 'realized_dim', 20)
            edge_dim = getattr(config, 'edge_dim', 8)
            package_dim = getattr(config, 'package_dim', 4)
        else:
            observable_dim = feature_dims['observable_dim']
            realized_dim = feature_dims['realized_dim']
            edge_dim = feature_dims['edge_dim']
            package_dim = feature_dims['package_dim']
        
        self.observable_dim = observable_dim
        self.realized_dim = realized_dim
        
        # === Node Categorical Embeddings ===
        node_categorical_features = [
            'event_type', 'location', 'postal', 'region',
            'carrier', 'leg_type', 'ship_method'
        ]
        self.node_embedding = MultiEmbedding(
            vocab_sizes=vocab_sizes,
            embed_dim=config.embed_dim,
            feature_names=node_categorical_features,
            dropout=config.dropout
        )
        
        # === Package Postal Embeddings ===
        self.postal_embedding = nn.Embedding(
            num_embeddings=vocab_sizes.get('postal', 1000),
            embedding_dim=config.embed_dim,
            padding_idx=0
        )
        nn.init.normal_(self.postal_embedding.weight, mean=0, std=0.02)
        with torch.no_grad():
            self.postal_embedding.weight[0].fill_(0)
        
        # === Calculate Input Dimensions ===
        node_cat_dim = self.node_embedding.get_output_dim()
        package_postal_dim = config.embed_dim * 2
        
        observable_input_dim = observable_dim + node_cat_dim + package_postal_dim + package_dim
        realized_input_dim = realized_dim
        
        print(f"\n=== Zero-Leakage Causal Model ===")
        print(f"Observable input dim: {observable_input_dim}")
        print(f"Realized input dim: {realized_input_dim}")
        print(f"Edge input dim: {edge_dim}")
        print(f"Hidden dim: {config.hidden_dim}")
        
        # === Input Projections ===
        self.observable_proj = nn.Sequential(
            nn.Linear(observable_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.realized_proj = nn.Sequential(
            nn.Linear(realized_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.combine_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # === Positional Encoding ===
        self.pos_encoding = PositionalEncoding(config.hidden_dim, max_len=100)
        
        # === Transformer Layers ===
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                edge_dim=config.hidden_dim if config.use_edge_features else None
            )
            for _ in range(config.num_layers)
        ])
        
        # === Output Head ===
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        self._init_weights()
        
        params = self.get_num_parameters()
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
    
    def _init_weights(self):
        for module in [self.observable_proj, self.realized_proj, 
                       self.combine_proj, self.edge_proj, self.output_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def _get_node_positions(self, batch: torch.Tensor) -> torch.Tensor:
        """Get position of each node within its graph."""
        device = batch.device
        num_nodes = batch.size(0)
        positions = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        _, counts = torch.unique(batch, return_counts=True)
        offset = 0
        for count in counts:
            positions[offset:offset + count] = torch.arange(count, device=device)
            offset += count
        
        return positions
    
    def _build_observable_features(self, data) -> torch.Tensor:
        """
        Build observable features for all nodes.
        Observable features are always available (known before event happens).
        
        Returns:
            observable_hidden: [num_nodes, hidden_dim]
        """
        batch = data.batch
        
        # Continuous observable features
        node_observable = data.node_observable
        
        # Categorical embeddings
        node_cat_indices = {
            'event_type': data.event_type_idx,
            'location': data.location_idx,
            'postal': data.postal_idx,
            'region': data.region_idx,
            'carrier': data.carrier_idx,
            'leg_type': data.leg_type_idx,
            'ship_method': data.ship_method_idx,
        }
        node_cat_emb = self.node_embedding(node_cat_indices)
        
        # Package postal embeddings (expanded to nodes)
        source_postal_emb = self.postal_embedding(data.source_postal_idx)
        dest_postal_emb = self.postal_embedding(data.dest_postal_idx)
        package_postal_emb = torch.cat([
            source_postal_emb[batch],
            dest_postal_emb[batch]
        ], dim=-1)
        
        # Package features (expanded to nodes)
        package_features = data.package_features[batch]
        
        # Combine all observable
        observable_combined = torch.cat([
            node_observable,
            node_cat_emb,
            package_postal_emb,
            package_features
        ], dim=-1)
        
        return self.observable_proj(observable_combined)
    
    def _build_realized_features(self, data) -> torch.Tensor:
        """
        Build realized features for all nodes (before masking).
        
        Returns:
            realized_hidden: [num_nodes, hidden_dim]
        """
        return self.realized_proj(data.node_realized)
    
    def _build_causal_mask(self, source_node_idx: int, batch: torch.Tensor, 
                           positions: torch.Tensor) -> torch.Tensor:
        """
        Build causal mask for a specific edge.
        
        For edge with source node at position P in graph G:
        - Nodes in graph G with position <= P: mask = True (use realized)
        - Nodes in graph G with position > P: mask = False (zero realized)
        - Nodes in other graphs: mask = True (doesn't affect this prediction)
        
        Args:
            source_node_idx: Global index of source node
            batch: [num_nodes] graph assignment
            positions: [num_nodes] position within each graph
        
        Returns:
            causal_mask: [num_nodes] boolean
        """
        device = batch.device
        num_nodes = batch.size(0)
        
        source_graph = batch[source_node_idx]
        source_position = positions[source_node_idx]
        
        # Mask for nodes in the same graph
        same_graph = (batch == source_graph)
        
        # Mask for nodes at or before source position
        position_ok = (positions <= source_position)
        
        # Final mask: 
        # - Same graph AND position <= source: True (happened)
        # - Different graph: True (doesn't matter, won't affect this prediction)
        # - Same graph AND position > source: False (future, zero out)
        causal_mask = ~same_graph | (same_graph & position_ok)
        
        return causal_mask
    
    def _apply_causal_mask(self, observable_hidden: torch.Tensor, 
                           realized_hidden: torch.Tensor,
                           causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Combine observable and masked realized features.
        
        Args:
            observable_hidden: [num_nodes, hidden_dim]
            realized_hidden: [num_nodes, hidden_dim]
            causal_mask: [num_nodes] boolean
        
        Returns:
            node_hidden: [num_nodes, hidden_dim]
        """
        # Zero out realized features for future nodes
        mask_expanded = causal_mask.unsqueeze(-1).float()  # [N, 1]
        realized_masked = realized_hidden * mask_expanded   # [N, H]
        
        # Combine
        combined = torch.cat([observable_hidden, realized_masked], dim=-1)
        return self.combine_proj(combined)
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with ZERO feature leakage.
        
        For each edge, we:
        1. Build causal mask based on source node position
        2. Zero out realized features for future nodes
        3. Run transformer
        4. Predict from source node representation
        
        Returns:
            predictions: [num_edges, output_dim]
        """
        device = data.node_observable.device
        batch = data.batch
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        
        if num_edges == 0:
            return torch.zeros((0, self.config.output_dim), device=device)
        
        # Precompute features that don't depend on causal mask
        observable_hidden = self._build_observable_features(data)  # [N, H]
        realized_hidden = self._build_realized_features(data)       # [N, H]
        edge_hidden = self.edge_proj(data.edge_features)            # [E, H]
        positions = self._get_node_positions(batch)                  # [N]
        
        # Process each edge with its own causal mask
        all_predictions = []
        source_nodes = edge_index[0]
        
        for e_idx in range(num_edges):
            source_idx = source_nodes[e_idx].item()
            
            # Build causal mask for this edge
            causal_mask = self._build_causal_mask(source_idx, batch, positions)
            
            # Apply mask and combine features
            node_hidden = self._apply_causal_mask(
                observable_hidden, realized_hidden, causal_mask
            )
            
            # Add positional encoding
            node_hidden = self.pos_encoding(node_hidden, batch)
            
            # Run transformer layers
            for layer in self.layers:
                node_hidden = layer(node_hidden, edge_index, edge_hidden)
            
            # Get source node representation
            source_hidden = node_hidden[source_idx:source_idx + 1]  # [1, H]
            this_edge_hidden = edge_hidden[e_idx:e_idx + 1]         # [1, H]
            
            # Predict
            combined = torch.cat([source_hidden, this_edge_hidden], dim=-1)
            pred = self.output_head(combined)
            all_predictions.append(pred)
        
        return torch.cat(all_predictions, dim=0)
    
    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
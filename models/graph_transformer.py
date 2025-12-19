import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import math
from typing import Dict, List, Optional
from config import ModelConfig


class MultiEmbedding(nn.Module):
    """Module for handling multiple categorical embeddings"""
    
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int, 
                 feature_names: List[str], dropout: float = 0.1):
        """
        Args:
            vocab_sizes: Dict mapping category name to vocabulary size
            embed_dim: Embedding dimension for each category
            feature_names: List of feature names to embed
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.feature_names = feature_names
        
        # Create embedding layer for each feature
        self.embeddings = nn.ModuleDict()
        
        for name in feature_names:
            # Find the base vocab name
            base_name = self._get_base_vocab_name(name, vocab_sizes)
            vocab_size = vocab_sizes.get(base_name, 100)
            
            self.embeddings[name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0  # PAD token
            )
        
        self.dropout = nn.Dropout(dropout)
        self._init_embeddings()
    
    def _get_base_vocab_name(self, name: str, vocab_sizes: Dict[str, int]) -> str:
        """Map feature name to base vocabulary name"""
        # Direct match
        if name in vocab_sizes:
            return name
        
        # Remove prefixes
        prefixes = ['next_', 'edge_', 'from_', 'to_']
        clean_name = name
        for prefix in prefixes:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        
        # Try cleaned name
        if clean_name in vocab_sizes:
            return clean_name
        
        # Handle specific mappings
        mappings = {
            'event_type': 'event_type',
            'location': 'location',
            'region': 'region',
            'carrier': 'carrier',
            'leg_type': 'leg_type',
            'ship_method': 'ship_method',
            'postal': 'postal',
            'carrier_from': 'carrier',
            'carrier_to': 'carrier',
            'ship_method_from': 'ship_method',
            'ship_method_to': 'ship_method',
        }
        
        return mappings.get(clean_name, clean_name)
    
    def _init_embeddings(self):
        """Initialize embedding weights"""
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0, std=0.02)
            with torch.no_grad():
                emb.weight[0].fill_(0)  # Zero out PAD embedding
    
    def forward(self, indices: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Lookup embeddings and concatenate
        
        Args:
            indices: Dict mapping feature name to indices tensor
        
        Returns:
            Concatenated embeddings [batch_size, num_features * embed_dim]
        """
        embeddings = []
        
        for name in self.feature_names:
            if name in indices:
                emb = self.embeddings[name](indices[name])
                emb = self.dropout(emb)
                embeddings.append(emb)
        
        return torch.cat(embeddings, dim=-1)
    
    def get_output_dim(self) -> int:
        """Get total output dimension"""
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
        """Add positional encoding to node features"""
        _, counts = torch.unique(batch, return_counts=True)
        positions = torch.cat([torch.arange(c, device=x.device) for c in counts])
        return x + self.pe[positions]


class GraphTransformerLayer(nn.Module):
    """Single Graph Transformer layer with edge features"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 edge_dim: Optional[int] = None):
        super().__init__()
        
        self.transformer_conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True,  # Learn to weight self-loops
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
        # Multi-head attention with residual
        h = self.transformer_conv(x, edge_index, edge_attr)
        x = self.norm1(x + h)
        
        # Feed-forward with residual
        h = self.ffn(x)
        x = self.norm2(x + h)
        
        return x


class GraphTransformerWithEmbeddings(nn.Module):
    """
    Graph Transformer for next event time prediction with:
    - Categorical embeddings for nodes and edges
    - Lookahead features (next event information)
    - Enhanced edge features with distance and region
    - Proper handling of from/to locations and postal codes
    """
    
    def __init__(self, config: ModelConfig, vocab_sizes: Dict[str, int]):
        """
        Args:
            config: ModelConfig object
            vocab_sizes: Dict mapping category name to vocabulary size
        """
        super().__init__()
        
        self.config = config
        self.vocab_sizes = vocab_sizes
        
        # === Node Embeddings (9 features) ===
        node_categorical_features = [
            'event_type',      # Current event type
            'from_location',   # From location (sort_center or delivery_station)
            'to_location',     # To location (sort_center or delivery_station)
            'to_postal',       # To postal code (DELIVERY only)
            'from_region',     # From region
            'to_region',       # To region
            'carrier',         # Carrier ID
            'leg_type',        # Leg type (FORWARD, etc.)
            'ship_method'      # Ship method
        ]
        self.node_embedding = MultiEmbedding(
            vocab_sizes=vocab_sizes,
            embed_dim=config.embed_dim,
            feature_names=node_categorical_features,
            dropout=config.dropout
        )
        
        # === Lookahead Embeddings (7 features) ===
        lookahead_categorical_features = [
            'next_event_type',   # Next event type
            'next_location',     # Next location
            'next_postal',       # Next postal code (DELIVERY only)
            'next_region',       # Next region
            'next_carrier',      # Next carrier
            'next_leg_type',     # Next leg type
            'next_ship_method'   # Next ship method
        ]
        self.lookahead_embedding = MultiEmbedding(
            vocab_sizes=vocab_sizes,
            embed_dim=config.embed_dim,
            feature_names=lookahead_categorical_features,
            dropout=config.dropout
        )
        
        # === Edge Embeddings (9 features) ===
        edge_categorical_features = [
            'edge_from_location',     # Source location
            'edge_to_location',       # Target location
            'edge_to_postal',         # Target postal code (DELIVERY only)
            'edge_from_region',       # Source region
            'edge_to_region',         # Target region
            'edge_carrier_from',      # Source carrier
            'edge_carrier_to',        # Target carrier
            'edge_ship_method_from',  # Source ship method
            'edge_ship_method_to'     # Target ship method
        ]
        self.edge_embedding = MultiEmbedding(
            vocab_sizes=vocab_sizes,
            embed_dim=config.embed_dim,
            feature_names=edge_categorical_features,
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
        node_cat_dim = self.node_embedding.get_output_dim()            # 9 * embed_dim
        lookahead_cat_dim = self.lookahead_embedding.get_output_dim()  # 7 * embed_dim
        package_postal_dim = config.embed_dim * 2                      # source + dest postal
        
        total_node_input_dim = (
            config.node_continuous_dim +
            node_cat_dim +
            lookahead_cat_dim +
            package_postal_dim
        )
        
        edge_cat_dim = self.edge_embedding.get_output_dim()  # 9 * embed_dim
        total_edge_input_dim = config.edge_continuous_dim + edge_cat_dim
        
        print(f"\n=== Model Dimensions ===")
        print(f"Node input dim: {total_node_input_dim}")
        print(f"  - Continuous: {config.node_continuous_dim}")
        print(f"  - Node categorical ({len(node_categorical_features)} features): {node_cat_dim}")
        print(f"  - Lookahead categorical ({len(lookahead_categorical_features)} features): {lookahead_cat_dim}")
        print(f"  - Package postal: {package_postal_dim}")
        print(f"Edge input dim: {total_edge_input_dim}")
        print(f"  - Continuous: {config.edge_continuous_dim}")
        print(f"  - Edge categorical ({len(edge_categorical_features)} features): {edge_cat_dim}")
        print(f"Hidden dim: {config.hidden_dim}")
        print(f"Num layers: {config.num_layers}")
        print(f"Num heads: {config.num_heads}")
        
        # === Input Projections ===
        self.node_input_proj = nn.Sequential(
            nn.Linear(total_node_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.edge_input_proj = nn.Sequential(
            nn.Linear(total_edge_input_dim, config.hidden_dim),
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
        
        # === Output Projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        self._init_weights()
        
        # Print parameter count
        params = self.get_num_parameters()
        print(f"\nTotal parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
    
    def _init_weights(self):
        """Initialize projection layer weights"""
        for module in [self.node_input_proj, self.edge_input_proj, self.output_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: PyG Data object from PackageLifecycleDataset
        
        Returns:
            Predictions tensor [num_nodes, output_dim]
        """
        batch = data.batch
        
        # === Node Features ===
        node_continuous = data.node_continuous
        
        # Node categorical embeddings (9 features)
        node_cat_indices = {
            'event_type': data.event_type_idx,
            'from_location': data.from_location_idx,
            'to_location': data.to_location_idx,
            'to_postal': data.to_postal_idx,
            'from_region': data.from_region_idx,
            'to_region': data.to_region_idx,
            'carrier': data.carrier_idx,
            'leg_type': data.leg_type_idx,
            'ship_method': data.ship_method_idx,
        }
        node_cat_emb = self.node_embedding(node_cat_indices)
        
        # Lookahead categorical embeddings (7 features)
        lookahead_cat_indices = {
            'next_event_type': data.next_event_type_idx,
            'next_location': data.next_location_idx,
            'next_postal': data.next_postal_idx,
            'next_region': data.next_region_idx,
            'next_carrier': data.next_carrier_idx,
            'next_leg_type': data.next_leg_type_idx,
            'next_ship_method': data.next_ship_method_idx,
        }
        lookahead_cat_emb = self.lookahead_embedding(lookahead_cat_indices)
        
        # Package postal embeddings (expanded to all nodes in batch)
        source_postal_emb = self.postal_embedding(data.source_postal_idx)  # [batch_size, embed_dim]
        dest_postal_emb = self.postal_embedding(data.dest_postal_idx)      # [batch_size, embed_dim]
        
        # Expand to match nodes using batch index
        source_postal_expanded = source_postal_emb[batch]  # [num_nodes, embed_dim]
        dest_postal_expanded = dest_postal_emb[batch]      # [num_nodes, embed_dim]
        package_postal_emb = torch.cat([source_postal_expanded, dest_postal_expanded], dim=-1)
        
        # Combine all node features
        node_features = torch.cat([
            node_continuous,
            node_cat_emb,
            lookahead_cat_emb,
            package_postal_emb
        ], dim=-1)
        
        # Project to hidden dimension
        x = self.node_input_proj(node_features)
        
        # === Edge Features ===
        edge_index = data.edge_index
        edge_attr = None
        
        if self.config.use_edge_features and data.edge_continuous.numel() > 0:
            edge_continuous = data.edge_continuous
            
            # Edge categorical embeddings (9 features)
            edge_cat_indices = {
                'edge_from_location': data.edge_from_location_idx,
                'edge_to_location': data.edge_to_location_idx,
                'edge_to_postal': data.edge_to_postal_idx,
                'edge_from_region': data.edge_from_region_idx,
                'edge_to_region': data.edge_to_region_idx,
                'edge_carrier_from': data.edge_carrier_from_idx,
                'edge_carrier_to': data.edge_carrier_to_idx,
                'edge_ship_method_from': data.edge_ship_method_from_idx,
                'edge_ship_method_to': data.edge_ship_method_to_idx,
            }
            edge_cat_emb = self.edge_embedding(edge_cat_indices)
            
            # Combine edge features
            edge_features = torch.cat([edge_continuous, edge_cat_emb], dim=-1)
            edge_attr = self.edge_input_proj(edge_features)
        
        # === Positional Encoding ===
        x = self.pos_encoding(x, batch)
        
        # === Transformer Layers ===
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # === Output Projection ===
        out = self.output_proj(x)
        
        return out
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable
        }
    
    def get_embedding_weights(self) -> Dict[str, torch.Tensor]:
        """Get embedding weights for analysis"""
        weights = {}
        
        for name, emb in self.node_embedding.embeddings.items():
            weights[f'node_{name}'] = emb.weight.detach().clone()
        
        for name, emb in self.lookahead_embedding.embeddings.items():
            weights[f'lookahead_{name}'] = emb.weight.detach().clone()
        
        for name, emb in self.edge_embedding.embeddings.items():
            weights[f'edge_{name}'] = emb.weight.detach().clone()
        
        weights['postal'] = self.postal_embedding.weight.detach().clone()
        
        return weights
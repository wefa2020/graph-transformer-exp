import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for sequential data"""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x, batch):
        """Add positional encoding to node features"""
        # Get sequence position for each node
        _, counts = torch.unique(batch, return_counts=True)
        positions = torch.cat([torch.arange(c) for c in counts]).to(x.device)
        
        return x + self.pe[positions]

class GraphTransformerLayer(nn.Module):
    """Single Graph Transformer layer"""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_edge_features=True):
        super().__init__()
        
        self.transformer_conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=None if not use_edge_features else hidden_dim,
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
    
    def forward(self, x, edge_index, edge_attr=None):
        # Multi-head attention
        h = self.transformer_conv(x, edge_index, edge_attr)
        x = self.norm1(x + h)
        
        # Feed-forward
        h = self.ffn(x)
        x = self.norm2(x + h)
        
        return x

class GraphTransformer(nn.Module):
    """Graph Transformer for package event sequences"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        if config.use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(config.edge_feature_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU()
            )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_edge_features=config.use_edge_features
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Encode inputs
        x = self.node_encoder(x)
        
        if self.config.use_edge_features and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None
        
        # Add positional encoding
        x = self.pos_encoding(x, batch)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Output projection
        out = self.output_proj(x)
        
        return out
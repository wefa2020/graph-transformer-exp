import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from config import ModelConfig
from models.graph_transformer import GraphTransformerWithEmbeddings


class EventTimePredictor(nn.Module):
    """Complete model for predicting next event time with categorical embeddings"""
    
    def __init__(self, config: ModelConfig, vocab_sizes: Dict[str, int]):
        """
        Args:
            config: ModelConfig object with model architecture settings
            vocab_sizes: Dict mapping category name to vocabulary size
        """
        super().__init__()
        
        self.config = config
        self.vocab_sizes = vocab_sizes
        
        # Initialize Graph Transformer with embeddings
        self.graph_transformer = GraphTransformerWithEmbeddings(config, vocab_sizes)
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: PyG Data batch from PackageLifecycleDataset
            
        Returns:
            predictions: Predicted time to next event for each node [num_nodes, 1]
        """
        predictions = self.graph_transformer(data)
        return predictions
    
    @torch.no_grad()
    def predict(self, data, return_all: bool = False) -> torch.Tensor:
        """
        Predict time to next event
        
        Args:
            data: PyG Data batch
            return_all: If True, return predictions for all nodes.
                       If False, return only predictions for nodes with valid labels.
        
        Returns:
            predictions: Predicted times [num_valid_nodes, 1] or [num_nodes, 1]
        """
        self.eval()
        
        predictions = self.forward(data)
        
        if return_all:
            return predictions
        else:
            # Return only predictions for nodes with labels (exclude last node per graph)
            if hasattr(data, 'label_mask'):
                return predictions[data.label_mask]
            else:
                return self._exclude_last_nodes(predictions, data.batch)
    
    @torch.no_grad()
    def predict_next_event(self, data) -> torch.Tensor:
        """
        Predict time to next event for the last node in each graph.
        Useful for inference on partial lifecycles.
        
        Args:
            data: PyG Data batch
            
        Returns:
            last_node_preds: Predictions for last node of each graph [batch_size, 1]
        """
        self.eval()
        
        predictions = self.forward(data)
        
        # Get prediction for last node in each graph
        batch_size = data.batch.max().item() + 1
        last_node_preds = []
        
        for i in range(batch_size):
            mask = data.batch == i
            graph_preds = predictions[mask]
            last_pred = graph_preds[-1]  # Last node prediction
            last_node_preds.append(last_pred)
        
        return torch.stack(last_node_preds)
    
    @torch.no_grad()
    def predict_all_transitions(self, data) -> List[Dict]:
        """
        Predict time for all transitions in each package lifecycle.
        
        Args:
            data: PyG Data batch
            
        Returns:
            List of dicts with predictions for each graph
        """
        self.eval()
        
        predictions = self.forward(data)
        batch_size = data.batch.max().item() + 1
        
        results = []
        
        for i in range(batch_size):
            mask = data.batch == i
            graph_preds = predictions[mask].cpu().numpy()
            
            # Get number of nodes in this graph
            num_nodes = mask.sum().item()
            
            # All nodes except last have valid predictions
            valid_preds = graph_preds[:-1] if num_nodes > 1 else graph_preds
            
            results.append({
                'graph_idx': i,
                'num_events': num_nodes,
                'predictions': valid_preds.flatten().tolist(),
                'last_event_prediction': graph_preds[-1].flatten().tolist()
            })
        
        return results
    
    def _exclude_last_nodes(self, preds: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Exclude the last node from each graph in the batch.
        
        Args:
            preds: Predictions for all nodes [num_nodes, ...]
            batch: Batch indices indicating which graph each node belongs to
        
        Returns:
            Predictions excluding last node of each graph
        """
        num_graphs = batch.max().item() + 1
        mask = torch.ones(len(preds), dtype=torch.bool, device=preds.device)
        
        for graph_id in range(num_graphs):
            node_indices = (batch == graph_id).nonzero(as_tuple=True)[0]
            if len(node_indices) > 0:
                last_node_idx = node_indices[-1]
                mask[last_node_idx] = False
        
        return preds[mask]
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters"""
        return self.graph_transformer.get_num_parameters()
    
    def get_embedding_weights(self) -> Dict[str, torch.Tensor]:
        """Get embedding weights for analysis"""
        return self.graph_transformer.get_embedding_weights()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None) -> 'EventTimePredictor':
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Target device
        
        Returns:
            Loaded EventTimePredictor model
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Reconstruct config
        config = ModelConfig.from_dict(checkpoint['model_config'])
        vocab_sizes = checkpoint['vocab_sizes']
        
        # Create model
        model = cls(config, vocab_sizes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        
        return model
    
    @classmethod
    def from_preprocessor(cls, preprocessor, device: torch.device = None, 
                          **config_kwargs) -> 'EventTimePredictor':
        """
        Create model from fitted preprocessor
        
        Args:
            preprocessor: Fitted PackageLifecyclePreprocessor
            device: Target device
            **config_kwargs: Override config values (e.g., hidden_dim=512)
        
        Returns:
            Initialized EventTimePredictor model
        """
        config = ModelConfig.from_preprocessor(preprocessor, **config_kwargs)
        vocab_sizes = preprocessor.get_vocab_sizes()
        
        model = cls(config, vocab_sizes)
        
        if device is not None:
            model.to(device)
        
        return model

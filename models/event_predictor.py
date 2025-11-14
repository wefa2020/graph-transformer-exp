import torch
import torch.nn as nn
from models.graph_transformer import GraphTransformer

class EventTimePredictor(nn.Module):
    """Complete model for predicting next event time"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.graph_transformer = GraphTransformer(config.model)
    
    def forward(self, data):
        """
        Args:
            data: PyG Data batch
            
        Returns:
            predictions: Predicted time to next event for each node
        """
        predictions = self.graph_transformer(data)
        return predictions
    
    def predict_next_event(self, data):
        """Predict time to next event given partial lifecycle"""
        self.eval()
        
        with torch.no_grad():
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
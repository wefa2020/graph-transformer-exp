import torch
import numpy as np
from datetime import datetime, timedelta
import json

from config import Config
from models.event_predictor import EventTimePredictor
from data.dataset import PackageLifecycleDataset, collate_fn

class EventTimeInference:
    """Inference class for predicting next event times"""
    
    def __init__(self, checkpoint_path):
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.config = self.checkpoint['config']
        self.preprocessor = self.checkpoint['preprocessor']
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EventTimePredictor(self.config).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from epoch {self.checkpoint['epoch']}")
        print(f"Validation loss: {self.checkpoint['val_loss']:.4f}")
    
    def predict_next_event_time(self, partial_lifecycle: dict) -> dict:
        """
        Predict next event time given partial package lifecycle
        
        Args:
            partial_lifecycle: Dictionary with package info and events list
                {
                    'package_id': str,
                    'events': [
                        {
                            'event_type': str,
                            'sort_center': str,
                            'event_time': str (ISO format),
                            ...
                        },
                        ...
                    ],
                    'weight': float,
                    'length': float,
                    'width': float,
                    'height': float
                }
        
        Returns:
            Dictionary with prediction results
        """
        
        # Preprocess
        processed = self.preprocessor.process_lifecycle(
            partial_lifecycle,
            return_labels=False
        )
        
        # Convert to PyG Data
        from torch_geometric.data import Data
        
        x = torch.tensor(processed['node_features'], dtype=torch.float)
        edge_index = torch.tensor(processed['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(processed['edge_features'], dtype=torch.float)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=processed['num_nodes']
        )
        
        data = data.to(self.device)
        
        # ✅ FIX: Create batch vector for single graph
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(data)
            
            # Get prediction for last node
            last_pred = predictions[-1].cpu().numpy()
            
            # Inverse transform
            time_delta_hours = self.preprocessor.inverse_transform_time(
                last_pred.reshape(1, -1)
            )[0, 0]
        
        # Calculate predicted event time
        last_event_time = datetime.fromisoformat(
            partial_lifecycle['events'][-1]['event_time'].replace('Z', '+00:00')
        )
        predicted_event_time = last_event_time + timedelta(hours=float(time_delta_hours))
        
        result = {
            'package_id': partial_lifecycle['package_id'],
            'last_event_type': partial_lifecycle['events'][-1]['event_type'],
            'last_event_time': last_event_time.isoformat(),
            'predicted_time_delta_hours': float(time_delta_hours),
            'predicted_next_event_time': predicted_event_time.isoformat(),
            'num_events_processed': len(partial_lifecycle['events'])
        }
        
        return result
    
    def batch_predict(self, lifecycles: list) -> list:
        """Predict for multiple lifecycles"""
        
        results = []
        
        for lifecycle in lifecycles:
            try:
                result = self.predict_next_event_time(lifecycle)
                results.append(result)
            except Exception as e:
                print(f"Error predicting for package {lifecycle['package_id']}: {e}")
                results.append({
                    'package_id': lifecycle['package_id'],
                    'error': str(e)
                })
        
        return results

def main():
    """Example inference usage"""
    
    # Load model
    inference = EventTimeInference('checkpoints/best_model.pt')
    
    # Example partial lifecycle
    partial_lifecycle =   {
    "package_id": "9361289768755732386935",
    "source_postal": "90301",
    "dest_postal": "77840",
    "pdd": "2025-11-04 04:00:00",
    "weight": 26.0,
    "length": 15.0,
    "width": 12.0,
    "height": 4.0,
    "events": [
      {
        "event_type": "INDUCT",
        "sort_center": "ONT5",
        "event_time": "2025-10-29 06:21:13",
        "plan_time": "2025-10-30T09:00:00",
        "cpt": "2025-10-30 09:00:00",
        "leg_type": "FORWARD",
        "carrier_id": "USPS",
        "load_id": "1142DDW3R",
        "ship_method": 'UNKNOWN'
      },
      {
        "event_type": "EXIT",
        "sort_center": "ONT5",
        "event_time": "2025-10-30 09:54:19",
        "dwelling_seconds": 99180.0,
        "leg_type": "FORWARD",
        "carrier_id": "USPS"
      },
      {
        "event_type": "LINEHAUL",
        "sort_center": "HOU1",
        "event_time": "2025-10-31 14:33:31",
        "plan_time": "2025-11-01T15:45:00",
        "cpt": "2025-11-02 08:30:00",
        "leg_type": "FORWARD",
        "carrier_id": "USPS",
        "ship_method": "AMTRAN_SORTCENTER"
      },
      {
        "event_type": "EXIT",
        "sort_center": "HOU1",
        "event_time": "2025-11-01 01:04:05",
        "dwelling_seconds": 37834.0,
        "leg_type": "FORWARD",
        "carrier_id": "USPS"
      },
      {
        "event_type": "DELIVERY",
        "event_time": "2025-11-01 10:01:00",
        "plan_time": "2025-11-03T04:00:00",
        "delivery_station": "HOU1",
        "ship_method": "USPS_ATS_PARCEL",
        "delivery_location": {
          "id": "77840",
          "city": "College Station",
          "county": "Brazos",
          "state": "TX",
          "lat": "30.6115",
          "lng": "-96.32332"
        }
      }
    ],
    "num_events": 5
  }
   
    
    
    
    # Predict
    result = inference.predict_next_event_time(partial_lifecycle)
    
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    print(f"Package ID: {result['package_id']}")
    print(f"Last Event: {result['last_event_type']} at {result['last_event_time']}")
    print(f"Predicted Time Delta: {result['predicted_time_delta_hours']:.2f} hours")
    print(f"Predicted Next Event Time: {result['predicted_next_event_time']}")
    print("="*80)
    
    # Save result
    with open('prediction_result.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()
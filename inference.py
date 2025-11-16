import torch
import numpy as np
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional, Union
import os
import copy

from config import Config
from models.event_predictor import EventTimePredictor
from data.dataset import PackageLifecycleDataset, collate_fn
from data.neptune_extractor import NeptuneDataExtractor
from data.data_preprocessor import PackageLifecyclePreprocessor

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, timedelta)):
            return obj.isoformat()
        return super().default(obj)

class EventTimeInference:
    """Inference class for predicting event times from Neptune data"""
    
    def __init__(self, checkpoint_path: str, neptune_endpoint: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            neptune_endpoint: Neptune endpoint (optional, for fetching data)
        """
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.config = self.checkpoint['config']
        self.preprocessor = self.checkpoint['preprocessor']
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EventTimePredictor(self.config).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Loaded model from epoch {self.checkpoint['epoch']}")
        print(f"✓ Validation loss: {self.checkpoint['val_loss']:.4f}")
        print(f"✓ Using device: {self.device}")
        
        # Initialize Neptune extractor if endpoint provided
        self.neptune_extractor = None
        if neptune_endpoint:
            self.neptune_extractor = NeptuneDataExtractor(
                endpoint=neptune_endpoint,
                max_workers=10
            )
            print(f"✓ Connected to Neptune: {neptune_endpoint}")
    
    def fetch_package_from_neptune(self, package_id: str) -> Optional[Dict]:
        """
        Fetch package lifecycle from Neptune
        
        Args:
            package_id: Package ID to fetch
            
        Returns:
            Package lifecycle dictionary or None if not found
        """
        if not self.neptune_extractor:
            raise ValueError("Neptune extractor not initialized. Provide neptune_endpoint in constructor.")
        
        print(f"Fetching package {package_id} from Neptune...")
        package_data = self.neptune_extractor.process_package(package_id)
        
        if package_data is None:
            print(f"✗ Package {package_id} not found or invalid")
            return None
        
        print(f"✓ Fetched package with {package_data['num_events']} events")
        return package_data
    
    def predict_single_step(self, partial_lifecycle: Dict) -> float:
        """
        Predict time delta for next event given partial lifecycle
        
        Args:
            partial_lifecycle: Dictionary with package info and events list
            
        Returns:
            Predicted time delta in hours
        """
        # Preprocess using the loaded preprocessor
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
        
        # Create batch vector for single graph
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(data)
            
            # Get prediction for last node (the one we're predicting from)
            last_pred = predictions[-1].cpu().numpy()
            
            # Inverse transform
            time_delta_hours = self.preprocessor.inverse_transform_time(
                last_pred.reshape(1, -1)
            )[0, 0]
        
        return float(time_delta_hours)
    
    def convert_to_serializable(self, obj):
        """
        Convert non-JSON-serializable objects to serializable format
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def predict_full_lifecycle(self, package_lifecycle: Dict) -> Dict:
        """
        Predict all event times starting from second event until delivery.
        Returns the same format as Neptune data with added prediction fields.
        
        Args:
            package_lifecycle: Complete package lifecycle from Neptune
            
        Returns:
            Dictionary in same format as Neptune with prediction fields added to events
        """
        # Deep copy to avoid modifying original
        result = copy.deepcopy(package_lifecycle)
        events = result['events']
        package_id = result['package_id']
        
        # Need at least 1 event
        if len(events) < 1:
            result['error'] = 'No events found'
            return result
        
        # Start from second event (index 1) and add predictions
        for i in range(1, len(events)):
            current_event = events[i]
            
            # Create partial lifecycle up to current event (not including it)
            partial_lifecycle = {
                'package_id': package_id,
                'source_postal': package_lifecycle['source_postal'],
                'dest_postal': package_lifecycle['dest_postal'],
                'pdd': package_lifecycle['pdd'],
                'weight': package_lifecycle['weight'],
                'length': package_lifecycle['length'],
                'width': package_lifecycle['width'],
                'height': package_lifecycle['height'],
                'events': events[:i]  # Events up to (not including) current event
            }
            
            # Predict time to current event
            try:
                predicted_delta_hours = self.predict_single_step(partial_lifecycle)
                
                # Calculate times
                previous_event_time = events[i-1]['event_time']
                actual_event_time = current_event['event_time']
                
                predicted_event_time = previous_event_time + timedelta(hours=predicted_delta_hours)
                
                # Calculate actual time delta
                actual_delta = actual_event_time - previous_event_time
                actual_delta_hours = actual_delta.total_seconds() / 3600
                
                # Calculate error
                error_hours = predicted_delta_hours - actual_delta_hours
                
                # Add prediction fields to the event
                events[i]['prediction'] = {
                    'predicted_event_time': predicted_event_time.isoformat(),
                    'predicted_delta_hours': round(predicted_delta_hours, 2),
                    'actual_delta_hours': round(actual_delta_hours, 2),
                    'error_hours': round(error_hours, 2)
                }
                
            except Exception as e:
                print(f"    ✗ Failed to predict event {i}: {e}")
                events[i]['prediction'] = {
                    'error': f"Prediction failed: {str(e)}"
                }
        
        # Convert to JSON-serializable format
        result = self.convert_to_serializable(result)
        
        return result
    
    def predict_from_package_id(self, package_id: str) -> Dict:
        """
        Fetch package from Neptune and predict full lifecycle
        
        Args:
            package_id: Package ID to predict
            
        Returns:
            Prediction results dictionary in Neptune format
        """
        # Fetch from Neptune
        package_data = self.fetch_package_from_neptune(package_id)
        
        if package_data is None:
            return {
                'package_id': package_id,
                'error': 'Package not found in Neptune'
            }
        
        # Predict
        return self.predict_full_lifecycle(package_data)
    
    def batch_predict_from_package_ids(self, package_ids: List[str]) -> List[Dict]:
        """
        Predict for multiple packages by fetching from Neptune
        
        Args:
            package_ids: List of package IDs
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"\nProcessing {len(package_ids)} packages...")
        
        for idx, package_id in enumerate(package_ids, 1):
            print(f"\n[{idx}/{len(package_ids)}] Processing {package_id}...")
            
            try:
                result = self.predict_from_package_id(package_id)
                results.append(result)
                
                # Check if prediction was successful
                if 'error' in result:
                    print(f"  ✗ {result['error']}")
                else:
                    # Count successful predictions
                    predicted_events = [e for e in result['events'] if 'prediction' in e]
                    successful = [e for e in predicted_events if 'error' not in e['prediction']]
                    print(f"  ✓ Predicted {len(successful)}/{len(predicted_events)} events")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'package_id': package_id,
                    'error': str(e)
                })
        
        return results
    
    def batch_predict_from_lifecycles(self, lifecycles: List[Dict]) -> List[Dict]:
        """
        Predict for multiple package lifecycles (already fetched data)
        
        Args:
            lifecycles: List of package lifecycle dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"\nProcessing {len(lifecycles)} packages...")
        
        for idx, lifecycle in enumerate(lifecycles, 1):
            package_id = lifecycle.get('package_id', f'unknown_{idx}')
            print(f"\n[{idx}/{len(lifecycles)}] Processing {package_id}...")
            
            try:
                result = self.predict_full_lifecycle(lifecycle)
                results.append(result)
                
                # Check if prediction was successful
                if 'error' in result:
                    print(f"  ✗ {result['error']}")
                else:
                    # Count successful predictions
                    predicted_events = [e for e in result['events'] if 'prediction' in e]
                    successful = [e for e in predicted_events if 'error' not in e['prediction']]
                    print(f"  ✓ Predicted {len(successful)}/{len(predicted_events)} events")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'package_id': package_id,
                    'error': str(e)
                })
        
        return results
    
    def save_results(self, results: Union[Dict, List[Dict]], output_path: str):
        """Save prediction results to JSON file"""
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Convert to JSON-serializable format if not already done
        serializable_results = self.convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
        
        print(f"\n✓ Saved results to {output_path}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics for batch predictions"""
        
        successful_packages = [r for r in results if 'error' not in r and 'events' in r]
        failed_packages = [r for r in results if 'error' in r]
        
        print(f"\n{'='*80}")
        print("BATCH PREDICTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total packages: {len(results)}")
        print(f"Successful: {len(successful_packages)}")
        print(f"Failed: {len(failed_packages)}")
        
        if successful_packages:
            total_predictions = 0
            successful_predictions = 0
            
            for r in successful_packages:
                predicted_events = [e for e in r['events'] if 'prediction' in e]
                total_predictions += len(predicted_events)
                successful_predictions += len([e for e in predicted_events if 'error' not in e['prediction']])
            
            print(f"\nPrediction Statistics:")
            print(f"  Total event predictions: {total_predictions}")
            print(f"  Successful predictions: {successful_predictions}")
            print(f"  Failed predictions: {total_predictions - successful_predictions}")
            if total_predictions > 0:
                print(f"  Success rate: {successful_predictions/total_predictions*100:.1f}%")
        
        print(f"{'='*80}\n")
    
    def close(self):
        """Clean up resources"""
        if self.neptune_extractor:
            self.neptune_extractor.close()
            print("✓ Closed Neptune connections")


def main():
    """Batch prediction from Neptune package IDs"""
    
    # Configuration
    CHECKPOINT_PATH = '/home/ubuntu/graph-transformer-exp/checkpoints/best_model.pt'
    NEPTUNE_ENDPOINT = "swa-shipgraph-neptune-instance-prod-us-east-1.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182"
    
    # ============================================================================
    # CONFIGURE PACKAGE IDs HERE
    # ============================================================================
    package_ids = [
        "TBA325482265502",
        "TBA325586913247",
        "TBA325378303118",
        "TBA324209756082",
        "TBA324290334010",
        "TBA324283629240",
        "TBA324259370787",
        "TBA324959772585",
        "TBA325313805791"

        # Add more package IDs here
    ]
    # ============================================================================
    
    print("\n" + "="*80)
    print("EVENT TIME PREDICTION - BATCH INFERENCE")
    print("="*80)
    print(f"Will process {len(package_ids)} packages")
    print("="*80 + "\n")
    
    # Initialize inference engine
    inference = EventTimeInference(
        checkpoint_path=CHECKPOINT_PATH,
        neptune_endpoint=NEPTUNE_ENDPOINT
    )
    
    try:
        # Run batch prediction from Neptune
        results = inference.batch_predict_from_package_ids(package_ids)
        
        # Print summary
        inference.print_summary(results)
        
        # Save results
        output_file = 'prediction_results.json'
        inference.save_results(results, output_file)
        
        print("\n" + "="*80)
        print("INFERENCE COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        inference.close()


# ============================================================================
# ALTERNATIVE: Batch prediction from JSON file (commented out)
# ============================================================================
# def main():
#     """Batch prediction from JSON file"""
#     
#     # Configuration
#     CHECKPOINT_PATH = 'checkpoints/best_model.pt'
#     JSON_FILE_PATH = 'data/graph-data/package_lifecycles.json'
#     LIMIT = 10  # Number of packages to process (None for all)
#     
#     print("\n" + "="*80)
#     print("EVENT TIME PREDICTION - BATCH INFERENCE FROM FILE")
#     print("="*80)
#     print(f"Loading from: {JSON_FILE_PATH}")
#     print("="*80 + "\n")
#     
#     # Initialize inference engine (no Neptune needed)
#     inference = EventTimeInference(checkpoint_path=CHECKPOINT_PATH)
#     
#     try:
#         # Load lifecycles from JSON file
#         print(f"Loading lifecycles from {JSON_FILE_PATH}...")
#         with open(JSON_FILE_PATH, 'r') as f:
#             lifecycles = json.load(f)
#         
#         print(f"✓ Loaded {len(lifecycles)} packages")
#         
#         # Limit if specified
#         if LIMIT is not None:
#             lifecycles = lifecycles[:LIMIT]
#             print(f"Processing first {len(lifecycles)} packages")
#         
#         # Run batch prediction
#         results = inference.batch_predict_from_lifecycles(lifecycles)
#         
#         # Print summary
#         inference.print_summary(results)
#         
#         # Save results
#         output_file = 'prediction_results.json'
#         inference.save_results(results, output_file)
#         
#         print("\n" + "="*80)
#         print("INFERENCE COMPLETE")
#         print("="*80)
#         
#     except Exception as e:
#         print(f"\n✗ Error during inference: {e}")
#         import traceback
#         traceback.print_exc()
#         
#     finally:
#         # Clean up
#         inference.close()


if __name__ == '__main__':
    main()
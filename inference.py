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
        
        print(f"✓ Loaded model from epoch {self.checkpoint.get('epoch', 'unknown')}")
        
        # Print validation metrics if available
        if 'val_loss' in self.checkpoint:
            print(f"✓ Validation loss: {self.checkpoint['val_loss']:.4f}")
        if 'val_mae' in self.checkpoint:
            print(f"✓ Validation MAE: {self.checkpoint['val_mae']:.4f} hours")
        if 'val_mape' in self.checkpoint:
            print(f"✓ Validation MAPE: {self.checkpoint['val_mape']:.2f}%")
        
        # Print training metrics if available
        if 'train_loss' in self.checkpoint:
            print(f"  Train loss: {self.checkpoint['train_loss']:.4f}")
        
        print(f"✓ Using device: {self.device}")
        
        # Initialize Neptune extractor if endpoint provided
        self.neptune_extractor = None
        if neptune_endpoint:
            self.neptune_extractor = NeptuneDataExtractor(
                endpoint=neptune_endpoint,
                max_workers=10
            )
            print(f"✓ Connected to Neptune: {neptune_endpoint}")
    
    def _parse_datetime(self, time_value: Union[str, datetime, None]) -> Union[datetime, None]:
        """
        Parse time value which can be:
        - datetime object (already parsed)
        - ISO string
        - None
        
        Returns:
            datetime object or None
        """
        if time_value is None:
            return None
        
        # Already a datetime object
        if isinstance(time_value, datetime):
            return time_value
        
        # String - try to parse
        if isinstance(time_value, str):
            if time_value == 'null' or time_value.strip() == '':
                return None
            
            try:
                # Handle ISO format with or without Z
                return datetime.fromisoformat(str(time_value).replace('Z', '+00:00'))
            except Exception as e:
                return None
        
        return None
    
    def _validate_package_data(self, package_data: Dict) -> bool:
        """
        Validate package data structure
        
        Args:
            package_data: Package lifecycle dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not package_data:
            print("  ✗ Package data is None or empty")
            return False
        
        if 'events' not in package_data:
            print("  ✗ Package data missing 'events' field")
            return False
        
        if package_data['events'] is None:
            print("  ✗ Events field is None")
            return False
        
        if not isinstance(package_data['events'], list):
            print(f"  ✗ Events field is not a list, got {type(package_data['events'])}")
            return False
        
        if len(package_data['events']) == 0:
            print("  ✗ Events list is empty")
            return False
        
        # Validate required fields
        required_fields = ['package_id', 'events']
        for field in required_fields:
            if field not in package_data:
                print(f"  ✗ Missing required field: {field}")
                return False
        
        # Validate each event has required fields
        for i, event in enumerate(package_data['events']):
            if not isinstance(event, dict):
                print(f"  ✗ Event {i} is not a dictionary")
                return False
            
            if 'event_type' not in event:
                print(f"  ✗ Event {i} missing event_type")
                return False
            
            if 'event_time' not in event:
                print(f"  ✗ Event {i} missing event_time")
                return False
        
        return True
    
    def _sort_events_by_sort_center(self, events: List[Dict]) -> List[Dict]:
        """
        ✅ Sort events by sort center and time
        
        Sorting logic:
        1. Group events by sort center (DELIVERY events use "DELIVERY" as their group key)
        2. Order sort centers by their earliest event time (ascending)
        3. Within each sort center, sort events by event_time (ascending)
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Sorted list of events
            
        Raises:
            ValueError: If any event has invalid or missing event_time
        """
        if not events:
            return []
        
        if not isinstance(events, list):
            raise ValueError(f"Events must be a list, got {type(events)}")
        
        # STEP 1: Parse and validate all event times
        for i, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"Event {i} is not a dictionary")
            
            event_time = self._parse_datetime(event.get('event_time'))
            if event_time is None:
                raise ValueError(
                    f"Invalid or missing event_time in event {i}: {event.get('event_type', 'UNKNOWN')} "
                    f"at sort_center: {event.get('sort_center', 'UNKNOWN')}"
                )
            event['_parsed_time'] = event_time
        
        # STEP 2: Group events by sort center
        sort_center_events = {}
        
        for event in events:
            event_type = event.get('event_type')
            
            # DELIVERY events don't have sort_center field - use "DELIVERY" as key
            if event_type == 'DELIVERY':
                sc_key = 'DELIVERY'
            else:
                sort_center = event.get('sort_center')
                
                # Handle None or 'null' sort centers
                if not sort_center or sort_center == 'null':
                    sc_key = 'UNKNOWN'
                else:
                    sc_key = str(sort_center)
            
            if sc_key not in sort_center_events:
                sort_center_events[sc_key] = []
            
            sort_center_events[sc_key].append(event)
        
        # STEP 3: Sort events within each sort center by event_time
        for sc_key in sort_center_events:
            sort_center_events[sc_key].sort(key=lambda e: e['_parsed_time'])
        
        # STEP 4: Order sort centers by their earliest event time
        sort_center_order = sorted(
            sort_center_events.keys(),
            key=lambda sc: sort_center_events[sc][0]['_parsed_time']  # First event (earliest) in each SC
        )
        
        # STEP 5: Reconstruct sorted event list
        sorted_events = []
        for sc_key in sort_center_order:
            sorted_events.extend(sort_center_events[sc_key])
        
        # Clean up temporary parsed time field
        for event in sorted_events:
            del event['_parsed_time']
        
        return sorted_events
    
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
        
        # Validate package data
        if not self._validate_package_data(package_data):
            print(f"✗ Package {package_id} has invalid data structure")
            return None
        
        print(f"✓ Fetched package with {len(package_data['events'])} events")
        
        # ✅ Apply event sorting
        try:
            package_data['events'] = self._sort_events_by_sort_center(package_data['events'])
            print(f"✓ Sorted events by sort center and time")
        except ValueError as e:
            print(f"✗ Failed to sort events: {e}")
            return None
        
        return package_data
    
    def predict_single_step(self, partial_lifecycle: Dict) -> Dict:
        """
        Predict transit time to next event given partial lifecycle
        
        Args:
            partial_lifecycle: Dictionary with package info and events list
            
        Returns:
            Dictionary with:
                - predicted_transit_hours: Predicted time from last event to next event
                - last_event_time: Time of last known event
                - predicted_next_event_time: Predicted time of next event (last_time + transit)
        
        Raises:
            ValueError: If partial_lifecycle is invalid
        """
        # Validate input
        if not partial_lifecycle:
            raise ValueError("partial_lifecycle is None or empty")
        
        if 'events' not in partial_lifecycle:
            raise ValueError("partial_lifecycle missing 'events' field")
        
        if partial_lifecycle['events'] is None:
            raise ValueError("partial_lifecycle['events'] is None")
        
        if not isinstance(partial_lifecycle['events'], list):
            raise ValueError(f"partial_lifecycle['events'] must be a list, got {type(partial_lifecycle['events'])}")
        
        if len(partial_lifecycle['events']) == 0:
            raise ValueError("partial_lifecycle['events'] is empty")
        
        try:
            # Preprocess using the loaded preprocessor
            processed = self.preprocessor.process_lifecycle(
                partial_lifecycle,
                return_labels=False
            )
        except Exception as e:
            raise ValueError(f"Failed to preprocess lifecycle: {str(e)}")
        
        # Convert to PyG Data
        from torch_geometric.data import Data
        
        x = torch.tensor(processed['node_features'], dtype=torch.float)
        edge_index = torch.tensor(processed['edge_index'], dtype=torch.long)
        
        # Handle edge features (should be constant 0 now)
        if processed['edge_features'] is not None:
            edge_attr = torch.tensor(processed['edge_features'], dtype=torch.float)
        else:
            edge_attr = None
        
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
            
            # Inverse transform to get transit time in hours
            transit_hours = self.preprocessor.inverse_transform_time(
                last_pred.reshape(1, -1)
            )[0, 0]
        
        # ✅ FIX: Convert numpy.float32 to Python float
        transit_hours = float(transit_hours)
        
        # Get the last event time
        last_event = partial_lifecycle['events'][-1]
        last_event_time = self._parse_datetime(last_event['event_time'])
        
        if last_event_time is None:
            raise ValueError("Invalid last event time")
        
        # Predicted next event time = last event time + predicted transit time
        predicted_next_event_time = last_event_time + timedelta(hours=transit_hours)
        
        result = {
            'predicted_transit_hours': transit_hours,  # Already converted to float
            'last_event_time': last_event_time,
            'predicted_next_event_time': predicted_next_event_time
        }
        
        return result
    
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
        # Validate input
        if not self._validate_package_data(package_lifecycle):
            return {
                'package_id': package_lifecycle.get('package_id', 'unknown'),
                'error': 'Invalid package data structure'
            }
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(package_lifecycle)
        
        # ✅ Apply event sorting
        try:
            result['events'] = self._sort_events_by_sort_center(result['events'])
        except ValueError as e:
            result['error'] = f'Failed to sort events: {str(e)}'
            return result
        
        events = result['events']
        package_id = result['package_id']
        
        # Need at least 2 events to make predictions
        if len(events) < 2:
            result['error'] = f'Need at least 2 events, found {len(events)}'
            return result
        
        # Start from second event (index 1) and add predictions
        for i in range(1, len(events)):
            current_event = events[i]
            
            # Create partial lifecycle up to current event (not including it)
            partial_lifecycle = {
                'package_id': package_id,
                'source_postal': package_lifecycle.get('source_postal', ''),
                'dest_postal': package_lifecycle.get('dest_postal', ''),
                'pdd': package_lifecycle.get('pdd', ''),
                'weight': package_lifecycle.get('weight', 0),
                'length': package_lifecycle.get('length', 0),
                'width': package_lifecycle.get('width', 0),
                'height': package_lifecycle.get('height', 0),
                'events': events[:i]  # Events up to (not including) current event
            }
            
            # Validate partial lifecycle
            if not partial_lifecycle['events']:
                print(f"    ✗ No events in partial lifecycle for event {i}")
                events[i]['prediction'] = {
                    'error': 'No previous events'
                }
                continue
            
            # Predict transit time for current event
            try:
                pred_result = self.predict_single_step(partial_lifecycle)
                predicted_transit_hours = float(pred_result['predicted_transit_hours'])
                predicted_event_time = pred_result['predicted_next_event_time']
                
                # Get times
                previous_event_time = events[i-1]['event_time']
                actual_event_time = current_event['event_time']
                
                # Parse datetimes if they're strings
                if isinstance(previous_event_time, str):
                    previous_event_time = self._parse_datetime(previous_event_time)
                if isinstance(actual_event_time, str):
                    actual_event_time = self._parse_datetime(actual_event_time)
                
                if previous_event_time is None or actual_event_time is None:
                    raise ValueError("Invalid event times")
                
                # Actual transit time
                actual_transit_hours = (actual_event_time - previous_event_time).total_seconds() / 3600
                
                # Transit time prediction error
                transit_error_hours = predicted_transit_hours - actual_transit_hours
                absolute_transit_error_hours = abs(transit_error_hours)
                
                # Get planned time for the current event
                current_plan_time = self._parse_datetime(
                    self.preprocessor._get_plan_time_for_event(current_event, events[i-1])
                )
                
                # ✅ Build prediction output with plan_time
                prediction_dict = {
                    'predicted_event_time': predicted_event_time.isoformat(),
                    'predicted_transit_hours': round(predicted_transit_hours, 2),
                    'actual_transit_hours': round(actual_transit_hours, 2),
                    'transit_error_hours': round(transit_error_hours, 2),
                    'absolute_transit_error_hours': round(absolute_transit_error_hours, 2)
                }
                
                # Add plan_time if available
                if current_plan_time is not None:
                    prediction_dict['plan_time'] = current_plan_time.isoformat()
                
                events[i]['prediction'] = prediction_dict
                    
            except Exception as e:
                print(f"    ✗ Failed to predict event {i}: {e}")
                import traceback
                traceback.print_exc()
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
        # Fetch from Neptune (includes sorting and validation)
        package_data = self.fetch_package_from_neptune(package_id)
        
        if package_data is None:
            return {
                'package_id': package_id,
                'error': 'Package not found in Neptune or invalid data'
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
                    
                    # Calculate average errors
                    if successful:
                        avg_transit_error = np.mean([e['prediction']['absolute_transit_error_hours'] for e in successful])
                        
                        print(f"  ✓ Predicted {len(successful)}/{len(predicted_events)} events")
                        print(f"    Avg transit error: {avg_transit_error:.2f} hours")
                    else:
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
                    
                    # Calculate average errors
                    if successful:
                        avg_transit_error = np.mean([e['prediction']['absolute_transit_error_hours'] for e in successful])
                        
                        print(f"  ✓ Predicted {len(successful)}/{len(predicted_events)} events")
                        print(f"    Avg transit error: {avg_transit_error:.2f} hours")
                    else:
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
            all_transit_errors = []
            
            for r in successful_packages:
                predicted_events = [e for e in r['events'] if 'prediction' in e]
                total_predictions += len(predicted_events)
                successful = [e for e in predicted_events if 'error' not in e['prediction']]
                successful_predictions += len(successful)
                
                # Collect errors
                for e in successful:
                    all_transit_errors.append(e['prediction']['absolute_transit_error_hours'])
            
            print(f"\nPrediction Statistics:")
            print(f"  Total event predictions: {total_predictions}")
            print(f"  Successful predictions: {successful_predictions}")
            print(f"  Failed predictions: {total_predictions - successful_predictions}")
            if total_predictions > 0:
                print(f"  Success rate: {successful_predictions/total_predictions*100:.1f}%")
            
            if all_transit_errors:
                print(f"\nTransit Time Error Metrics:")
                print(f"  MAE (hours): {np.mean(all_transit_errors):.2f}")
                print(f"  Median AE (hours): {np.median(all_transit_errors):.2f}")
                print(f"  P25 (hours): {np.percentile(all_transit_errors, 25):.2f}")
                print(f"  P75 (hours): {np.percentile(all_transit_errors, 75):.2f}")
                print(f"  P90 (hours): {np.percentile(all_transit_errors, 90):.2f}")
                print(f"  P95 (hours): {np.percentile(all_transit_errors, 95):.2f}")
                print(f"  Min error (hours): {np.min(all_transit_errors):.2f}")
                print(f"  Max error (hours): {np.max(all_transit_errors):.2f}")
        
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
       "TBA325136379051",
        "TBA325136522364",
        "TBA325136422852",
        "TBA325136522895",
        "TBA325136430244"
    ]
    # ============================================================================
    
    print("\n" + "="*80)
    print("EVENT TIME PREDICTION - BATCH INFERENCE (TRANSIT TIME)")
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


if __name__ == '__main__':
    main()
# inference.py - Neptune-based inference with AE metrics

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm

from config import ModelConfig
from models.event_predictor import EventTimePredictor
from data.data_preprocessor import PackageLifecyclePreprocessor
from data.neptune_extractor import (
    NeptuneDataExtractor,
    _extract_package_edges,
    _validate_package_sequence,
    _deduplicate_events
)
from data.dataset import PackageLifecycleDataset
from torch_geometric.data import Batch


# ============================================================================
# CONFIGURATION - Modify these values as needed
# ============================================================================

PACKAGE_IDS = [
    "TBA325642724908",
    "TBA325643698747",
    "TBA325482265502",
    "TBA325586913247",
    "TBA325378303118",
    'TBA325674859283',
    "TBA325691051481",
    "TBA325638294740"
]

CHECKPOINT_PATH = "checkpoints/20251219_235042/best_model.pt"
PREPROCESSOR_PATH = "checkpoints/20251219_235042/preprocessor.pkl"
NEPTUNE_ENDPOINT = "swa-shipgraph-neptune-instance-prod-us-east-1-read-replica.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182"
OUTPUT_PATH = "predictions.json"
DEVICE = "cuda"

# ============================================================================


class EventTimeInference:
    """Inference class for event time prediction using Neptune"""
    
    def __init__(
        self,
        checkpoint_path: str,
        preprocessor_path: str,
        neptune_endpoint: str,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        print(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = PackageLifecyclePreprocessor.load(preprocessor_path)
        
        print(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model_config = ModelConfig.from_dict(self.checkpoint['model_config'])
        self.vocab_sizes = self.checkpoint['vocab_sizes']
        
        print("Initializing model...")
        self.model = EventTimePredictor(self.model_config, self.vocab_sizes)
        self.model = self.model.to(self.device)
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from epoch {self.checkpoint.get('epoch', 'unknown') + 1}")
        metrics = self.checkpoint.get('metrics', {})
        if metrics:
            print(f"Checkpoint metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        
        print(f"Connecting to Neptune at {neptune_endpoint}")
        self.neptune_endpoint = neptune_endpoint
        self.extractor = NeptuneDataExtractor(
            endpoint=neptune_endpoint,
            max_workers=1
        )
        print("Neptune connection established")
    
    def process_package(self, package_id: str) -> Optional[Dict]:
        """Fetch and process a single package from Neptune."""
        try:
            package_data = _extract_package_edges(self.extractor.main_client, package_id)
            
            if package_data is None:
                return None
            
            is_valid, invalid_reason = _validate_package_sequence(package_data)
            if not is_valid:
                return None
            
            package_data = _deduplicate_events(package_data)
            return package_data
            
        except Exception as e:
            print(f"  Error fetching package {package_id}: {e}")
            return None
    
    def _convert_to_training_format(self, package_data: Dict) -> Dict:
        """
        Convert Neptune extractor format to training data format.
        
        The preprocessor expects events with:
        - sort_center: for INDUCT, EXIT, LINEHAUL events
        - delivery_station: for DELIVERY events
        - delivery_location: dict with 'id' for postal code
        - carrier_id, leg_type, ship_method
        - event_time, plan_time, cpt
        - dwelling_seconds, missort, problem
        """
        pkg_id = package_data.get('package_id')
        
        converted = {
            'package_id': pkg_id,
            'tracking_id': pkg_id,
            'source_postal': package_data.get('source_postal'),
            'dest_postal': package_data.get('dest_postal'),
            'pdd': package_data.get('pdd'),
            'weight': package_data.get('weight', 0),
            'length': package_data.get('length', 0),
            'width': package_data.get('width', 0),
            'height': package_data.get('height', 0),
            'events': []
        }
        
        for event in package_data.get('events', []):
            # Preserve the original event structure that preprocessor expects
            converted_event = {
                'event_type': event.get('event_type', 'UNKNOWN'),
                'event_time': event.get('event_time'),
                # Location fields - preprocessor extracts these directly
                'sort_center': event.get('sort_center'),
                'delivery_station': event.get('delivery_station'),
                'delivery_location': event.get('delivery_location'),  # Keep as dict with 'id'
                # Carrier and method - preprocessor expects 'carrier_id' not 'carrier'
                'carrier_id': event.get('carrier_id'),
                'leg_type': event.get('leg_type'),
                'ship_method': event.get('ship_method'),
                # Time fields
                'plan_time': event.get('plan_time'),
                'cpt': event.get('cpt'),
                # Other fields
                'dwelling_seconds': event.get('dwelling_seconds', 0),
                'missort': event.get('missort', False),
                'problem': event.get('problem'),
            }
            
            converted['events'].append(converted_event)
        
        return converted
    
    def _parse_event_time(self, event_time) -> Optional[datetime]:
        """Parse event time to datetime object."""
        if event_time is None:
            return None
        
        if isinstance(event_time, datetime):
            return event_time
        
        if isinstance(event_time, str):
            # Try multiple formats
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(event_time, fmt)
                except ValueError:
                    continue
            
            # Try with timezone
            try:
                from dateutil import parser
                return parser.parse(event_time)
            except:
                pass
        
        return None
    
    def _calculate_actual_time_diffs(self, events: List[Dict]) -> List[float]:
        """Calculate actual time differences between consecutive events in hours."""
        time_diffs = []
        
        for i in range(1, len(events)):
            prev_time = self._parse_event_time(events[i-1].get('event_time'))
            curr_time = self._parse_event_time(events[i].get('event_time'))
            
            if prev_time and curr_time:
                diff_seconds = (curr_time - prev_time).total_seconds()
                diff_hours = diff_seconds / 3600.0
                time_diffs.append(diff_hours)
            else:
                time_diffs.append(None)
        
        return time_diffs
    
    def _get_event_location(self, event: Dict) -> str:
        """Get the display location for an event."""
        event_type = event.get('event_type', '')
        
        if event_type == 'DELIVERY':
            # For DELIVERY, show delivery_station
            station = event.get('delivery_station')
            if station:
                return str(station)
            # Fallback to postal code
            delivery_loc = event.get('delivery_location')
            if delivery_loc and isinstance(delivery_loc, dict):
                return delivery_loc.get('id', 'DELIVERY')
            return 'DELIVERY'
        else:
            # For sort center events
            sort_center = event.get('sort_center')
            if sort_center:
                return str(sort_center)
            return 'UNKNOWN'
    
    @torch.no_grad()
    def predict_single(self, package_id: str) -> Dict:
        """Predict event times for a single package."""
        package_data = self.process_package(package_id)
        
        if package_data is None:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'Package not found or invalid'
            }
        
        events = package_data.get('events', [])
        
        if len(events) < 2:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': f'Insufficient events: {len(events)}'
            }
        
        try:
            converted_data = self._convert_to_training_format(package_data)
            
            if not converted_data.get('package_id'):
                return {
                    'package_id': package_id,
                    'status': 'error',
                    'error': 'Missing package_id'
                }
            
            # Calculate actual time differences
            actual_time_diffs = self._calculate_actual_time_diffs(events)
            
            df = pd.DataFrame([converted_data])
            
            dataset = PackageLifecycleDataset(
                df,
                self.preprocessor,
                return_labels=False
            )
            
            graph_data = dataset[0]
            graph_data = graph_data.to(self.device)
            
            batch = Batch.from_data_list([graph_data])
            
            predictions = self.model(batch)
            
            preds_scaled = predictions.cpu().numpy()
            preds_hours = self.preprocessor.inverse_transform_time(preds_scaled).flatten()
            
            # Build event details with predictions, actuals, and AE
            event_predictions = []
            all_ae = []
            
            for i, event in enumerate(events):
                event_info = {
                    'event_idx': i,
                    'event_type': event.get('event_type'),
                    'location': self._get_event_location(event),
                }
                
                # Add prediction and actual for events after the first
                if i > 0:
                    pred_idx = i - 1
                    
                    # Predicted hours
                    if pred_idx < len(preds_hours):
                        pred_hours = float(preds_hours[pred_idx])
                        event_info['predicted_hours'] = pred_hours
                    else:
                        pred_hours = None
                        event_info['predicted_hours'] = None
                    
                    # Actual hours
                    if pred_idx < len(actual_time_diffs) and actual_time_diffs[pred_idx] is not None:
                        actual_hours = float(actual_time_diffs[pred_idx])
                        event_info['actual_hours'] = actual_hours
                    else:
                        actual_hours = None
                        event_info['actual_hours'] = None
                    
                    # Absolute Error
                    if pred_hours is not None and actual_hours is not None:
                        ae = abs(pred_hours - actual_hours)
                        event_info['ae'] = ae
                        all_ae.append(ae)
                    else:
                        event_info['ae'] = None
                
                event_predictions.append(event_info)
            
            # Calculate summary metrics
            valid_preds = [e['predicted_hours'] for e in event_predictions if e.get('predicted_hours') is not None]
            valid_actuals = [e['actual_hours'] for e in event_predictions if e.get('actual_hours') is not None]
            
            result = {
                'package_id': package_id,
                'status': 'success',
                'num_events': len(events),
                'source_postal': package_data.get('source_postal'),
                'dest_postal': package_data.get('dest_postal'),
                'events': event_predictions,
                'metrics': {
                    'mae': float(np.mean(all_ae)) if all_ae else None,
                    'max_ae': float(np.max(all_ae)) if all_ae else None,
                    'min_ae': float(np.min(all_ae)) if all_ae else None,
                    'total_predicted_hours': float(np.sum(valid_preds)) if valid_preds else None,
                    'total_actual_hours': float(np.sum(valid_actuals)) if valid_actuals else None,
                }
            }
            
            # Add total time error
            if result['metrics']['total_predicted_hours'] is not None and result['metrics']['total_actual_hours'] is not None:
                result['metrics']['total_time_ae'] = abs(
                    result['metrics']['total_predicted_hours'] - result['metrics']['total_actual_hours']
                )
            
            return result
            
        except Exception as e:
            import traceback
            return {
                'package_id': package_id,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    @torch.no_grad()
    def predict_batch(self, package_ids: List[str]) -> List[Dict]:
        """Predict event times for multiple packages."""
        results = []
        
        for pkg_id in tqdm(package_ids, desc="Running inference"):
            result = self.predict_single(pkg_id)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save prediction results to file"""
        ext = os.path.splitext(output_path)[1].lower()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if ext == '.csv':
            flat_results = []
            for r in results:
                flat = {
                    'package_id': r['package_id'],
                    'status': r['status'],
                    'error': r.get('error'),
                    'num_events': r.get('num_events'),
                    'source_postal': r.get('source_postal'),
                    'dest_postal': r.get('dest_postal'),
                }
                if 'metrics' in r and r['metrics']:
                    flat.update({
                        'mae': r['metrics'].get('mae'),
                        'max_ae': r['metrics'].get('max_ae'),
                        'total_predicted_hours': r['metrics'].get('total_predicted_hours'),
                        'total_actual_hours': r['metrics'].get('total_actual_hours'),
                        'total_time_ae': r['metrics'].get('total_time_ae'),
                    })
                flat_results.append(flat)
            pd.DataFrame(flat_results).to_csv(output_path, index=False)
        else:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
    
    def close(self):
        """Clean up resources"""
        if self.extractor:
            self.extractor.close()
            print("Neptune connection closed")


def print_summary(results: List[Dict]):
    """Print summary statistics"""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nPackages: {len(results)} total | {len(successful)} success | {len(failed)} failed")
    
    if failed:
        print(f"\nFailed Packages:")
        for r in failed:
            print(f"  ✗ {r['package_id']}: {r.get('error', 'Unknown')}")
    
    if successful:
        # Collect all metrics
        all_mae = [r['metrics']['mae'] for r in successful if r['metrics'].get('mae') is not None]
        all_total_ae = [r['metrics']['total_time_ae'] for r in successful if r['metrics'].get('total_time_ae') is not None]
        all_total_pred = [r['metrics']['total_predicted_hours'] for r in successful if r['metrics'].get('total_predicted_hours') is not None]
        all_total_actual = [r['metrics']['total_actual_hours'] for r in successful if r['metrics'].get('total_actual_hours') is not None]
        
        print(f"\n{'─' * 80}")
        print("AGGREGATE METRICS")
        print(f"{'─' * 80}")
        
        if all_mae:
            print(f"\nPer-Event Absolute Error (hours):")
            print(f"  Mean MAE:   {np.mean(all_mae):.2f}")
            print(f"  Std MAE:    {np.std(all_mae):.2f}")
            print(f"  Min MAE:    {np.min(all_mae):.2f}")
            print(f"  Max MAE:    {np.max(all_mae):.2f}")
        
        if all_total_ae:
            print(f"\nTotal Journey Time Error (hours):")
            print(f"  Mean:       {np.mean(all_total_ae):.2f}")
            print(f"  Std:        {np.std(all_total_ae):.2f}")
            print(f"  Min:        {np.min(all_total_ae):.2f}")
            print(f"  Max:        {np.max(all_total_ae):.2f}")
        
        if all_total_pred and all_total_actual:
            print(f"\nTotal Journey Time (hours):")
            print(f"  Avg Predicted: {np.mean(all_total_pred):.2f}")
            print(f"  Avg Actual:    {np.mean(all_total_actual):.2f}")
        
        # Per-package summary table
        print(f"\n{'─' * 80}")
        print("PER-PACKAGE SUMMARY")
        print(f"{'─' * 80}")
        print(f"{'Package ID':<25} {'Events':<8} {'MAE(h)':<10} {'Total Pred':<12} {'Total Actual':<12} {'Total AE':<10}")
        print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
        
        for r in successful:
            m = r['metrics']
            mae_str = f"{m['mae']:.2f}" if m.get('mae') is not None else "N/A"
            pred_str = f"{m['total_predicted_hours']:.2f}" if m.get('total_predicted_hours') is not None else "N/A"
            actual_str = f"{m['total_actual_hours']:.2f}" if m.get('total_actual_hours') is not None else "N/A"
            ae_str = f"{m['total_time_ae']:.2f}" if m.get('total_time_ae') is not None else "N/A"
            
            print(f"{r['package_id']:<25} {r['num_events']:<8} {mae_str:<10} {pred_str:<12} {actual_str:<12} {ae_str:<10}")
        
        # Event-level details
        print(f"\n{'─' * 80}")
        print("EVENT-LEVEL DETAILS")
        print(f"{'─' * 80}")
        
        for r in successful:
            print(f"\n{r['package_id']} ({r['num_events']} events):")
            print(f"  {'#':<3} {'Type':<12} {'Location':<12} {'Pred(h)':<10} {'Actual(h)':<10} {'AE(h)':<10}")
            print(f"  {'-'*3} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
            
            for event in r['events']:
                pred_str = f"{event['predicted_hours']:.2f}" if event.get('predicted_hours') is not None else "-"
                actual_str = f"{event['actual_hours']:.2f}" if event.get('actual_hours') is not None else "-"
                ae_str = f"{event['ae']:.2f}" if event.get('ae') is not None else "-"
                
                loc = str(event.get('location', 'N/A'))[:12]
                
                print(f"  {event['event_idx']:<3} {event['event_type']:<12} {loc:<12} {pred_str:<10} {actual_str:<10} {ae_str:<10}")


def main():
    """Main entry point"""
    
    print("=" * 80)
    print("EVENT TIME PREDICTION - NEPTUNE INFERENCE")
    print("=" * 80)
    print(f"\nPackages to process: {len(PACKAGE_IDS)}")
    for pkg_id in PACKAGE_IDS:
        print(f"  - {pkg_id}")
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
    print("=" * 80)
    
    try:
        inference = EventTimeInference(
            checkpoint_path=CHECKPOINT_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
            neptune_endpoint=NEPTUNE_ENDPOINT,
            device=DEVICE
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    try:
        results = inference.predict_batch(PACKAGE_IDS)
        print_summary(results)
        
        if OUTPUT_PATH:
            inference.save_results(results, OUTPUT_PATH)
        
        print("\n" + "=" * 80)
        print("✓ INFERENCE COMPLETE")
        print("=" * 80)
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        inference.close()


if __name__ == '__main__':
    exit(main())
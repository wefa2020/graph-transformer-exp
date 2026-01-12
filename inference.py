#!/usr/bin/env python3
"""
inference.py - Neptune-based inference with leg_plan skeleton and iterative rolling predictions
Uses predicted times (not plan times) to continue the prediction chain
Includes plan_time for each event

Updated for Time2Vec preprocessor format with separate time/other features.
Aligned with CausalH5BatchCollator output format.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import gc
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm

from torch_geometric.data import Data, Batch

from config import Config
from models.event_predictor import EventTimePredictor
from data.data_preprocessor import PackageLifecyclePreprocessor
from data.neptune_extractor import (
    NeptuneDataExtractor,
    _extract_package_edges_optimized,
    _validate_package_sequence,
    _deduplicate_events
)


# ============================================================================
# CONFIGURATION
# ============================================================================

PACKAGE_IDS = [
    "TBA325968945926",
    "TBA327582930610",
    "TBA326599241029",
    "TBA326755394001",
    "TBA326560223900",
    "TBA326630194468",
    "TBA326639885895",
    "TBA327231975747",
    "TBA327484808975",
    "TBA327399631329",
    "TBA327508421635",
    "TBA327514698633",
    "TBA327585203057",
]

CHECKPOINT_PATH = "checkpoints/131/best_model.pt"
PREPROCESSOR_PATH = "checkpoints/131/preprocessor.pkl"
NEPTUNE_ENDPOINT = "swa-shipgraph-neptune-instance-prod-us-east-1-read-replica.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182"
OUTPUT_PATH = "predictions.json"
DEVICE = "cuda"

STRICT_LEG_PLAN_VALIDATION = False
ALLOW_UNDELIVERED = True


# ============================================================================
# TENSOR UTILITIES
# ============================================================================

def to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert numpy array to tensor with proper dtype."""
    if dtype == torch.long:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
    elif dtype == torch.float32:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    elif dtype == torch.bool:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.bool_))
    return torch.from_numpy(arr)


# ============================================================================
# INFERENCE CLASS
# ============================================================================

class EventTimeInference:
    """Inference class with iterative rolling predictions."""
    
    # Node categorical fields (matches CausalH5BatchCollator)
    _NODE_CAT_FIELDS = [
        'event_type', 'location', 'postal', 'region',
        'carrier', 'leg_type', 'ship_method'
    ]
    
    def __init__(
        self,
        checkpoint_path: str,
        preprocessor_path: str,
        neptune_endpoint: str,
        device: str = 'cuda',
        strict_validation: bool = False,
        allow_undelivered: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.strict_validation = strict_validation
        self.allow_undelivered = allow_undelivered
        
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load preprocessor
        print(f"Loading preprocessor from {preprocessor_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        self.preprocessor = PackageLifecyclePreprocessor.load(preprocessor_path)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract vocab_sizes and feature_dims from checkpoint
        self.vocab_sizes = self.checkpoint['vocab_sizes']
        self.feature_dims = self.checkpoint['feature_dims']
        
        # Load full config from checkpoint
        full_config = Config.from_dict(self.checkpoint['config'])
        self.model_config = full_config.model  # Store for reference
        
        print("Initializing model...")
        self.model = EventTimePredictor.from_config(
            config=full_config,
            vocab_sizes=self.vocab_sizes,
            feature_dims=self.feature_dims,
            device=self.device,
        )
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
        
        # Connect to Neptune
        print(f"Connecting to Neptune at {neptune_endpoint}")
        self.neptune_endpoint = neptune_endpoint
        self.extractor = NeptuneDataExtractor(
            endpoint=neptune_endpoint,
            max_workers=1
        )
        print("Neptune connection established")
        print(f"Strict leg_plan validation: {self.strict_validation}")
        print(f"Allow undelivered packages: {self.allow_undelivered}")
    
    def fetch_package(self, package_id: str) -> Optional[Dict]:
        """Fetch package from Neptune."""
        try:
            package_data = _extract_package_edges_optimized(self.extractor.main_client, package_id)
            
            if package_data is None:
                return None
            
            if not self.allow_undelivered:
                is_valid, invalid_reason = _validate_package_sequence(package_data)
                if not is_valid:
                    print(f"  Package {package_id} invalid: {invalid_reason}")
                    return None
            
            package_data = _deduplicate_events(package_data)
            return package_data
            
        except Exception as e:
            print(f"  Error fetching package {package_id}: {e}")
            return None
    
    def _is_origin_id(self, location_id: str) -> bool:
        return len(location_id) > 20 and location_id.isalnum()
    
    def _is_postal_code(self, location_id: str) -> bool:
        return bool(re.match(r'^\d{5}$', location_id))
    
    def _is_sort_center(self, location_id: str) -> bool:
        return bool(re.match(r'^[A-Z0-9]{3,4}$', location_id))
    
    def _parse_leg_plan(self, leg_plan_str: str) -> Optional[Dict]:
        if not leg_plan_str:
            return None
        try:
            leg_plan = json.loads(leg_plan_str)
            if not isinstance(leg_plan, dict) or len(leg_plan) == 0:
                return None
            return leg_plan
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            print(f"    Warning: Failed to parse leg_plan: {e}")
            return None
    
    def _create_skeleton_from_leg_plan(self, leg_plan: Dict, dest_postal: str) -> List[Dict]:
        """Create event skeleton from leg_plan."""
        skeleton = []
        location_ids = list(leg_plan.keys())
        
        origin_id = None
        sort_centers = []
        dest_postal_entry = None
        
        for loc_id in location_ids:
            if self._is_origin_id(loc_id):
                origin_id = loc_id
            elif self._is_postal_code(loc_id):
                dest_postal_entry = (loc_id, leg_plan[loc_id])
            elif self._is_sort_center(loc_id):
                sort_centers.append((loc_id, leg_plan[loc_id]))
        
        if not sort_centers:
            return []
        
        event_idx = 0
        
        for i, (sc_id, sc_data) in enumerate(sort_centers):
            is_first_sc = (i == 0)
            
            if is_first_sc:
                # INDUCT - use plan_time and cpt from leg_plan
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'INDUCT',
                    'location': sc_id,
                    'plan_time': sc_data.get('plan_time'),
                    'cpt': sc_data.get('cpt'),
                    'ship_method': sc_data.get('ship_method'),
                    'is_first_event': True,
                })
                event_idx += 1
                
                # EXIT - plan_time is previous event's CPT
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'EXIT',
                    'location': sc_id,
                    'plan_time': sc_data.get('cpt'),  # EXIT plan_time = previous CPT
                    'cpt': None,
                    'ship_method': None,
                    'is_first_event': False,
                })
                event_idx += 1
            else:
                # LINEHAUL - use plan_time and cpt from leg_plan
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'LINEHAUL',
                    'location': sc_id,
                    'plan_time': sc_data.get('plan_time'),
                    'cpt': sc_data.get('cpt'),
                    'ship_method': sc_data.get('ship_method'),
                    'is_first_event': False,
                })
                event_idx += 1
                
                # EXIT - plan_time is previous event's CPT
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'EXIT',
                    'location': sc_id,
                    'plan_time': sc_data.get('cpt'),  # EXIT plan_time = previous CPT
                    'cpt': None,
                    'ship_method': None,
                    'is_first_event': False,
                })
                event_idx += 1
        
        # DELIVERY event
        if sort_centers:
            last_sc_id = sort_centers[-1][0]
            delivery_plan_time = None
            delivery_ship_method = None
            
            if dest_postal_entry:
                delivery_plan_time = dest_postal_entry[1].get('plan_time')
                delivery_ship_method = dest_postal_entry[1].get('ship_method')
            
            skeleton.append({
                'event_idx': event_idx,
                'event_type': 'DELIVERY',
                'location': last_sc_id,
                'plan_time': delivery_plan_time,
                'cpt': None,
                'ship_method': delivery_ship_method,
                'is_first_event': False,
            })
        
        return skeleton
    
    def _get_event_location(self, event: Dict) -> str:
        event_type = event.get('event_type', '')
        
        if event_type == 'DELIVERY':
            station = event.get('delivery_station')
            if station:
                return str(station)
            delivery_loc = event.get('delivery_location')
            if delivery_loc and isinstance(delivery_loc, dict):
                return delivery_loc.get('id', '')
            return ''
        else:
            sort_center = event.get('sort_center')
            if sort_center:
                return str(sort_center)
            return ''
    
    def _determine_delivery_status(self, filled_skeleton: List[Dict]) -> str:
        if not filled_skeleton:
            return 'UNKNOWN'
        
        delivery_event = filled_skeleton[-1] if filled_skeleton else None
        if delivery_event and delivery_event.get('event_type') == 'DELIVERY':
            if delivery_event.get('neptune_matched', False):
                return 'DELIVERED'
        
        matched_count = sum(1 for s in filled_skeleton if s.get('neptune_matched', False))
        
        if matched_count == 0:
            return 'NOT_STARTED'
        elif matched_count == len(filled_skeleton):
            return 'DELIVERED'
        else:
            return 'IN_TRANSIT'
    
    def _get_last_known_location(self, filled_skeleton: List[Dict]) -> Optional[Dict]:
        for skel in reversed(filled_skeleton):
            if skel.get('neptune_matched', False):
                return {
                    'event_idx': skel['event_idx'],
                    'event_type': skel['event_type'],
                    'location': skel['location'],
                    'event_time': skel.get('event_time')
                }
        return None
    
    def _get_plan_time_for_event(
        self, 
        event_idx: int, 
        event_type: str,
        skeleton: List[Dict],
        neptune_event: Optional[Dict]
    ) -> Optional[str]:
        """
        Get plan_time for an event:
        - For actual events: use Neptune event's plan_time if available
        - For EXIT events: use previous event's CPT
        - Otherwise: use skeleton's plan_time
        """
        # For actual events, try Neptune's plan_time first
        if neptune_event:
            neptune_plan_time = neptune_event.get('plan_time')
            if neptune_plan_time:
                return neptune_plan_time
        
        # For EXIT events, use previous event's CPT
        if event_type == 'EXIT' and event_idx > 0:
            prev_skel = skeleton[event_idx - 1]
            prev_cpt = prev_skel.get('cpt')
            if prev_cpt:
                return prev_cpt
            # Also check context for CPT
            prev_context = prev_skel.get('context', {})
            if prev_context.get('cpt'):
                return prev_context.get('cpt')
        
        # Fall back to skeleton's plan_time
        skel = skeleton[event_idx]
        return skel.get('plan_time')
    
    def _match_neptune_events_to_skeleton(
        self, 
        skeleton: List[Dict], 
        neptune_events: List[Dict]
    ) -> Tuple[List[Dict], List[str], Dict[int, Dict]]:
        """
        Match Neptune events to skeleton.
        Returns: (filled_skeleton, errors, matched_neptune_events_by_idx)
        """
        errors = []
        
        filled_skeleton = []
        for skel in skeleton:
            filled_skeleton.append(dict(skel))
        
        neptune_index = {}
        for i, event in enumerate(neptune_events):
            loc = self._get_event_location(event)
            event_type = event.get('event_type', '')
            key = (loc, event_type)
            if key not in neptune_index:
                neptune_index[key] = []
            neptune_index[key].append((i, event))
        
        matched_neptune_indices = set()
        matched_neptune_events_by_skel_idx = {}  # skeleton_idx -> neptune_event
        
        for skel_idx, skel in enumerate(filled_skeleton):
            skel_loc = skel['location']
            skel_type = skel['event_type']
            key = (skel_loc, skel_type)
            
            neptune_matches = neptune_index.get(key, [])
            
            matched_event = None
            for idx, event in neptune_matches:
                if idx not in matched_neptune_indices:
                    matched_event = event
                    matched_neptune_indices.add(idx)
                    matched_neptune_events_by_skel_idx[skel_idx] = event
                    break
            
            if matched_event:
                skel['event_time'] = matched_event.get('event_time')
                skel['neptune_matched'] = True
                skel['is_predicted'] = False
                
                # Get plan_time for this event
                plan_time = self._get_plan_time_for_event(
                    skel_idx, skel_type, filled_skeleton, matched_event
                )
                
                context = {'has_problem': False}
                
                # Problem handling - works with both old (EXIT) and new (INDUCT/LINEHAUL) formats
                problem = matched_event.get('problem')
                if problem:
                    context['problem'] = problem
                    context['has_problem'] = True
                
                missort = matched_event.get('missort')
                if missort is not None:
                    context['missort'] = bool(missort)
                
                dwelling_seconds = matched_event.get('dwelling_seconds')
                if dwelling_seconds and dwelling_seconds > 0:
                    context['dwelling_seconds'] = float(dwelling_seconds)
                    context['dwelling_hours'] = round(dwelling_seconds / 3600.0, 2)
                
                carrier = matched_event.get('carrier_id')
                if carrier:
                    context['carrier'] = carrier
                
                leg_type = matched_event.get('leg_type')
                if leg_type:
                    context['leg_type'] = leg_type
                
                ship_method = matched_event.get('ship_method') or skel.get('ship_method')
                if ship_method:
                    context['ship_method'] = ship_method
                
                sort_center = matched_event.get('sort_center')
                if sort_center:
                    context['sort_center'] = sort_center
                
                delivery_station = matched_event.get('delivery_station')
                if delivery_station:
                    context['delivery_station'] = delivery_station
                
                # Add CPT from Neptune event or skeleton
                cpt = matched_event.get('cpt') or skel.get('cpt')
                if cpt:
                    context['cpt'] = cpt
                    skel['cpt'] = cpt  # Store for EXIT event reference
                
                # Store plan_time
                skel['plan_time'] = plan_time
                
                skel['context'] = context
            else:
                skel['event_time'] = None
                skel['neptune_matched'] = False
                skel['is_predicted'] = True
                
                # Get plan_time for predicted event
                plan_time = self._get_plan_time_for_event(
                    skel_idx, skel_type, filled_skeleton, None
                )
                skel['plan_time'] = plan_time
                
                context = {'has_problem': False}
                if skel.get('ship_method'):
                    context['ship_method'] = skel['ship_method']
                if skel.get('cpt'):
                    context['cpt'] = skel['cpt']
                context['sort_center'] = skel['location']
                
                skel['context'] = context
        
        unmatched_count = len(neptune_events) - len(matched_neptune_indices)
        if unmatched_count > 0:
            unmatched_events = [
                f"{neptune_events[i].get('event_type')} at {self._get_event_location(neptune_events[i])}"
                for i in range(len(neptune_events)) if i not in matched_neptune_indices
            ]
            errors.append(f"Unmatched Neptune events ({unmatched_count}): {unmatched_events[:3]}")
        
        return filled_skeleton, errors, matched_neptune_events_by_skel_idx
    
    def _build_synthetic_package(
        self, 
        package_data: Dict, 
        events_with_times: List[Dict]
    ) -> Dict:
        """Build a synthetic package for inference."""
        synthetic = {
            'package_id': package_data.get('package_id'),
            'tracking_id': package_data.get('package_id'),
            'source_postal': package_data.get('source_postal'),
            'dest_postal': package_data.get('dest_postal'),
            'pdd': package_data.get('pdd'),
            'weight': package_data.get('weight', 0),
            'length': package_data.get('length', 0),
            'width': package_data.get('width', 0),
            'height': package_data.get('height', 0),
            'events': []
        }
        
        for evt in events_with_times:
            context = evt.get('context', {})
            event_type = evt.get('event_type')
            location = evt.get('location')
            
            event = {
                'event_type': event_type,
                'event_time': evt.get('event_time'),
                'sort_center': location if event_type != 'DELIVERY' else None,
                'delivery_station': location if event_type == 'DELIVERY' else None,
                'delivery_location': {'id': package_data.get('dest_postal')} if event_type == 'DELIVERY' else None,
                'carrier_id': context.get('carrier', 'AMZN_US'),
                'leg_type': context.get('leg_type', 'FORWARD'),
                'ship_method': context.get('ship_method') or evt.get('ship_method'),
                'plan_time': evt.get('plan_time'),
                'cpt': evt.get('cpt') or context.get('cpt'),
                'dwelling_seconds': context.get('dwelling_seconds', 0),
                'missort': context.get('missort', False),
                'problem': context.get('problem'),
            }
            synthetic['events'].append(event)
        
        return synthetic
    
    def _features_to_pyg_data(self, features: Dict) -> Data:
        """
        Convert preprocessor features dict to PyG Data object.
        
        Aligned with CausalH5BatchCollator output format for consistency.
        """
        node_cat = features['node_categorical_indices']
        pkg_cat = features['package_categorical']
        
        # Handle package_features shape - collator outputs [1, package_dim]
        pkg_feat = features['package_features']
        if pkg_feat.ndim == 1:
            pkg_feat = pkg_feat.reshape(1, -1)
        
        data = Data(
            # Time features (for Time2Vec)
            node_observable_time=to_tensor(features['node_observable_time'], torch.float32),
            node_observable_other=to_tensor(features['node_observable_other'], torch.float32),
            node_realized_time=to_tensor(features['node_realized_time'], torch.float32),
            node_realized_other=to_tensor(features['node_realized_other'], torch.float32),
            
            # Categorical indices
            event_type_idx=to_tensor(node_cat['event_type'], torch.long),
            location_idx=to_tensor(node_cat['location'], torch.long),
            postal_idx=to_tensor(node_cat['postal'], torch.long),
            region_idx=to_tensor(node_cat['region'], torch.long),
            carrier_idx=to_tensor(node_cat['carrier'], torch.long),
            leg_type_idx=to_tensor(node_cat['leg_type'], torch.long),
            ship_method_idx=to_tensor(node_cat['ship_method'], torch.long),
            
            # Edge features
            edge_index=to_tensor(features['edge_index'], torch.long),
            edge_features=to_tensor(features['edge_features'], torch.float32),
            
            # Package features
            package_features=to_tensor(pkg_feat, torch.float32),
            source_postal_idx=torch.tensor([pkg_cat['source_postal']], dtype=torch.long),
            dest_postal_idx=torch.tensor([pkg_cat['dest_postal']], dtype=torch.long),
            
            num_nodes=features['num_nodes'],
        )
        
        # Add labels if present - use edge_labels to match collator
        if 'labels' in features:
            labels = features['labels']
            if labels.ndim > 1:
                labels = labels.flatten()
            data.edge_labels = to_tensor(labels, torch.float32)
        
        if 'labels_raw' in features:
            labels_raw = features['labels_raw']
            if labels_raw.ndim > 1:
                labels_raw = labels_raw.flatten()
            data.edge_labels_raw = to_tensor(labels_raw, torch.float32)
        
        return data
    
    def _parse_event_time(self, event_time) -> Optional[datetime]:
        if event_time is None:
            return None
        
        if isinstance(event_time, datetime):
            return event_time
        
        if isinstance(event_time, str):
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
            
            try:
                from dateutil import parser
                return parser.parse(event_time)
            except:
                pass
        
        return None
    
    def _format_event_time(self, event_time) -> Optional[str]:
        if event_time is None:
            return None
        if isinstance(event_time, str):
            dt = self._parse_event_time(event_time)
            if dt:
                return dt.isoformat()
            return event_time
        if isinstance(event_time, datetime):
            return event_time.isoformat()
        return None
    
    def _calculate_predicted_datetime(self, prev_event_time, predicted_hours: float) -> Optional[str]:
        prev_dt = self._parse_event_time(prev_event_time)
        if prev_dt is None or predicted_hours is None:
            return None
        
        predicted_dt = prev_dt + timedelta(hours=predicted_hours)
        return predicted_dt.isoformat()
    
    @torch.no_grad()
    def _run_single_inference(self, synthetic_package: Dict) -> Optional[np.ndarray]:
        """Run model inference on a synthetic package."""
        try:
            features = self.preprocessor.process_lifecycle(synthetic_package, return_labels=True)
            if features is None:
                return None
            
            # Check minimum events
            if features['num_nodes'] < 2:
                return None
            
            graph_data = self._features_to_pyg_data(features)
            graph_data = graph_data.to(self.device)
            
            # Create batch - matches CausalH5BatchCollator output
            batch = Batch.from_data_list([graph_data])
            batch.node_counts = torch.tensor([graph_data.num_nodes], dtype=torch.long, device=self.device)
            batch.edge_counts = torch.tensor([graph_data.edge_index.shape[1]], dtype=torch.long, device=self.device)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                predictions = self.model(batch)
            
            # Get all predictions (one per edge)
            preds = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
            preds_scaled = preds.float().cpu().numpy()
            preds_hours = self.preprocessor.inverse_transform_time(preds_scaled).flatten()
            
            return preds_hours
        except Exception as e:
            print(f"    Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @torch.no_grad()
    def _run_iterative_predictions(
        self, 
        package_data: Dict, 
        filled_skeleton: List[Dict]
    ) -> Tuple[List[Dict], Dict[int, float]]:
        """
        Run predictions iteratively using predicted times to continue the chain.
        """
        working_skeleton = [dict(s) for s in filled_skeleton]
        predictions_by_edge = {}
        
        # Find the last actual event index
        last_actual_idx = -1
        for i, skel in enumerate(working_skeleton):
            if skel.get('neptune_matched', False):
                last_actual_idx = i
        
        if last_actual_idx < 0:
            return working_skeleton, predictions_by_edge
        
        # First pass: get predictions for transitions between actual events
        if last_actual_idx >= 1:
            actual_events = [working_skeleton[i] for i in range(last_actual_idx + 1) 
                           if working_skeleton[i].get('neptune_matched', False)]
            
            if len(actual_events) >= 2:
                synthetic = self._build_synthetic_package(package_data, actual_events)
                preds = self._run_single_inference(synthetic)
                
                if preds is not None:
                    for i, pred_h in enumerate(preds):
                        predictions_by_edge[i] = float(pred_h)
        
        # Build list of events with times (actual events first)
        events_with_times = []
        for i in range(last_actual_idx + 1):
            if working_skeleton[i].get('neptune_matched', False):
                events_with_times.append(dict(working_skeleton[i]))
        
        # Iteratively predict each remaining event
        for target_idx in range(last_actual_idx + 1, len(working_skeleton)):
            target_skel = working_skeleton[target_idx]
            
            if len(events_with_times) < 1:
                break
            
            prev_event_time = events_with_times[-1].get('event_time')
            if not prev_event_time:
                break
            
            target_event_for_inference = dict(target_skel)
            target_event_for_inference['event_time'] = prev_event_time
            
            inference_events = events_with_times + [target_event_for_inference]
            synthetic = self._build_synthetic_package(package_data, inference_events)
            
            preds = self._run_single_inference(synthetic)
            
            if preds is not None and len(preds) > 0:
                pred_hours = float(preds[-1])
                edge_idx = target_idx - 1
                predictions_by_edge[edge_idx] = pred_hours
                
                predicted_dt = self._calculate_predicted_datetime(prev_event_time, pred_hours)
                
                working_skeleton[target_idx]['event_time'] = predicted_dt
                working_skeleton[target_idx]['predicted_time'] = predicted_dt
                
                updated_event = dict(target_skel)
                updated_event['event_time'] = predicted_dt
                events_with_times.append(updated_event)
            else:
                break
        
        return working_skeleton, predictions_by_edge
    
    def _build_output_events(
        self, 
        working_skeleton: List[Dict], 
        predictions_by_edge: Dict[int, float],
        original_skeleton: List[Dict]
    ) -> List[Dict]:
        """Build final output events with all predictions, metrics, and plan_time."""
        output_events = []
        
        for i, skel in enumerate(working_skeleton):
            is_predicted = skel.get('is_predicted', False)
            original_skel = original_skeleton[i]
            
            # Get actual event time (only for Neptune-matched events)
            actual_event_time = None
            if original_skel.get('neptune_matched', False):
                actual_event_time = original_skel.get('event_time')
            
            # Get plan_time
            plan_time = skel.get('plan_time')
            
            # Build event output with is_predicted first
            event_output = {
                'is_predicted': is_predicted,
                'event_idx': skel['event_idx'],
                'event_type': skel['event_type'],
                'location': skel['location'],
                'plan_time': self._format_event_time(plan_time),
                'event_time': self._format_event_time(skel.get('event_time')) if not is_predicted else None,
                'predicted_time': self._format_event_time(skel.get('predicted_time')) if is_predicted else None,
                'context': skel.get('context', {}),
            }
            
            # Add prediction info for events after the first
            if i > 0:
                edge_idx = i - 1
                pred_hours = predictions_by_edge.get(edge_idx)
                
                if pred_hours is not None:
                    event_output['predicted_hours'] = pred_hours
                    
                    # Calculate predicted_time for actual events too
                    if not is_predicted:
                        prev_time = working_skeleton[i-1].get('event_time')
                        if prev_time:
                            event_output['predicted_time'] = self._calculate_predicted_datetime(prev_time, pred_hours)
                    
                    # Calculate actual_hours and ae for actual events
                    if not is_predicted and actual_event_time:
                        prev_original = original_skeleton[i-1]
                        if prev_original.get('neptune_matched', False):
                            prev_actual_time = prev_original.get('event_time')
                            if prev_actual_time:
                                prev_dt = self._parse_event_time(prev_actual_time)
                                curr_dt = self._parse_event_time(actual_event_time)
                                if prev_dt and curr_dt:
                                    actual_hours = (curr_dt - prev_dt).total_seconds() / 3600.0
                                    event_output['actual_hours'] = actual_hours
                                    event_output['ae'] = abs(pred_hours - actual_hours)
            
            output_events.append(event_output)
        
        return output_events
    
    def _calculate_eta(self, output_events: List[Dict]) -> Optional[str]:
        if not output_events:
            return None
        
        last_event = output_events[-1]
        if last_event.get('event_type') == 'DELIVERY':
            # For predicted events, use predicted_time; for actual, use event_time
            if last_event.get('is_predicted'):
                return last_event.get('predicted_time')
            else:
                return last_event.get('event_time')
        
        return None
    
    @torch.no_grad()
    def predict_single(self, package_id: str, package_data: Optional[Dict] = None) -> Dict:
        """Predict event times with iterative rolling predictions."""
        if package_data is None:
            package_data = self.fetch_package(package_id)
        
        if package_data is None:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'Package not found or invalid'
            }
        
        neptune_events = package_data.get('events', [])
        
        leg_plan_str = package_data.get('leg_plan')
        leg_plan = self._parse_leg_plan(leg_plan_str)
        
        if not leg_plan:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'No leg_plan available or failed to parse'
            }
        
        dest_postal = package_data.get('dest_postal')
        skeleton = self._create_skeleton_from_leg_plan(leg_plan, dest_postal)
        
        if not skeleton:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'Failed to create skeleton from leg_plan'
            }
        
        # Match Neptune events to skeleton
        filled_skeleton, match_errors, matched_neptune = self._match_neptune_events_to_skeleton(
            skeleton, neptune_events
        )
        
        # Keep original for comparison
        original_skeleton = [dict(s) for s in filled_skeleton]
        
        # Determine delivery status
        delivery_status = self._determine_delivery_status(filled_skeleton)
        
        # Get last known location
        last_known_location = None
        if delivery_status == 'IN_TRANSIT':
            last_known_location = self._get_last_known_location(filled_skeleton)
        
        # Check if first event has actual data
        if filled_skeleton[0].get('is_predicted', True):
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'First event (INDUCT) has no Neptune data - cannot start predictions',
                'delivery_status': delivery_status,
            }
        
        # Count events
        actual_event_count = sum(1 for s in filled_skeleton if s.get('neptune_matched', False))
        predicted_event_count = sum(1 for s in filled_skeleton if s.get('is_predicted', False))
        
        # In strict mode, reject delivered packages with missing events
        if self.strict_validation and delivery_status == 'DELIVERED' and predicted_event_count > 0:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': f'Delivered package has missing events: {predicted_event_count} events not found',
                'delivery_status': delivery_status,
                'match_errors': match_errors,
            }
        
        try:
            # Run iterative predictions
            working_skeleton, predictions_by_edge = self._run_iterative_predictions(
                package_data, filled_skeleton
            )
            
            # Build output events
            output_events = self._build_output_events(
                working_skeleton, predictions_by_edge, original_skeleton
            )
            
            # Calculate ETA
            eta = self._calculate_eta(output_events)
            
            # Calculate metrics
            all_ae = [e['ae'] for e in output_events if e.get('ae') is not None]
            valid_preds = [e['predicted_hours'] for e in output_events 
                          if e.get('predicted_hours') is not None]
            valid_actuals = [e['actual_hours'] for e in output_events 
                           if e.get('actual_hours') is not None]
            
            # Check for problems
            has_any_problem = any(
                e.get('context', {}).get('has_problem', False) 
                for e in output_events
            )
            problem_events = [
                e['event_idx'] for e in output_events 
                if e.get('context', {}).get('has_problem', False)
            ]
            
            result = {
                'package_id': package_id,
                'status': 'success',
                'delivery_status': delivery_status,
                'num_events': len(output_events),
                'actual_events': actual_event_count,
                'predicted_events': predicted_event_count,
                'source_postal': package_data.get('source_postal'),
                'dest_postal': dest_postal,
                'pdd': self._format_event_time(package_data.get('pdd')),
                'eta': eta,
                'has_problems': has_any_problem,
                'problem_event_indices': problem_events if problem_events else None,
                'last_known_location': last_known_location,
                'events': output_events,
                'metrics': {
                    'mae': float(np.mean(all_ae)) if all_ae else None,
                    'max_ae': float(np.max(all_ae)) if all_ae else None,
                    'min_ae': float(np.min(all_ae)) if all_ae else None,
                    'total_predicted_hours': float(np.sum(valid_preds)) if valid_preds else None,
                    'total_actual_hours': float(np.sum(valid_actuals)) if valid_actuals else None,
                }
            }
            
            if (result['metrics']['total_predicted_hours'] is not None and 
                result['metrics']['total_actual_hours'] is not None):
                result['metrics']['total_time_ae'] = abs(
                    result['metrics']['total_predicted_hours'] - 
                    result['metrics']['total_actual_hours']
                )
            
            # Calculate remaining time for in-transit packages
            if delivery_status == 'IN_TRANSIT':
                remaining_preds = [
                    e['predicted_hours'] for e in output_events 
                    if e.get('is_predicted', False) and e.get('predicted_hours') is not None
                ]
                if remaining_preds:
                    result['remaining_hours'] = float(np.sum(remaining_preds))
            
            return result
            
        except Exception as e:
            import traceback
            return {
                'package_id': package_id,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'delivery_status': delivery_status if 'delivery_status' in dir() else 'UNKNOWN',
            }
    
    @torch.no_grad()
    def predict_batch(self, package_ids: List[str]) -> List[Dict]:
        """Predict event times for multiple packages."""
        results = []
        
        print(f"Fetching {len(package_ids)} packages from Neptune...")
        package_data_map = {}
        
        for pkg_id in tqdm(package_ids, desc="Fetching packages"):
            package_data = self.fetch_package(pkg_id)
            if package_data is not None:
                package_data_map[pkg_id] = package_data
            else:
                results.append({
                    'package_id': pkg_id,
                    'status': 'error',
                    'error': 'Package not found or invalid'
                })
        
        print(f"Successfully fetched {len(package_data_map)} packages")
        
        for pkg_id in tqdm(list(package_data_map.keys()), desc="Running inference"):
            result = self.predict_single(pkg_id, package_data_map[pkg_id])
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save prediction results to local file."""
        ext = os.path.splitext(output_path)[1].lower()
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if ext == '.csv':
            flat_results = []
            for r in results:
                flat = {
                    'package_id': r['package_id'],
                    'status': r['status'],
                    'delivery_status': r.get('delivery_status'),
                    'error': r.get('error'),
                    'num_events': r.get('num_events'),
                    'actual_events': r.get('actual_events'),
                    'predicted_events': r.get('predicted_events'),
                    'source_postal': r.get('source_postal'),
                    'dest_postal': r.get('dest_postal'),
                    'pdd': r.get('pdd'),
                    'eta': r.get('eta'),
                    'remaining_hours': r.get('remaining_hours'),
                    'has_problems': r.get('has_problems'),
                }
                if 'metrics' in r and r['metrics']:
                    flat.update({
                        'mae': r['metrics'].get('mae'),
                        'max_ae': r['metrics'].get('max_ae'),
                        'min_ae': r['metrics'].get('min_ae'),
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
        """Clean up all resources."""
        if hasattr(self, 'extractor') and self.extractor is not None:
            try:
                self.extractor.close()
                print("Neptune connection closed")
            except Exception as e:
                print(f"Warning: Error closing Neptune connection: {e}")
            self.extractor = None
        
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        if hasattr(self, 'checkpoint') and self.checkpoint is not None:
            del self.checkpoint
            self.checkpoint = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        print("Resources cleaned up")


# ============================================================================
# SUMMARY PRINTING
# ============================================================================

def print_summary(results: List[Dict]):
    """Print summary statistics."""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nPackages: {len(results)} total | {len(successful)} success | {len(failed)} failed")
    
    delivered = [r for r in successful if r.get('delivery_status') == 'DELIVERED']
    in_transit = [r for r in successful if r.get('delivery_status') == 'IN_TRANSIT']
    
    print(f"\nDelivery Status:")
    print(f"  DELIVERED:  {len(delivered)}")
    print(f"  IN_TRANSIT: {len(in_transit)}")
    
    packages_with_problems = [r for r in successful if r.get('has_problems', False)]
    if packages_with_problems:
        print(f"\nPackages with problems: {len(packages_with_problems)}")
    
    if failed:
        print(f"\nFailed Packages:")
        for r in failed:
            error_msg = r.get('error', 'Unknown')
            if len(error_msg) > 80:
                error_msg = error_msg[:80] + "..."
            print(f"  ✗ {r['package_id']}: {error_msg}")
    
    if not successful:
        print("\nNo successful predictions to summarize.")
        return
    
    delivered_with_metrics = [r for r in delivered if r['metrics'].get('mae') is not None]
    
    all_mae = [r['metrics']['mae'] for r in delivered_with_metrics]
    all_total_ae = [r['metrics']['total_time_ae'] for r in delivered_with_metrics if r['metrics'].get('total_time_ae') is not None]
    all_total_pred = [r['metrics']['total_predicted_hours'] for r in delivered_with_metrics if r['metrics'].get('total_predicted_hours') is not None]
    all_total_actual = [r['metrics']['total_actual_hours'] for r in delivered_with_metrics if r['metrics'].get('total_actual_hours') is not None]
    
    print(f"\n{'─' * 80}")
    print("AGGREGATE METRICS (DELIVERED packages only)")
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
    
    if in_transit:
        print(f"\n{'─' * 80}")
        print("IN-TRANSIT PACKAGES - ETA PREDICTIONS")
        print(f"{'─' * 80}")
        
        remaining_hours = [r.get('remaining_hours') for r in in_transit if r.get('remaining_hours') is not None]
        if remaining_hours:
            print(f"\nRemaining Time (hours):")
            print(f"  Avg:  {np.mean(remaining_hours):.2f}")
            print(f"  Min:  {np.min(remaining_hours):.2f}")
            print(f"  Max:  {np.max(remaining_hours):.2f}")
    
    print(f"\n{'─' * 80}")
    print("PER-PACKAGE SUMMARY")
    print(f"{'─' * 80}")
    print(f"{'Package ID':<25} {'Status':<12} {'Total':<6} {'Act':<5} {'Pred':<5} {'MAE(h)':<8} {'ETA/Remaining':<20}")
    print(f"{'-'*25} {'-'*12} {'-'*6} {'-'*5} {'-'*5} {'-'*8} {'-'*20}")
    
    for r in successful:
        m = r['metrics']
        mae_str = f"{m['mae']:.2f}" if m.get('mae') is not None else "N/A"
        
        delivery_status = r.get('delivery_status', 'UNKNOWN')
        
        if delivery_status == 'IN_TRANSIT':
            eta = r.get('eta', '')
            remaining = r.get('remaining_hours')
            if eta:
                eta_str = eta[:16].replace('T', ' ')
                if remaining:
                    eta_str += f" ({remaining:.1f}h)"
            elif remaining:
                eta_str = f"~{remaining:.1f}h remaining"
            else:
                eta_str = "N/A"
        else:
            eta_str = "DELIVERED"
        
        print(f"{r['package_id']:<25} {delivery_status:<12} {r['num_events']:<6} {r.get('actual_events', '-'):<5} {r.get('predicted_events', '-'):<5} {mae_str:<8} {eta_str:<20}")
    
    print(f"\n{'─' * 80}")
    print("EVENT-LEVEL DETAILS")
    print(f"{'─' * 80}")
    
    for r in successful:
        status_indicator = f"[{r.get('delivery_status', 'UNKNOWN')}]"
        problem_indicator = " ⚠️ HAS PROBLEMS" if r.get('has_problems', False) else ""
        
        print(f"\n{r['package_id']} {status_indicator}{problem_indicator}")
        print(f"  Events: {r['num_events']} total ({r.get('actual_events', 0)} actual, {r.get('predicted_events', 0)} predicted)")
        
        if r.get('delivery_status') == 'IN_TRANSIT':
            last_loc = r.get('last_known_location', {})
            if last_loc:
                print(f"  Last Known: {last_loc.get('event_type')} at {last_loc.get('location')}")
            if r.get('eta'):
                print(f"  ETA: {r.get('eta')[:19].replace('T', ' ')} (in {r.get('remaining_hours', 0):.1f}h)")
            elif r.get('remaining_hours'):
                print(f"  Remaining: ~{r.get('remaining_hours'):.1f}h")
        
        print(f"\n  {'#':<3} {'Type':<10} {'Location':<8} {'Status':<6} {'Plan Time':<20} {'Event Time':<20} {'Pred Time':<20} {'Pred(h)':<8} {'Act(h)':<8} {'AE(h)':<8}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*6} {'-'*20} {'-'*20} {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
        
        for event in r['events']:
            pred_h_str = f"{event['predicted_hours']:.2f}" if event.get('predicted_hours') is not None else ""
            actual_h_str = f"{event['actual_hours']:.2f}" if event.get('actual_hours') is not None else ""
            ae_str = f"{event['ae']:.2f}" if event.get('ae') is not None else ""
            
            loc = str(event.get('location', 'N/A'))[:8]
            event_type = str(event.get('event_type', 'N/A'))[:10]
            
            is_predicted = event.get('is_predicted', False)
            status = "PRED" if is_predicted else "ACT"
            
            # Plan time
            plan_time = event.get('plan_time', '')
            if plan_time:
                plan_time_str = plan_time[:16].replace('T', ' ')
            else:
                plan_time_str = ""
            
            # Event time (empty for predicted events)
            event_time = event.get('event_time', '')
            if event_time:
                event_time_str = event_time[:16].replace('T', ' ')
            else:
                event_time_str = ""
            
            # Predicted time
            pred_time = event.get('predicted_time', '')
            if pred_time:
                pred_time_str = pred_time[:16].replace('T', ' ')
            else:
                pred_time_str = ""
            
            context = event.get('context', {})
            prefix = "  "
            if is_predicted:
                prefix = "→ "
            elif context.get('has_problem', False):
                prefix = "⚠ "
            
            print(f"{prefix}{event['event_idx']:<3} {event_type:<10} {loc:<8} {status:<6} {plan_time_str:<20} {event_time_str:<20} {pred_time_str:<20} {pred_h_str:<8} {actual_h_str:<8} {ae_str:<8}")
            
            if context.get('has_problem', False):
                problem = context.get('problem', '')
                if problem:
                    print(f"      └─ Problem: {problem}")
                if context.get('missort'):
                    print(f"      └─ Missort: Yes")
                if context.get('dwelling_hours'):
                    print(f"      └─ Dwelling: {context['dwelling_hours']}h")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    print("=" * 80)
    print("EVENT TIME PREDICTION - ITERATIVE ROLLING PREDICTIONS")
    print("With plan_time for each event")
    print("=" * 80)
    print(f"\nPackages to process: {len(PACKAGE_IDS)}")
    for pkg_id in PACKAGE_IDS:
        print(f"  - {pkg_id}")
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Strict validation: {STRICT_LEG_PLAN_VALIDATION}")
    print(f"Allow undelivered: {ALLOW_UNDELIVERED}")
    print("=" * 80)
    
    inference = None
    exit_code = 0
    
    try:
        inference = EventTimeInference(
            checkpoint_path=CHECKPOINT_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
            neptune_endpoint=NEPTUNE_ENDPOINT,
            device=DEVICE,
            strict_validation=STRICT_LEG_PLAN_VALIDATION,
            allow_undelivered=ALLOW_UNDELIVERED
        )
        
        results = inference.predict_batch(PACKAGE_IDS)
        
        print_summary(results)
        
        if OUTPUT_PATH:
            inference.save_results(results, OUTPUT_PATH)
        
        print("\n" + "=" * 80)
        print("✓ INFERENCE COMPLETE")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        exit_code = 1
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        if inference is not None:
            inference.close()
    
    return exit_code


if __name__ == '__main__':
    exit(main())
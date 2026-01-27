#!/usr/bin/env python3

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


PACKAGE_IDS = [
    "TBA328104955122",
    "TBA327930638904",
    "TBA328017923727",
    "TBA327930638904",
    "TBA327907450335",
    "TBA328017923366",
    "TBA327917781805",
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

CHECKPOINT_PATH = "best_model/best_model.pt"
PREPROCESSOR_PATH = "best_model/preprocessor.pkl"
NEPTUNE_ENDPOINT = "swa-shipgraph-neptune-instance-prod-us-east-1-read-replica.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182"
OUTPUT_PATH = "predictions.json"
DEVICE = "cuda"

STRICT_LEG_PLAN_VALIDATION = False
ALLOW_UNDELIVERED = True


def to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.long:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
    elif dtype == torch.float32:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    elif dtype == torch.bool:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.bool_))
    return torch.from_numpy(arr)


class EventTimeInference:
    
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
        
        print(f"Using device: {self.device}", flush=True)
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        
        print(f"Loading preprocessor from {preprocessor_path}", flush=True)
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        self.preprocessor = PackageLifecyclePreprocessor.load(preprocessor_path)
        
        print(f"Loading checkpoint from {checkpoint_path}", flush=True)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.vocab_sizes = self.checkpoint['vocab_sizes']
        self.feature_dims = self.checkpoint['feature_dims']
        
        full_config = Config.from_dict(self.checkpoint['config'])
        self.model_config = full_config.model
        
        print("Initializing model...", flush=True)
        self.model = EventTimePredictor.from_config(
            config=full_config,
            vocab_sizes=self.vocab_sizes,
            feature_dims=self.feature_dims,
            device=self.device,
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from epoch {self.checkpoint.get('epoch', 'unknown') + 1}", flush=True)
        metrics = self.checkpoint.get('metrics', {})
        if metrics:
            print(f"Checkpoint metrics:", flush=True)
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}", flush=True)
                else:
                    print(f"  {k}: {v}", flush=True)
        
        print(f"Connecting to Neptune at {neptune_endpoint}", flush=True)
        self.neptune_endpoint = neptune_endpoint
        self.extractor = NeptuneDataExtractor(
            endpoint=neptune_endpoint,
            max_workers=1
        )
        print("Neptune connection established", flush=True)
        print(f"Strict leg_plan validation: {self.strict_validation}", flush=True)
        print(f"Allow undelivered packages: {self.allow_undelivered}", flush=True)
    
    def fetch_package(self, package_id: str) -> Optional[Dict]:
        try:
            package_data = _extract_package_edges_optimized(self.extractor.main_client, package_id)
            
            if package_data is None:
                return None
            
            if not self.allow_undelivered:
                is_valid, invalid_reason = _validate_package_sequence(package_data)
                if not is_valid:
                    print(f"  Package {package_id} invalid: {invalid_reason}", flush=True)
                    return None
            
            package_data = _deduplicate_events(package_data)
            return package_data
            
        except Exception as e:
            print(f"  Error fetching package {package_id}: {e}", flush=True)
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
            print(f"    Warning: Failed to parse leg_plan: {e}", flush=True)
            return None
    
    def _create_skeleton_from_leg_plan(self, leg_plan: Dict, dest_postal: str) -> List[Dict]:
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
                
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'EXIT',
                    'location': sc_id,
                    'plan_time': sc_data.get('cpt'),
                    'cpt': None,
                    'ship_method': None,
                    'is_first_event': False,
                })
                event_idx += 1
            else:
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
                
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'EXIT',
                    'location': sc_id,
                    'plan_time': sc_data.get('cpt'),
                    'cpt': None,
                    'ship_method': None,
                    'is_first_event': False,
                })
                event_idx += 1
        
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
        if neptune_event:
            neptune_plan_time = neptune_event.get('plan_time')
            if neptune_plan_time:
                return neptune_plan_time
        
        if event_type == 'EXIT' and event_idx > 0:
            prev_skel = skeleton[event_idx - 1]
            prev_cpt = prev_skel.get('cpt')
            if prev_cpt:
                return prev_cpt
            prev_context = prev_skel.get('context', {})
            if prev_context.get('cpt'):
                return prev_context.get('cpt')
        
        skel = skeleton[event_idx]
        return skel.get('plan_time')
    
    def _match_neptune_events_to_skeleton(
        self, 
        skeleton: List[Dict], 
        neptune_events: List[Dict]
    ) -> Tuple[List[Dict], List[str], Dict[int, Dict]]:
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
        matched_neptune_events_by_skel_idx = {}
        
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
                
                plan_time = self._get_plan_time_for_event(
                    skel_idx, skel_type, filled_skeleton, matched_event
                )
                
                context = {'has_problem': False}
                
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
                
                cpt = matched_event.get('cpt') or skel.get('cpt')
                if cpt:
                    context['cpt'] = cpt
                    skel['cpt'] = cpt
                
                skel['plan_time'] = plan_time
                
                skel['context'] = context
            else:
                skel['event_time'] = None
                skel['neptune_matched'] = False
                skel['is_predicted'] = True
                
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
        node_cat = features['node_categorical_indices']
        pkg_cat = features['package_categorical']
        
        pkg_feat = features['package_features']
        if pkg_feat.ndim == 1:
            pkg_feat = pkg_feat.reshape(1, -1)
        
        data = Data(
            node_observable_time=to_tensor(features['node_observable_time'], torch.float32),
            node_observable_other=to_tensor(features['node_observable_other'], torch.float32),
            node_realized_time=to_tensor(features['node_realized_time'], torch.float32),
            node_realized_other=to_tensor(features['node_realized_other'], torch.float32),
            
            event_type_idx=to_tensor(node_cat['event_type'], torch.long),
            location_idx=to_tensor(node_cat['location'], torch.long),
            postal_idx=to_tensor(node_cat['postal'], torch.long),
            region_idx=to_tensor(node_cat['region'], torch.long),
            carrier_idx=to_tensor(node_cat['carrier'], torch.long),
            leg_type_idx=to_tensor(node_cat['leg_type'], torch.long),
            ship_method_idx=to_tensor(node_cat['ship_method'], torch.long),
            
            edge_index=to_tensor(features['edge_index'], torch.long),
            edge_features=to_tensor(features['edge_features'], torch.float32),
            
            package_features=to_tensor(pkg_feat, torch.float32),
            source_postal_idx=torch.tensor([pkg_cat['source_postal']], dtype=torch.long),
            dest_postal_idx=torch.tensor([pkg_cat['dest_postal']], dtype=torch.long),
            
            num_nodes=features['num_nodes'],
        )
        
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
    
    def _check_if_delivered(self, package_data: Dict) -> bool:
        """Check if package is delivered based on Neptune events."""
        neptune_events = package_data.get('events', [])
        for event in neptune_events:
            if event.get('event_type') == 'DELIVERY':
                return True
        return False
    
    @torch.no_grad()
    def _run_single_inference(self, synthetic_package: Dict) -> Optional[np.ndarray]:
        try:
            features = self.preprocessor.process_lifecycle(synthetic_package, return_labels=True)
            if features is None:
                return None
            
            if features['num_nodes'] < 2:
                return None
            
            graph_data = self._features_to_pyg_data(features)
            graph_data = graph_data.to(self.device)
            
            batch = Batch.from_data_list([graph_data])
            batch.node_counts = torch.tensor([graph_data.num_nodes], dtype=torch.long, device=self.device)
            batch.edge_counts = torch.tensor([graph_data.edge_index.shape[1]], dtype=torch.long, device=self.device)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                predictions = self.model(batch)
            
            preds = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
            preds_scaled = preds.float().cpu().numpy()
            preds_hours = self.preprocessor.inverse_transform_time(preds_scaled).flatten()
            
            return preds_hours
        except Exception as e:
            print(f"    Inference error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    @torch.no_grad()
    def predict_next_event(self, package_id: str, package_data: Optional[Dict] = None) -> Dict:
        if package_data is None:
            package_data = self.fetch_package(package_id)
        
        if package_data is None:
            return {
                'tracking_id': package_id,
                'status': 'error',
                'error': 'Package not found or invalid'
            }
        
        # Check if already delivered - return simple response
        if self._check_if_delivered(package_data):
            return {
                'tracking_id': package_id,
                'status': 'delivered'
            }
        
        neptune_events = package_data.get('events', [])
        
        leg_plan_str = package_data.get('leg_plan')
        leg_plan = self._parse_leg_plan(leg_plan_str)
        
        if not leg_plan:
            return {
                'tracking_id': package_id,
                'status': 'error',
                'error': 'No leg_plan available or failed to parse'
            }
        
        dest_postal = package_data.get('dest_postal')
        skeleton = self._create_skeleton_from_leg_plan(leg_plan, dest_postal)
        
        if not skeleton:
            return {
                'tracking_id': package_id,
                'status': 'error',
                'error': 'Failed to create skeleton from leg_plan'
            }
        
        filled_skeleton, match_errors, matched_neptune = self._match_neptune_events_to_skeleton(
            skeleton, neptune_events
        )
        
        original_skeleton = [dict(s) for s in filled_skeleton]
        
        delivery_status = self._determine_delivery_status(filled_skeleton)
        
        # Double check - if delivery status is DELIVERED, return simple response
        if delivery_status == 'DELIVERED':
            return {
                'tracking_id': package_id,
                'status': 'delivered'
            }
        
        last_known_location = self._get_last_known_location(filled_skeleton)
        
        if filled_skeleton[0].get('is_predicted', True):
            return {
                'tracking_id': package_id,
                'status': 'error',
                'error': 'First event (INDUCT) has no Neptune data - cannot start predictions',
                'delivery_status': delivery_status,
            }
        
        next_event_idx = None
        for i, skel in enumerate(filled_skeleton):
            if skel.get('is_predicted', False):
                next_event_idx = i
                break
        
        if next_event_idx is None:
            return {
                'tracking_id': package_id,
                'status': 'error',
                'error': 'All events matched but delivery status is not DELIVERED',
                'delivery_status': delivery_status,
            }
        
        if next_event_idx == 0:
            return {
                'tracking_id': package_id,
                'status': 'error',
                'error': 'Cannot predict - no actual events before the next event',
                'delivery_status': delivery_status,
            }
        
        actual_event_count = sum(1 for s in filled_skeleton if s.get('neptune_matched', False))
        
        try:
            events_for_inference = []
            for i in range(next_event_idx):
                if filled_skeleton[i].get('neptune_matched', False):
                    events_for_inference.append(dict(filled_skeleton[i]))
            
            if not events_for_inference:
                return {
                    'tracking_id': package_id,
                    'status': 'error',
                    'error': 'No actual events available for inference',
                    'delivery_status': delivery_status,
                }
            
            prev_event = events_for_inference[-1]
            prev_event_time = prev_event.get('event_time')
            
            if not prev_event_time:
                return {
                    'tracking_id': package_id,
                    'status': 'error',
                    'error': 'Previous event has no timestamp',
                    'delivery_status': delivery_status,
                }
            
            target_skel = filled_skeleton[next_event_idx]
            target_event_for_inference = dict(target_skel)
            target_event_for_inference['event_time'] = prev_event_time
            
            inference_events = events_for_inference + [target_event_for_inference]
            synthetic = self._build_synthetic_package(package_data, inference_events)
            
            preds = self._run_single_inference(synthetic)
            
            if preds is None or len(preds) == 0:
                return {
                    'tracking_id': package_id,
                    'status': 'error',
                    'error': 'Model inference failed',
                    'delivery_status': delivery_status,
                }
            
            pred_hours = float(preds[-1])
            predicted_dt = self._calculate_predicted_datetime(prev_event_time, pred_hours)
            
            output_events = []
            
            for i in range(next_event_idx):
                skel = filled_skeleton[i]
                
                event_output = {
                    'is_predicted': False,
                    'event_idx': skel['event_idx'],
                    'event_type': skel['event_type'],
                    'location': skel['location'],
                    'plan_time': self._format_event_time(skel.get('plan_time')),
                    'event_time': self._format_event_time(skel.get('event_time')),
                    'predicted_time': None,
                    'context': skel.get('context', {}),
                }
                
                if i > 0:
                    prev_skel = original_skeleton[i-1]
                    if prev_skel.get('neptune_matched', False):
                        prev_actual_time = prev_skel.get('event_time')
                        curr_actual_time = skel.get('event_time')
                        if prev_actual_time and curr_actual_time:
                            prev_dt = self._parse_event_time(prev_actual_time)
                            curr_dt = self._parse_event_time(curr_actual_time)
                            if prev_dt and curr_dt:
                                actual_hours = (curr_dt - prev_dt).total_seconds() / 3600.0
                                event_output['actual_hours'] = actual_hours
                
                output_events.append(event_output)
            
            predicted_event_output = {
                'is_predicted': True,
                'event_idx': target_skel['event_idx'],
                'event_type': target_skel['event_type'],
                'location': target_skel['location'],
                'plan_time': self._format_event_time(target_skel.get('plan_time')),
                'event_time': None,
                'predicted_time': self._format_event_time(predicted_dt),
                'predicted_hours': pred_hours,
                'context': target_skel.get('context', {}),
            }
            output_events.append(predicted_event_output)
            
            has_any_problem = any(
                e.get('context', {}).get('has_problem', False) 
                for e in output_events
            )
            problem_events = [
                e['event_idx'] for e in output_events 
                if e.get('context', {}).get('has_problem', False)
            ]
            
            remaining_events = len(filled_skeleton) - next_event_idx - 1
            
            result = {
                'tracking_id': package_id,
                'status': 'in_transit',
                'delivery_status': delivery_status,
                'num_events': len(output_events),
                'actual_events': actual_event_count,
                'predicted_events': 1,
                'next_event_idx': next_event_idx,
                'remaining_events_after_next': remaining_events,
                'source_postal': package_data.get('source_postal'),
                'dest_postal': dest_postal,
                'pdd': self._format_event_time(package_data.get('pdd')),
                'next_event_prediction': {
                    'event_idx': next_event_idx,
                    'event_type': target_skel['event_type'],
                    'location': target_skel['location'],
                    'predicted_time': self._format_event_time(predicted_dt),
                    'predicted_hours_from_previous': pred_hours,
                    'previous_event_time': self._format_event_time(prev_event_time),
                },
                'eta': None,
                'has_problems': has_any_problem,
                'problem_event_indices': problem_events if problem_events else None,
                'last_known_location': last_known_location,
                'events': output_events,
                'metrics': {
                    'mae': None,
                    'max_ae': None,
                    'min_ae': None,
                    'total_predicted_hours': pred_hours,
                    'total_actual_hours': None,
                }
            }
            
            return result
            
        except Exception as e:
            import traceback
            return {
                'tracking_id': package_id,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'delivery_status': delivery_status if 'delivery_status' in dir() else 'UNKNOWN',
            }
    
    @torch.no_grad()
    def predict_single(self, package_id: str, package_data: Optional[Dict] = None) -> Dict:
        if package_data is None:
            package_data = self.fetch_package(package_id)
        
        if package_data is None:
            return {
                'tracking_id': package_id,
                'status': 'error',
                'error': 'Package not found or invalid'
            }
        
        # Check if already delivered - return simple response and skip prediction
        if self._check_if_delivered(package_data):
            return {
                'tracking_id': package_id,
                'status': 'delivered'
            }
        
        # Not delivered - run prediction
        return self.predict_next_event(package_id, package_data)
    
    @torch.no_grad()
    def predict_batch(self, package_ids: List[str]) -> List[Dict]:
        results = []
        
        print(f"Fetching {len(package_ids)} packages from Neptune...", flush=True)
        package_data_map = {}
        
        for pkg_id in tqdm(package_ids, desc="Fetching packages"):
            package_data = self.fetch_package(pkg_id)
            if package_data is not None:
                package_data_map[pkg_id] = package_data
            else:
                results.append({
                    'tracking_id': pkg_id,
                    'status': 'error',
                    'error': 'Package not found or invalid'
                })
        
        print(f"Successfully fetched {len(package_data_map)} packages", flush=True)
        
        # Separate delivered and non-delivered packages
        delivered_packages = []
        packages_to_predict = []
        
        for pkg_id, pkg_data in package_data_map.items():
            if self._check_if_delivered(pkg_data):
                delivered_packages.append(pkg_id)
                results.append({
                    'tracking_id': pkg_id,
                    'status': 'delivered'
                })
            else:
                packages_to_predict.append((pkg_id, pkg_data))
        
        print(f"Skipping {len(delivered_packages)} delivered packages", flush=True)
        print(f"Running inference on {len(packages_to_predict)} in-transit packages", flush=True)
        
        for pkg_id, pkg_data in tqdm(packages_to_predict, desc="Running inference"):
            result = self.predict_next_event(pkg_id, pkg_data)
            results.append(result)
        
        return results
    
    @torch.no_grad()
    def predict_next_event_batch(self, package_ids: List[str]) -> List[Dict]:
        results = []
        
        print(f"Fetching {len(package_ids)} packages from Neptune...", flush=True)
        package_data_map = {}
        
        for pkg_id in tqdm(package_ids, desc="Fetching packages"):
            package_data = self.fetch_package(pkg_id)
            if package_data is not None:
                package_data_map[pkg_id] = package_data
            else:
                results.append({
                    'tracking_id': pkg_id,
                    'status': 'error',
                    'error': 'Package not found or invalid'
                })
        
        print(f"Successfully fetched {len(package_data_map)} packages", flush=True)
        
        # Separate delivered and non-delivered packages
        delivered_packages = []
        packages_to_predict = []
        
        for pkg_id, pkg_data in package_data_map.items():
            if self._check_if_delivered(pkg_data):
                delivered_packages.append(pkg_id)
                results.append({
                    'tracking_id': pkg_id,
                    'status': 'delivered'
                })
            else:
                packages_to_predict.append((pkg_id, pkg_data))
        
        print(f"Skipping {len(delivered_packages)} delivered packages", flush=True)
        print(f"Running inference on {len(packages_to_predict)} in-transit packages", flush=True)
        
        for pkg_id, pkg_data in tqdm(packages_to_predict, desc="Running next-event inference"):
            result = self.predict_next_event(pkg_id, pkg_data)
            results.append(result)
        
        return results
    
    def close(self):
        if hasattr(self, 'extractor') and self.extractor is not None:
            try:
                self.extractor.close()
                print("Neptune connection closed", flush=True)
            except Exception as e:
                print(f"Warning: Error closing Neptune connection: {e}", flush=True)
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
        print("Resources cleaned up", flush=True)


def print_summary(results: List[Dict]):
    
    def format_time(t):
        if t is None:
            return 'N/A'
        if isinstance(t, datetime):
            return t.strftime('%Y-%m-%d %H:%M')
        if isinstance(t, str):
            return t[:16].replace('T', ' ')
        return 'N/A'
    
    # Categorize results
    delivered = [r for r in results if r.get('status') == 'delivered']
    in_transit = [r for r in results if r.get('status') == 'in_transit']
    failed = [r for r in results if r.get('status') == 'error']
    
    print("\n" + "=" * 80, flush=True)
    print("INFERENCE RESULTS SUMMARY", flush=True)
    print("=" * 80, flush=True)
    
    print(f"\nPackages: {len(results)} total", flush=True)
    print(f"  DELIVERED (skipped):  {len(delivered)}", flush=True)
    print(f"  IN_TRANSIT (predicted): {len(in_transit)}", flush=True)
    print(f"  ERROR: {len(failed)}", flush=True)
    
    if delivered:
        print(f"\n{'─' * 80}", flush=True)
        print("DELIVERED PACKAGES (Skipped - No Prediction Needed)", flush=True)
        print(f"{'─' * 80}", flush=True)
        for r in delivered:
            print(f"  ✓ {r['tracking_id']}: delivered", flush=True)
    
    if failed:
        print(f"\n{'─' * 80}", flush=True)
        print("FAILED PACKAGES", flush=True)
        print(f"{'─' * 80}", flush=True)
        for r in failed:
            error_msg = r.get('error', 'Unknown')
            if len(error_msg) > 60:
                error_msg = error_msg[:60] + "..."
            print(f"  ✗ {r['tracking_id']}: {error_msg}", flush=True)
    
    if in_transit:
        print(f"\n{'─' * 80}", flush=True)
        print("IN-TRANSIT PACKAGES - NEXT EVENT PREDICTIONS", flush=True)
        print(f"{'─' * 80}", flush=True)
        
        print(f"\n{'Tracking ID':<20} {'Last Event':<12} {'Next Event':<12} {'Pred Time':<20} {'Pred(h)':<10}", flush=True)
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*20} {'-'*10}", flush=True)
        
        for r in in_transit:
            last_loc = r.get('last_known_location', {})
            last_event_type = last_loc.get('event_type', 'N/A')[:12] if last_loc else 'N/A'
            
            next_pred = r.get('next_event_prediction', {})
            next_event_type = next_pred.get('event_type', 'N/A')[:12] if next_pred else 'N/A'
            
            pred_time_str = format_time(next_pred.get('predicted_time')) if next_pred else 'N/A'
            
            pred_hours = next_pred.get('predicted_hours_from_previous') if next_pred else None
            pred_hours_str = f"{pred_hours:.2f}" if pred_hours is not None else 'N/A'
            
            print(f"{r['tracking_id']:<20} {last_event_type:<12} {next_event_type:<12} {pred_time_str:<20} {pred_hours_str:<10}", flush=True)
        
        # Detailed view for in-transit packages
        print(f"\n{'─' * 80}", flush=True)
        print("IN-TRANSIT DETAILED EVENT VIEW", flush=True)
        print(f"{'─' * 80}", flush=True)
        
        for r in in_transit:
            print(f"\n{'=' * 80}", flush=True)
            print(f"{r['tracking_id']} [IN_TRANSIT]", flush=True)
            print(f"{'=' * 80}", flush=True)
            
            print(f"  Source: {r.get('source_postal', 'N/A')} → Dest: {r.get('dest_postal', 'N/A')}", flush=True)
            print(f"  PDD: {format_time(r.get('pdd'))}", flush=True)
            print(f"  Events shown: {r.get('num_events', 0)} ({r.get('actual_events', 0)} actual + 1 predicted)", flush=True)
            print(f"  Remaining after next: {r.get('remaining_events_after_next', 'N/A')} events", flush=True)
            
            last_loc = r.get('last_known_location', {})
            if last_loc:
                last_time_str = format_time(last_loc.get('event_time'))
                print(f"  Last Known: {last_loc.get('event_type')} at {last_loc.get('location')} ({last_time_str})", flush=True)
            
            next_pred = r.get('next_event_prediction', {})
            if next_pred:
                pred_time_str = format_time(next_pred.get('predicted_time'))
                prev_time_str = format_time(next_pred.get('previous_event_time'))
                pred_h = next_pred.get('predicted_hours_from_previous')
                pred_h_str = f"{pred_h:.2f}h" if pred_h is not None else 'N/A'
                
                print(f"\n  >>> NEXT EVENT PREDICTION <<<", flush=True)
                print(f"      Event: {next_pred.get('event_type')} at {next_pred.get('location')}", flush=True)
                print(f"      Previous Event Time: {prev_time_str}", flush=True)
                print(f"      Predicted Time: {pred_time_str}", flush=True)
                print(f"      Time from Previous: {pred_h_str}", flush=True)
            
            events = r.get('events', [])
            if events:
                print(f"\n  {'#':<3} {'Type':<10} {'Location':<8} {'Status':<6} {'Plan Time':<20} {'Event Time':<20} {'Pred Time':<20}", flush=True)
                print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*6} {'-'*20} {'-'*20} {'-'*20}", flush=True)
                
                for event in events:
                    is_predicted = event.get('is_predicted', False)
                    status = "PRED" if is_predicted else "ACT"
                    
                    plan_time_str = format_time(event.get('plan_time'))
                    event_time_str = format_time(event.get('event_time'))
                    pred_time_str = format_time(event.get('predicted_time'))
                    
                    loc = str(event.get('location', 'N/A'))[:8]
                    event_type = str(event.get('event_type', 'N/A'))[:10]
                    
                    prefix = "→ " if is_predicted else "  "
                    
                    print(f"{prefix}{event.get('event_idx', 0):<3} {event_type:<10} {loc:<8} {status:<6} {plan_time_str:<20} {event_time_str:<20} {pred_time_str:<20}", flush=True)
    
    print(f"\n{'=' * 80}", flush=True)
    print("END OF RESULTS", flush=True)
    print(f"{'=' * 80}", flush=True)


def main():
    print("=" * 80, flush=True)
    print("EVENT TIME PREDICTION - NEXT EVENT PREDICTION", flush=True)
    print("=" * 80, flush=True)
    print(f"\nPackages to process: {len(PACKAGE_IDS)}", flush=True)
    for pkg_id in PACKAGE_IDS:
        print(f"  - {pkg_id}", flush=True)
    print(f"\nCheckpoint: {CHECKPOINT_PATH}", flush=True)
    print(f"Preprocessor: {PREPROCESSOR_PATH}", flush=True)
    print(f"Output: {OUTPUT_PATH}", flush=True)
    print(f"Strict validation: {STRICT_LEG_PLAN_VALIDATION}", flush=True)
    print(f"Allow undelivered: {ALLOW_UNDELIVERED}", flush=True)
    print("=" * 80, flush=True)
    
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
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {OUTPUT_PATH}", flush=True)
        
        print("\n" + "=" * 80, flush=True)
        print("✓ INFERENCE COMPLETE", flush=True)
        print("=" * 80, flush=True)
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}", flush=True)
        exit_code = 1
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user", flush=True)
        exit_code = 1
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        if inference is not None:
            inference.close()
    
    return exit_code


if __name__ == '__main__':
    exit(main())
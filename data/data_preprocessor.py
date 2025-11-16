import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Set, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import ast

class PackageLifecyclePreprocessor:
    """Preprocess package lifecycle data for graph transformer"""
    
    def __init__(self, config):
        self.config = config
        
        self.event_type_encoder = LabelEncoder()
        self.sort_center_encoder = LabelEncoder()
        self.carrier_encoder = LabelEncoder()
        self.leg_type_encoder = LabelEncoder()
        self.ship_method_encoder = LabelEncoder()
        
        self.time_scaler = StandardScaler()
        self.package_feature_scaler = StandardScaler()
        self.plan_time_diff_scaler = StandardScaler()
        
        self.event_types = config.data.event_types
        self.problem_types = config.data.problem_types
        self.fitted = False
    
    def _parse_problem_field(self, problem_value) -> List[str]:
        """
        Parse problem field which can be:
        - None
        - "[\"WRONG_NODE\"]" (JSON string)
        - ["WRONG_NODE"] (list)
        """
        if problem_value is None or problem_value == 'null':
            return []
        
        # If already a list
        if isinstance(problem_value, list):
            return [str(p) for p in problem_value]
        
        # If string, try to parse as JSON
        if isinstance(problem_value, str):
            try:
                # Try JSON parsing
                parsed = json.loads(problem_value)
                if isinstance(parsed, list):
                    return [str(p) for p in parsed]
                return [str(parsed)]
            except json.JSONDecodeError:
                try:
                    # Try ast.literal_eval for Python literals
                    parsed = ast.literal_eval(problem_value)
                    if isinstance(parsed, list):
                        return [str(p) for p in parsed]
                    return [str(parsed)]
                except:
                    # If all parsing fails, return as single item
                    if problem_value.strip():
                        return [problem_value.strip()]
                    return []
        
        return []
    
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
        
        # ✅ Already a datetime object
        if isinstance(time_value, datetime):
            return time_value
        
        # ✅ String - try to parse
        if isinstance(time_value, str):
            if time_value == 'null' or time_value.strip() == '':
                return None
            
            try:
                # Handle ISO format with or without Z
                return datetime.fromisoformat(str(time_value).replace('Z', '+00:00'))
            except Exception as e:
                return None
        
        return None
    
    def _calculate_time_vs_plan(self, event_time: Union[str, datetime], 
                                plan_time: Union[str, datetime, None]) -> float:
        """
        Calculate time difference between event_time and plan_time in hours
        Positive = late, Negative = early, 0 = on time
        
        Args:
            event_time: Can be datetime object or string
            plan_time: Can be datetime object, string, or None
        
        Returns:
            Time difference in hours, or 0.0 if plan_time is not available
        """
        # ✅ Parse both times to datetime objects
        event_dt = self._parse_datetime(event_time)
        plan_dt = self._parse_datetime(plan_time)
        
        if event_dt is None or plan_dt is None:
            return 0.0
        
        try:
            diff_hours = (event_dt - plan_dt).total_seconds() / 3600
            
            # Clamp to reasonable range (-720 to +720 hours = 30 days)
            diff_hours = max(-720, min(diff_hours, 720))
            
            return float(diff_hours)
        except Exception as e:
            return 0.0
    
    def _get_plan_time_for_event(self, event: Dict, prev_event: Dict = None) -> Union[str, datetime, None]:
        """
        Get the appropriate plan_time for an event
        
        Args:
            event: Current event
            prev_event: Previous event (for EXIT events)
        
        Returns:
            plan_time value (string or datetime) or None
        """
        event_type = str(event.get('event_type', ''))
        
        # ✅ For EXIT events, use previous event's CPT
        if event_type == 'EXIT' and prev_event is not None:
            cpt = prev_event.get('cpt')
            if cpt and cpt != 'null':
                return cpt
        
        # ✅ For other events, use their own plan_time
        plan_time = event.get('plan_time')
        if plan_time and plan_time != 'null':
            return plan_time
        
        return None
    
    def fit(self, df: pd.DataFrame):
        """Fit encoders and scalers on training data"""
        
        all_sort_centers = set()
        all_carriers = set()
        all_leg_types = set()
        all_ship_methods = set()
    
        for _, row in df.iterrows():
            events = row['events']
            for event in events:
                if 'sort_center' in event and event['sort_center']:
                    all_sort_centers.add(str(event['sort_center']))
                    
                if 'carrier_id' in event and event['carrier_id']:
                    all_carriers.add(str(event['carrier_id']))
                    
                if 'leg_type' in event and event['leg_type']:
                    all_leg_types.add(str(event['leg_type']))
                
                if 'ship_method' in event and event['ship_method']:
                    all_ship_methods.add(str(event['ship_method']))
        
        # Add unknown tokens
        all_sort_centers.add('UNKNOWN')
        all_carriers.add('UNKNOWN')
        all_leg_types.add('UNKNOWN')
        all_ship_methods.add('UNKNOWN')
        
        # Fit encoders
        self.event_type_encoder.fit(self.event_types)
        self.sort_center_encoder.fit(sorted(list(all_sort_centers)))
        self.carrier_encoder.fit(sorted(list(all_carriers)))
        self.leg_type_encoder.fit(sorted(list(all_leg_types)))
        self.ship_method_encoder.fit(sorted(list(all_ship_methods)))
        
        # Store problem types for multi-hot encoding
        self.problem_type_to_idx = {pt: idx for idx, pt in enumerate(self.problem_types)}
        
        print(f"Found {len(all_sort_centers)} sort centers")
        print(f"Found {len(all_carriers)} carriers")
        print(f"Found {len(all_leg_types)} leg types")
        print(f"Found {len(all_ship_methods)} ship methods")
        print(f"Found {len(self.problem_types)} problem types")
    
        # Fit time scaler
        time_deltas = []
        for _, row in df.iterrows():
            events = row['events']
            for i in range(1, len(events)):
                try:
                    prev_time = self._parse_datetime(events[i-1]['event_time'])
                    curr_time = self._parse_datetime(events[i]['event_time'])
                    
                    if prev_time and curr_time:
                        delta = (curr_time - prev_time).total_seconds() / 3600
                        time_deltas.append([delta])
                except Exception as e:
                    continue
        
        if time_deltas:
            self.time_scaler.fit(np.array(time_deltas))
        
        # ✅ Fit plan time difference scaler
        plan_time_diffs = []
        for _, row in df.iterrows():
            events = row['events']
            for i, event in enumerate(events):
                # Get previous event for EXIT events
                prev_event = events[i-1] if i > 0 else None
                
                # Get appropriate plan_time
                plan_time = self._get_plan_time_for_event(event, prev_event)
                
                diff = self._calculate_time_vs_plan(
                    event.get('event_time'),
                    plan_time
                )
                plan_time_diffs.append([diff])
        
        if plan_time_diffs:
            self.plan_time_diff_scaler.fit(np.array(plan_time_diffs))
            print(f"\nFitted plan_time_diff scaler on {len(plan_time_diffs)} samples")
            print(f"  Mean: {np.mean(plan_time_diffs):.2f} hours")
            print(f"  Std: {np.std(plan_time_diffs):.2f} hours")
            print(f"  Min: {np.min(plan_time_diffs):.2f} hours")
            print(f"  Max: {np.max(plan_time_diffs):.2f} hours")
        
        # Fit package feature scaler
        package_features = df[['weight', 'length', 'width', 'height']].fillna(0).values
        self.package_feature_scaler.fit(package_features)
        
        self.fitted = True
        return self
    
    def _safe_compare(self, val1, val2) -> int:
        """Safely compare two values, handling None/empty cases"""
        if val1 is None or val2 is None:
            return 0
        if val1 == '' or val2 == '':
            return 0
        return int(str(val1) == str(val2))
    
    def _encode_problems(self, problem_value) -> np.ndarray:
        """
        Create multi-hot encoding for problem types
        Returns array of shape (num_problem_types,)
        """
        encoding = np.zeros(len(self.problem_types), dtype=np.float32)
        
        problems = self._parse_problem_field(problem_value)
        
        if not problems:
            # No problem - set NO_PROBLEM flag
            if 'NO_PROBLEM' in self.problem_type_to_idx:
                encoding[self.problem_type_to_idx['NO_PROBLEM']] = 1.0
        else:
            # Set flags for each problem type
            for problem in problems:
                if problem in self.problem_type_to_idx:
                    encoding[self.problem_type_to_idx[problem]] = 1.0
        
        return encoding
    
    def _extract_edge_features(self, event_from: Dict, event_to: Dict, 
                               time_from: datetime, time_to: datetime,
                               delay_from: float, delay_to: float) -> list:
        """Extract features for an edge between two events"""
        
        # 1. Time delta (hours)
        try:
            time_delta = (time_to - time_from).total_seconds() / 3600
            # Clamp to reasonable range to avoid extreme values
            time_delta = max(0, min(time_delta, 720))  # Max 30 days
        except Exception as e:
            print(f"Warning: Could not calculate time delta: {e}")
            time_delta = 0.0
        
        # 2. Same sort center flag
        sc_from = event_from.get('sort_center')
        sc_to = event_to.get('sort_center')
        same_sc = self._safe_compare(sc_from, sc_to)
        
        # 3. Same carrier flag
        carrier_from = event_from.get('carrier_id')
        carrier_to = event_to.get('carrier_id')
        same_carrier = self._safe_compare(carrier_from, carrier_to)
        
        # 4. Same ship method flag
        ship_method_from = event_from.get('ship_method')
        ship_method_to = event_to.get('ship_method')
        same_ship_method = self._safe_compare(ship_method_from, ship_method_to)
        
        # 5. Missort flag from source event (if INDUCT or LINEHAUL)
        has_missort_from = 0.0
        if event_from['event_type'] in ['INDUCT', 'LINEHAUL']:
            has_missort_from = float(event_from.get('missort', False))
        
        # 6. Problem flag from source event (if EXIT)
        has_problem_from = 0.0
        if event_from['event_type'] == 'EXIT':
            problem_value = event_from.get('problem')
            problems = self._parse_problem_field(problem_value)
            has_problem_from = 1.0 if problems else 0.0
        
        # 7. Delay propagation (change in delay from previous event)
        delay_change = delay_to - delay_from
        
        return [
            float(time_delta), 
            float(same_sc), 
            float(same_carrier),
            float(same_ship_method),
            has_missort_from,
            has_problem_from,
            float(delay_change)
        ]
    
    def process_lifecycle(self, package_data: Dict, return_labels: bool = True) -> Dict:
        """Process a single package lifecycle into graph features"""
        
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before processing")
        
        events = package_data['events']
        num_events = len(events)
        
        # Validate minimum events
        if num_events < 1:
            raise ValueError("Package must have at least 1 event")
        
        node_features = []
        event_times = []
        event_delays = []
        
        # Process node features
        for i, event in enumerate(events):
            # Event type (one-hot)
            event_type = str(event['event_type'])
            event_type_idx = self.event_type_encoder.transform([event_type])[0]
            event_type_onehot = np.zeros(len(self.event_types))
            event_type_onehot[event_type_idx] = 1
            
            # Sort center (current)
            sort_center = event.get('sort_center', 'UNKNOWN')
            sort_center = str(sort_center) if sort_center else 'UNKNOWN'
            if sort_center not in self.sort_center_encoder.classes_:
                sort_center = 'UNKNOWN'
            sort_center_idx = self.sort_center_encoder.transform([sort_center])[0]
            
            # ✅ From sort center
            if event_type in ['INDUCT', 'EXIT']:
                # For INDUCT and EXIT, from_sort_center is own sort_center
                from_sort_center_idx = sort_center_idx
            else:  # LINEHAUL, DELIVERY
                # For LINEHAUL and DELIVERY, from_sort_center is previous event's sort_center
                if i > 0:
                    prev_sort_center = events[i-1].get('sort_center', 'UNKNOWN')
                    prev_sort_center = str(prev_sort_center) if prev_sort_center else 'UNKNOWN'
                    if prev_sort_center not in self.sort_center_encoder.classes_:
                        prev_sort_center = 'UNKNOWN'
                    from_sort_center_idx = self.sort_center_encoder.transform([prev_sort_center])[0]
                else:
                    # First event but not INDUCT/EXIT (edge case)
                    from_sort_center_idx = sort_center_idx
            
            # Carrier
            carrier = event.get('carrier_id', 'UNKNOWN')
            carrier = str(carrier) if carrier else 'UNKNOWN'
            if carrier not in self.carrier_encoder.classes_:
                carrier = 'UNKNOWN'
            carrier_idx = self.carrier_encoder.transform([carrier])[0]
            
            # Leg type
            leg_type = event.get('leg_type', 'UNKNOWN')
            leg_type = str(leg_type) if leg_type else 'UNKNOWN'
            if leg_type not in self.leg_type_encoder.classes_:
                leg_type = 'UNKNOWN'
            leg_type_idx = self.leg_type_encoder.transform([leg_type])[0]
            
            # Ship method
            ship_method = event.get('ship_method', 'UNKNOWN')
            ship_method = str(ship_method) if ship_method else 'UNKNOWN'
            if ship_method not in self.ship_method_encoder.classes_:
                ship_method = 'UNKNOWN'
            ship_method_idx = self.ship_method_encoder.transform([ship_method])[0]
            
            # Event time
            event_time = self._parse_datetime(event['event_time'])
            if event_time is None:
                raise ValueError(f"Invalid event_time for event {i}")
            event_times.append(event_time)
            
            # ✅ Get appropriate plan_time (use CPT from previous event for EXIT)
            prev_event = events[i-1] if i > 0 else None
            plan_time = self._get_plan_time_for_event(event, prev_event)
            
            # ✅ Calculate time vs plan (delay)
            time_vs_plan = self._calculate_time_vs_plan(event['event_time'], plan_time)
            event_delays.append(time_vs_plan)
            
            # Normalize time_vs_plan
            time_vs_plan_scaled = self.plan_time_diff_scaler.transform([[time_vs_plan]])[0, 0]
            
            # Has plan time flag
            has_plan_time = 1.0 if plan_time is not None else 0.0
            
            if i == 0:
                time_since_start = 0
                time_since_prev = 0
            else:
                time_since_start = (event_time - event_times[0]).total_seconds() / 3600
                time_since_prev = (event_time - event_times[i-1]).total_seconds() / 3600
            
            # Positional encoding
            position = i / max(1, num_events - 1)
            
            # Dwelling time (for EXIT events)
            dwelling_time = event.get('dwelling_seconds', 0) / 3600 if event.get('dwelling_seconds') else 0
            
            # Missort flag (for INDUCT and LINEHAUL events)
            missort_flag = 0.0
            if event_type in ['INDUCT', 'LINEHAUL']:
                missort_flag = float(event.get('missort', False))
            
            # Problem encoding (for EXIT events)
            problem_encoding = np.zeros(len(self.problem_types), dtype=np.float32)
            if event_type == 'EXIT':
                problem_encoding = self._encode_problems(event.get('problem'))
            
            # ✅ Combine features (added from_sort_center_idx)
            features = np.concatenate([
                event_type_onehot,
                [sort_center_idx, from_sort_center_idx, carrier_idx, leg_type_idx, ship_method_idx],
                [time_since_start, time_since_prev, position, dwelling_time],
                [time_vs_plan_scaled, has_plan_time],
                [missort_flag],
                problem_encoding
            ])
            
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Package-level features
        package_features = np.array([
            package_data.get('weight', 0),
            package_data.get('length', 0),
            package_data.get('width', 0),
            package_data.get('height', 0)
        ], dtype=np.float32).reshape(1, -1)
        
        package_features = self.package_feature_scaler.transform(package_features).flatten()
        package_features_expanded = np.tile(package_features, (num_events, 1))
        node_features = np.concatenate([node_features, package_features_expanded], axis=1)
        
        # Edge features (sequential connections)
        edge_index = []
        edge_features = []
        
        if num_events > 1:
            for i in range(num_events - 1):
                edge_index.append([i, i+1])
                
                # Extract edge features using helper method
                edge_feat = self._extract_edge_features(
                    events[i], 
                    events[i+1],
                    event_times[i],
                    event_times[i+1],
                    event_delays[i],
                    event_delays[i+1]
                )
                edge_features.append(edge_feat)
            
            edge_index = np.array(edge_index, dtype=np.int64).T
            edge_features = np.array(edge_features, dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 7), dtype=np.float32)
        
        # Validate edge features shape
        expected_num_edges = max(0, num_events - 1)
        if edge_features.shape[0] != expected_num_edges:
            print(f"WARNING: Edge feature count mismatch. Expected {expected_num_edges}, got {edge_features.shape[0]}")
            edge_features = np.zeros((expected_num_edges, 7), dtype=np.float32)
        
        if edge_features.shape[1] != 7:
            print(f"ERROR: Edge feature dimension is {edge_features.shape[1]}, expected 7")
            edge_features = np.zeros((expected_num_edges, 7), dtype=np.float32)
        
        result = {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'num_nodes': num_events,
            'package_id': package_data['package_id']
        }
        
        # Labels: time to next event for each node (except last)
        if return_labels:
            labels = []
            for i in range(num_events - 1):
                time_to_next = (event_times[i+1] - event_times[i]).total_seconds() / 3600
                labels.append(time_to_next)
            
            if labels:
                labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
                
                if self.config.data.normalize_time:
                    labels = self.time_scaler.transform(labels)
            else:
                labels = np.zeros((0, 1), dtype=np.float32)
            
            result['labels'] = labels
            
            # Create mask with SAME length as nodes
            label_mask = np.zeros(num_events, dtype=bool)
            label_mask[:-1] = True
            result['label_mask'] = label_mask
        
        return result
    
    def inverse_transform_time(self, scaled_time: np.ndarray) -> np.ndarray:
        """Convert scaled time back to hours"""
        if self.config.data.normalize_time:
            return self.time_scaler.inverse_transform(scaled_time)
        return scaled_time
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of all feature components"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return {
            'event_type_dim': len(self.event_types),
            'sort_center_vocab': len(self.sort_center_encoder.classes_),
            'carrier_vocab': len(self.carrier_encoder.classes_),
            'leg_type_vocab': len(self.leg_type_encoder.classes_),
            'ship_method_vocab': len(self.ship_method_encoder.classes_),
            'problem_types_dim': len(self.problem_types),
            'time_features_dim': 4,
            'plan_time_features_dim': 2,
            'missort_dim': 1,
            'package_features_dim': 4,
            'edge_features_dim': 7,
            'total_node_features': (
                len(self.event_types) +
                5 +  # ✅ Changed from 4 to 5 (sort_center_idx, from_sort_center_idx, carrier_idx, leg_type_idx, ship_method_idx)
                4 +
                2 +
                1 +
                len(self.problem_types) +
                4
            )
        }
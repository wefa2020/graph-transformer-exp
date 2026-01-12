"""
data/preprocessing.py - Causal Package Lifecycle Preprocessor
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import ast
import os
import pickle


class PackageLifecyclePreprocessor:
    """
    Causal preprocessor for package lifecycle data.
    
    CRITICAL: Prevents data leakage by separating features into:
    
    1. OBSERVABLE features (known BEFORE event happens):
       - event_type, location, carrier, ship_method, leg_type
       - plan_time (when we expect it to happen)
       - distance, region info (static/known)
       
    2. REALIZED features (only known AFTER event happens):
       - actual event_time
       - time_since_prev (computed from actual times)
       - dwelling_time
       - time_vs_plan (how late/early vs plan)
       - problems (discovered during processing)
    
    For causal prediction of edge i→i+1:
    - Nodes 0..i: Use OBSERVABLE + REALIZED features
    - Node i+1 (target): Use OBSERVABLE features only
    - Nodes i+2..N: Use OBSERVABLE features only (or masked)
    
    Plan Time Logic:
    - EXIT events: plan_time = previous event's CPT (INDUCT/LINEHAUL)
    - Other events: use their own plan_time
    """
    
    def __init__(self, config, distance_df: pd.DataFrame = None, distance_file_path: str = None):
        """
        Args:
            config: Configuration object with data.event_types and data.problem_types
            distance_df: Pre-loaded DataFrame with distance data
            distance_file_path: Path to location_distances_complete.csv
        """
        self.config = config
        
        # Distance and region lookup
        self.distance_lookup = {}
        self.region_lookup = {}
        self.distance_unit = 'miles'
        self.distance_file_path = distance_file_path
        
        self._load_distance_data(distance_df=distance_df)
        
        # === Categorical Encoders ===
        self.event_type_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.carrier_encoder = LabelEncoder()
        self.leg_type_encoder = LabelEncoder()
        self.ship_method_encoder = LabelEncoder()
        self.postal_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()
        
        # === Scalers for REALIZED features ===
        self.time_since_prev_scaler = StandardScaler()
        self.dwelling_time_scaler = StandardScaler()
        self.time_vs_plan_scaler = StandardScaler()
        
        # === Scalers for OBSERVABLE features ===
        self.time_until_plan_scaler = StandardScaler()
        self.edge_distance_scaler = StandardScaler()
        
        # === Label scaler ===
        self.label_time_scaler = StandardScaler()
        
        # === Package Feature Scaler ===
        self.package_feature_scaler = StandardScaler()
        
        # Event and problem types from config
        self.event_types = config.data.event_types
        self.problem_types = config.data.problem_types
        self.problem_type_to_idx = {}
        
        self.fitted = False
        self.vocab_sizes = {}
    
    # =========================================================================
    # DISTANCE DATA LOADING
    # =========================================================================
    
    def _load_distance_data(self, distance_df: pd.DataFrame = None):
        """Load distance and region lookup tables."""
        df_dist = None
        
        if distance_df is not None:
            df_dist = distance_df
            print("Using provided distance DataFrame")
        elif self.distance_file_path is not None and os.path.exists(self.distance_file_path):
            try:
                df_dist = pd.read_csv(self.distance_file_path)
                print(f"Loaded distance data from: {self.distance_file_path}")
            except Exception as e:
                print(f"Error loading distance file: {e}")
        else:
            default_path = os.path.join('data', 'location_distances_complete.csv')
            if os.path.exists(default_path):
                try:
                    df_dist = pd.read_csv(default_path)
                    self.distance_file_path = default_path
                    print(f"Loaded distance data from default path: {default_path}")
                except Exception as e:
                    print(f"Error loading default distance file: {e}")
        
        if df_dist is not None:
            self._process_distance_dataframe(df_dist)
        else:
            print("Warning: No distance data available. Distance features will be 0.")
    
    def _process_distance_dataframe(self, df_dist: pd.DataFrame):
        """Process distance DataFrame and populate lookup tables."""
        try:
            required_cols = ['location_id_1', 'location_id_2']
            if not all(col in df_dist.columns for col in required_cols):
                print(f"Warning: Expected columns {required_cols} not found")
                return
            
            # Determine distance column
            if 'distance_miles' in df_dist.columns:
                dist_col = 'distance_miles'
                self.distance_unit = 'miles'
            elif 'distance_km' in df_dist.columns:
                dist_col = 'distance_km'
                self.distance_unit = 'km'
            else:
                print("Warning: No distance column found")
                return
            
            # Build lookups
            for _, row in df_dist.iterrows():
                loc1 = str(row['location_id_1']).strip()
                loc2 = str(row['location_id_2']).strip()
                
                try:
                    distance = float(row[dist_col])
                except (ValueError, TypeError):
                    continue
                
                if pd.isna(distance) or distance < 0:
                    continue
                
                # Bidirectional
                self.distance_lookup[(loc1, loc2)] = distance
                self.distance_lookup[(loc2, loc1)] = distance
                
                # Region info
                if 'super_region_1' in df_dist.columns:
                    region1 = row.get('super_region_1')
                    if pd.notna(region1) and str(region1).strip():
                        self.region_lookup[loc1] = str(region1).strip()
                
                if 'super_region_2' in df_dist.columns:
                    region2 = row.get('super_region_2')
                    if pd.notna(region2) and str(region2).strip():
                        self.region_lookup[loc2] = str(region2).strip()
            
            print(f"Loaded {len(self.distance_lookup) // 2} unique distance pairs")
            print(f"Locations with region info: {len(self.region_lookup)}")
            
        except Exception as e:
            print(f"Error processing distance data: {e}")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _get_distance(self, from_location: str, to_location: str) -> Tuple[float, bool]:
        """Get distance between two locations."""
        if not self.distance_lookup:
            return 0.0, False
        
        from_loc = str(from_location).strip()
        to_loc = str(to_location).strip()
        
        if from_loc == to_loc:
            return 0.0, True
        
        if (from_loc, to_loc) in self.distance_lookup:
            return self.distance_lookup[(from_loc, to_loc)], True
        
        return 0.0, False
    
    def _get_region(self, location: str) -> str:
        """Get region for a location."""
        if not self.region_lookup:
            return 'UNKNOWN'
        return self.region_lookup.get(str(location).strip(), 'UNKNOWN')
    
    def _get_sort_center(self, event: Dict) -> str:
        """Get sort_center from event."""
        sort_center = event.get('sort_center')
        if sort_center and str(sort_center) != 'nan':
            return str(sort_center)
        return 'UNKNOWN'
    
    def _get_delivery_station(self, event: Dict) -> str:
        """Get delivery_station from event."""
        station = event.get('delivery_station')
        if station and str(station) != 'nan':
            return str(station)
        return 'UNKNOWN'
    
    def _get_delivery_postal(self, event: Dict) -> str:
        """Get postal code from delivery_location (DELIVERY events only)."""
        if str(event.get('event_type', '')) != 'DELIVERY':
            return 'UNKNOWN'
        
        delivery_loc = event.get('delivery_location')
        if delivery_loc and isinstance(delivery_loc, dict):
            postal_id = delivery_loc.get('id')
            if postal_id:
                return str(postal_id)
        return 'UNKNOWN'
    
    def _get_location(self, event: Dict) -> str:
        """Get the primary location for an event."""
        event_type = str(event.get('event_type', ''))
        if event_type == 'DELIVERY':
            return self._get_delivery_station(event)
        return self._get_sort_center(event)
    
    def _parse_datetime(self, time_value) -> Optional[datetime]:
        """Parse time value to datetime object."""
        if time_value is None:
            return None
        if isinstance(time_value, datetime):
            return time_value
        if isinstance(time_value, str):
            if time_value == 'null' or time_value.strip() == '':
                return None
            try:
                return datetime.fromisoformat(str(time_value).replace('Z', '+00:00'))
            except:
                return None
        return None
    
    def _parse_problem_field(self, problem_value) -> List[str]:
        """Parse problem field which can be None, JSON string, or list."""
        if problem_value is None or problem_value == 'null':
            return []
        
        if isinstance(problem_value, list):
            return [str(p) for p in problem_value]
        
        if isinstance(problem_value, str):
            try:
                parsed = json.loads(problem_value)
                if isinstance(parsed, list):
                    return [str(p) for p in parsed]
                return [str(parsed)]
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(problem_value)
                    if isinstance(parsed, list):
                        return [str(p) for p in parsed]
                    return [str(parsed)]
                except:
                    if problem_value.strip():
                        return [problem_value.strip()]
        return []
    
    def _encode_problems(self, problem_value) -> np.ndarray:
        """Create multi-hot encoding for problem types."""
        encoding = np.zeros(len(self.problem_types), dtype=np.float32)
        problems = self._parse_problem_field(problem_value)
        
        if not problems:
            if 'NO_PROBLEM' in self.problem_type_to_idx:
                encoding[self.problem_type_to_idx['NO_PROBLEM']] = 1.0
        else:
            for problem in problems:
                if problem in self.problem_type_to_idx:
                    encoding[self.problem_type_to_idx[problem]] = 1.0
        
        return encoding
    
    def _safe_encode(self, encoder: LabelEncoder, value, default: str = 'UNKNOWN') -> int:
        """Safely encode a value, returning UNKNOWN index if not found."""
        if value is None or value == '' or str(value) == 'nan':
            value = default
        else:
            value = str(value)
        
        if value not in encoder.classes_:
            value = default
        
        return int(encoder.transform([value])[0])
    
    def _extract_cyclical_time(self, dt: datetime, prefix: str = '') -> Dict[str, float]:
        """Extract cyclical time features from datetime."""
        return {
            f'{prefix}hour_sin': np.sin(2 * np.pi * dt.hour / 24),
            f'{prefix}hour_cos': np.cos(2 * np.pi * dt.hour / 24),
            f'{prefix}dow_sin': np.sin(2 * np.pi * dt.weekday() / 7),
            f'{prefix}dow_cos': np.cos(2 * np.pi * dt.weekday() / 7),
            f'{prefix}dom_sin': np.sin(2 * np.pi * dt.day / 31),
            f'{prefix}dom_cos': np.cos(2 * np.pi * dt.day / 31),
            f'{prefix}month_sin': np.sin(2 * np.pi * dt.month / 12),
            f'{prefix}month_cos': np.cos(2 * np.pi * dt.month / 12),
        }
    
    def _get_plan_time_for_event(self, event: Dict, prev_event: Dict = None) -> Optional[str]:
        """
        Get plan_time for an event.
        - EXIT: use previous event's CPT (if INDUCT/LINEHAUL)
        - Others: use own plan_time
        """
        event_type = str(event.get('event_type', ''))
        
        if event_type == 'EXIT' and prev_event is not None:
            prev_type = str(prev_event.get('event_type', ''))
            if prev_type in ['INDUCT', 'LINEHAUL']:
                cpt = prev_event.get('cpt')
                if cpt and cpt != 'null':
                    return cpt
        
        plan_time = event.get('plan_time')
        if plan_time and plan_time != 'null':
            return plan_time
        
        return None
    
    def _get_exit_problem(self, event: Dict, events: List[Dict], 
                          event_idx: int) -> Tuple[np.ndarray, float]:
        """
        For INDUCT/LINEHAUL events, get problem from next EXIT at same sort center.
        """
        event_type = str(event.get('event_type', ''))
        
        if event_type not in ['INDUCT', 'LINEHAUL']:
            return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
        
        current_sc = self._get_sort_center(event)
        if current_sc == 'UNKNOWN':
            return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
        
        for i in range(event_idx + 1, len(events)):
            next_event = events[i]
            next_type = str(next_event.get('event_type', ''))
            next_sc = self._get_sort_center(next_event)
            
            if next_type == 'EXIT' and next_sc == current_sc:
                problems = self._parse_problem_field(next_event.get('problem'))
                encoding = self._encode_problems(next_event.get('problem'))
                has_problem = 1.0 if problems else 0.0
                return encoding, has_problem
            
            if next_sc != current_sc and next_sc != 'UNKNOWN':
                break
        
        return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
    
    # =========================================================================
    # FITTING
    # =========================================================================
    
    def fit(self, df: pd.DataFrame):
        """Fit encoders and scalers on training data."""
        
        all_locations = set()
        all_carriers = set()
        all_leg_types = set()
        all_ship_methods = set()
        all_postals = set()
        all_regions = set()
        
        # Add regions from distance file
        all_regions.update(self.region_lookup.values())
        
        # Collect categorical values
        for _, row in df.iterrows():
            # Package level postal codes
            source_postal = row.get('source_postal')
            dest_postal = row.get('dest_postal')
            if source_postal and str(source_postal) != 'nan':
                all_postals.add(str(source_postal))
            if dest_postal and str(dest_postal) != 'nan':
                all_postals.add(str(dest_postal))
            
            events = row['events']
            for event in events:
                loc = self._get_location(event)
                if loc != 'UNKNOWN':
                    all_locations.add(loc)
                    region = self._get_region(loc)
                    if region != 'UNKNOWN':
                        all_regions.add(region)
                
                postal = self._get_delivery_postal(event)
                if postal != 'UNKNOWN':
                    all_postals.add(postal)
                
                if event.get('carrier_id'):
                    all_carriers.add(str(event['carrier_id']))
                if event.get('leg_type'):
                    all_leg_types.add(str(event['leg_type']))
                if event.get('ship_method'):
                    all_ship_methods.add(str(event['ship_method']))
        
        # Fit encoders with special tokens (PAD=0, UNKNOWN=1)
        special = ['PAD', 'UNKNOWN']
        self.event_type_encoder.fit(['PAD'] + self.event_types)
        self.location_encoder.fit(special + sorted(list(all_locations - {'UNKNOWN'})))
        self.carrier_encoder.fit(special + sorted(list(all_carriers - {'UNKNOWN'})))
        self.leg_type_encoder.fit(special + sorted(list(all_leg_types - {'UNKNOWN'})))
        self.ship_method_encoder.fit(special + sorted(list(all_ship_methods - {'UNKNOWN'})))
        self.postal_encoder.fit(special + sorted(list(all_postals - {'UNKNOWN'})))
        self.region_encoder.fit(special + sorted(list(all_regions - {'UNKNOWN'})))
        
        # Store vocabulary sizes
        self.vocab_sizes = {
            'event_type': len(self.event_type_encoder.classes_),
            'location': len(self.location_encoder.classes_),
            'carrier': len(self.carrier_encoder.classes_),
            'leg_type': len(self.leg_type_encoder.classes_),
            'ship_method': len(self.ship_method_encoder.classes_),
            'postal': len(self.postal_encoder.classes_),
            'region': len(self.region_encoder.classes_),
        }
        
        self.problem_type_to_idx = {pt: idx for idx, pt in enumerate(self.problem_types)}
        
        print(f"\n=== Vocabulary Sizes ===")
        for name, size in self.vocab_sizes.items():
            print(f"  {name}: {size}")
        print(f"  problem_types: {len(self.problem_types)}")
        
        # Collect values for scalers
        time_since_prev_vals = []
        dwelling_vals = []
        time_vs_plan_vals = []
        time_until_plan_vals = []
        distance_vals = []
        label_vals = []
        
        for _, row in df.iterrows():
            events = row['events']
            event_times = []
            
            for i, event in enumerate(events):
                event_time = self._parse_datetime(event['event_time'])
                if event_time is None:
                    continue
                event_times.append(event_time)
                
                prev_event = events[i-1] if i > 0 else None
                
                # Time since previous (REALIZED)
                if i > 0 and len(event_times) > 1:
                    time_since_prev = (event_time - event_times[-2]).total_seconds() / 3600
                    time_since_prev_vals.append([time_since_prev])
                
                # Dwelling time (REALIZED)
                dwelling = (event.get('dwelling_seconds', 0) or 0) / 3600
                dwelling_vals.append([dwelling])
                
                # Time vs plan (REALIZED)
                plan_time = self._get_plan_time_for_event(event, prev_event)
                plan_dt = self._parse_datetime(plan_time)
                if plan_dt:
                    time_vs_plan = (event_time - plan_dt).total_seconds() / 3600
                    time_vs_plan = max(-720, min(time_vs_plan, 720))
                    time_vs_plan_vals.append([time_vs_plan])
                
                # Time until next plan (OBSERVABLE)
                if i < len(events) - 1:
                    next_event = events[i + 1]
                    next_plan_time = self._get_plan_time_for_event(next_event, event)
                    next_plan_dt = self._parse_datetime(next_plan_time)
                    if next_plan_dt:
                        time_until_plan = (next_plan_dt - event_time).total_seconds() / 3600
                        time_until_plan = max(-720, min(time_until_plan, 720))
                        time_until_plan_vals.append([time_until_plan])
                
                # Label (transit time)
                if i < len(events) - 1:
                    next_time = self._parse_datetime(events[i+1]['event_time'])
                    if next_time:
                        label = (next_time - event_time).total_seconds() / 3600
                        label_vals.append([label])
            
            # Distances
            for i in range(len(events) - 1):
                from_loc = self._get_location(events[i])
                to_loc = self._get_location(events[i + 1])
                dist, has_dist = self._get_distance(from_loc, to_loc)
                if has_dist and dist > 0:
                    distance_vals.append([dist])
        
        # Fit scalers
        self._fit_scaler(self.time_since_prev_scaler, time_since_prev_vals, 'time_since_prev')
        self._fit_scaler(self.dwelling_time_scaler, dwelling_vals, 'dwelling')
        self._fit_scaler(self.time_vs_plan_scaler, time_vs_plan_vals, 'time_vs_plan')
        self._fit_scaler(self.time_until_plan_scaler, time_until_plan_vals, 'time_until_plan')
        self._fit_scaler(self.edge_distance_scaler, distance_vals, 'distance')
        self._fit_scaler(self.label_time_scaler, label_vals, 'label')
        
        # Package features
        package_features = df[['weight', 'length', 'width', 'height']].fillna(0).values
        self.package_feature_scaler.fit(package_features)
        
        self._print_scaler_stats()
        
        self.fitted = True
        return self
    
    def _fit_scaler(self, scaler: StandardScaler, values: List, name: str):
        """Fit a scaler with fallback for empty data."""
        if values:
            scaler.fit(np.array(values))
        else:
            print(f"Warning: No data for {name} scaler, using default")
            scaler.fit(np.array([[0.0]]))
    
    def _print_scaler_stats(self):
        """Print statistics for all fitted scalers."""
        print("\n=== Scaler Statistics ===")
        scalers = {
            'time_since_prev': self.time_since_prev_scaler,
            'dwelling_time': self.dwelling_time_scaler,
            'time_vs_plan': self.time_vs_plan_scaler,
            'time_until_plan': self.time_until_plan_scaler,
            'edge_distance': self.edge_distance_scaler,
            'label_time': self.label_time_scaler,
        }
        for name, scaler in scalers.items():
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                print(f"  {name}: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    
    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================
    
    def _extract_observable_features(self, event: Dict, prev_event: Dict,
                                      reference_time: datetime,
                                      events: List[Dict], event_idx: int) -> np.ndarray:
        """
        Extract OBSERVABLE features for an event.
        These are features known BEFORE the event happens.
        
        Used for: Target node, future nodes
        
        Features:
        - time_until_plan (scaled): Time from reference to this event's plan
        - has_plan: Flag if plan time exists
        - is_delivery: Flag if this is a DELIVERY event
        - plan_time cyclical (8): When event is planned
        - position (1): Normalized position in sequence
        """
        event_type = str(event.get('event_type', 'UNKNOWN'))
        num_events = len(events)
        
        # Plan time for this event
        plan_time = self._get_plan_time_for_event(event, prev_event)
        plan_dt = self._parse_datetime(plan_time)
        
        # Time until plan (from reference_time perspective)
        has_plan = 0.0
        time_until_plan_scaled = 0.0
        plan_cyclical = np.zeros(8, dtype=np.float32)
        
        if plan_dt and reference_time:
            has_plan = 1.0
            time_until_plan = (plan_dt - reference_time).total_seconds() / 3600
            time_until_plan = max(-720, min(time_until_plan, 720))
            time_until_plan_scaled = self.time_until_plan_scaler.transform([[time_until_plan]])[0, 0]
            
            plan_features = self._extract_cyclical_time(plan_dt, prefix='')
            plan_cyclical = np.array([
                plan_features['hour_sin'], plan_features['hour_cos'],
                plan_features['dow_sin'], plan_features['dow_cos'],
                plan_features['dom_sin'], plan_features['dom_cos'],
                plan_features['month_sin'], plan_features['month_cos'],
            ], dtype=np.float32)
        
        # Is delivery flag
        is_delivery = 1.0 if event_type == 'DELIVERY' else 0.0
        
        # Position in sequence (normalized)
        position = event_idx / max(1, num_events - 1)
        
        # Combine observable features
        # Total: 1 + 1 + 1 + 8 + 1 = 12
        observable = np.concatenate([
            [time_until_plan_scaled],  # 1
            [has_plan],                # 1
            [is_delivery],             # 1
            plan_cyclical,             # 8
            [position],                # 1
        ]).astype(np.float32)
        
        return observable
    
    def _extract_realized_features(self, event: Dict, prev_event: Dict,
                                    prev_time: Optional[datetime],
                                    events: List[Dict], event_idx: int) -> np.ndarray:
        """
        Extract REALIZED features for an event.
        These are features only known AFTER the event happens.
        
        Used for: Source node, past nodes
        
        Features:
        - time_since_prev (scaled): Time since previous event
        - dwelling (scaled): Dwelling time at location
        - has_dwelling: Flag if dwelling > 0
        - time_vs_plan (scaled): How late/early vs plan
        - actual_time cyclical (8): When event actually happened
        - missort flag: If package was missorted
        - has_problem flag: If problems discovered
        - problem_encoding: Multi-hot encoding of problems
        """
        event_type = str(event.get('event_type', ''))
        event_time = self._parse_datetime(event['event_time'])
        
        # Time since previous
        time_since_prev_scaled = 0.0
        if prev_time and event_time:
            time_since_prev = (event_time - prev_time).total_seconds() / 3600
            time_since_prev_scaled = self.time_since_prev_scaler.transform([[time_since_prev]])[0, 0]
        
        # Dwelling time
        dwelling = (event.get('dwelling_seconds', 0) or 0) / 3600
        dwelling_scaled = self.dwelling_time_scaler.transform([[dwelling]])[0, 0]
        has_dwelling = 1.0 if dwelling > 0 else 0.0
        
        # Time vs plan (how late/early)
        plan_time = self._get_plan_time_for_event(event, prev_event)
        plan_dt = self._parse_datetime(plan_time)
        time_vs_plan_scaled = 0.0
        if plan_dt and event_time:
            time_vs_plan = (event_time - plan_dt).total_seconds() / 3600
            time_vs_plan = max(-720, min(time_vs_plan, 720))
            time_vs_plan_scaled = self.time_vs_plan_scaler.transform([[time_vs_plan]])[0, 0]
        
        # Actual event time cyclical features
        actual_cyclical = np.zeros(8, dtype=np.float32)
        if event_time:
            actual_features = self._extract_cyclical_time(event_time, prefix='')
            actual_cyclical = np.array([
                actual_features['hour_sin'], actual_features['hour_cos'],
                actual_features['dow_sin'], actual_features['dow_cos'],
                actual_features['dom_sin'], actual_features['dom_cos'],
                actual_features['month_sin'], actual_features['month_cos'],
            ], dtype=np.float32)
        
        # Missort flag
        missort = 0.0
        if event_type in ['INDUCT', 'LINEHAUL']:
            missort = float(event.get('missort', False))
        
        # Problem encoding
        problem_encoding, has_problem = self._get_exit_problem(event, events, event_idx)
        
        # Combine realized features
        # Total: 1 + 1 + 1 + 1 + 8 + 1 + 1 + len(problem_types) = 14 + len(problem_types)
        realized = np.concatenate([
            [time_since_prev_scaled],  # 1
            [dwelling_scaled],         # 1
            [has_dwelling],            # 1
            [time_vs_plan_scaled],     # 1
            actual_cyclical,           # 8
            [missort],                 # 1
            [has_problem],             # 1
            problem_encoding,          # len(problem_types)
        ]).astype(np.float32)
        
        return realized
    
    def _extract_edge_features(self, source_event: Dict, target_event: Dict,
                                source_time: datetime) -> np.ndarray:
        """
        Extract edge features (all OBSERVABLE - based on known info).
        
        Features:
        - distance (scaled): Distance between locations
        - has_distance: Flag if distance is known
        - same_location: Flag if same location
        - cross_region: Flag if crossing regions
        - source_time cyclical (4): Hour and day of week from source
        """
        source_loc = self._get_location(source_event)
        target_loc = self._get_location(target_event)
        
        # Distance
        distance, has_distance = self._get_distance(source_loc, target_loc)
        distance_scaled = self.edge_distance_scaler.transform([[distance]])[0, 0]
        
        # Same location flag
        same_location = float(source_loc == target_loc and source_loc != 'UNKNOWN')
        
        # Cross-region flag
        source_region = self._get_region(source_loc)
        target_region = self._get_region(target_loc)
        cross_region = float(source_region != target_region and 
                            source_region != 'UNKNOWN' and target_region != 'UNKNOWN')
        
        # Source time features (4 features: hour sin/cos, dow sin/cos)
        source_time_features = self._extract_cyclical_time(source_time, prefix='')
        
        # Total: 1 + 1 + 1 + 1 + 4 = 8
        edge_features = np.array([
            distance_scaled,
            float(has_distance),
            same_location,
            cross_region,
            source_time_features['hour_sin'],
            source_time_features['hour_cos'],
            source_time_features['dow_sin'],
            source_time_features['dow_cos'],
        ], dtype=np.float32)
        
        return edge_features
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    def process_lifecycle(self, package_data: Dict, return_labels: bool = True) -> Dict:
        """
        Process a package lifecycle with CAUSAL feature extraction.
        
        Returns:
        - node_observable_features: (N, observable_dim) - always available
        - node_realized_features: (N, realized_dim) - only for past nodes
        - node_categorical_indices: Dict of (N,) arrays
        - edge_index: (2, E) - sequential edges
        - edge_features: (E, edge_dim)
        - package_features: (package_dim,)
        - package_categorical: Dict with source/dest postal
        - labels: (E, 1) - transit times (scaled)
        - labels_raw: (E, 1) - transit times (hours)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before processing")
        
        events = package_data['events']
        num_events = len(events)
        
        if num_events < 1:
            raise ValueError("Package must have at least 1 event")
        
        # Parse all event times
        event_times = []
        for event in events:
            et = self._parse_datetime(event['event_time'])
            if et is None:
                raise ValueError("Invalid event_time")
            event_times.append(et)
        
        # === Extract features for ALL nodes ===
        node_observable_features = []
        node_realized_features = []
        node_categorical_indices = {
            'event_type': [], 'location': [], 'postal': [], 'region': [],
            'carrier': [], 'leg_type': [], 'ship_method': [],
        }
        
        for i, event in enumerate(events):
            event_type = str(event.get('event_type', 'UNKNOWN'))
            prev_event = events[i-1] if i > 0 else None
            prev_time = event_times[i-1] if i > 0 else None
            
            # Reference time for observable features
            reference_time = prev_time if i > 0 else event_times[0]
            
            # Observable features
            obs_features = self._extract_observable_features(
                event, prev_event, reference_time, events, i
            )
            node_observable_features.append(obs_features)
            
            # Realized features
            real_features = self._extract_realized_features(
                event, prev_event, prev_time, events, i
            )
            node_realized_features.append(real_features)
            
            # Categorical indices
            location = self._get_location(event)
            postal = self._get_delivery_postal(event)
            region = self._get_region(location)
            
            node_categorical_indices['event_type'].append(
                self._safe_encode(self.event_type_encoder, event_type)
            )
            node_categorical_indices['location'].append(
                self._safe_encode(self.location_encoder, location)
            )
            node_categorical_indices['postal'].append(
                self._safe_encode(self.postal_encoder, postal)
            )
            node_categorical_indices['region'].append(
                self._safe_encode(self.region_encoder, region)
            )
            node_categorical_indices['carrier'].append(
                self._safe_encode(self.carrier_encoder, event.get('carrier_id'))
            )
            node_categorical_indices['leg_type'].append(
                self._safe_encode(self.leg_type_encoder, event.get('leg_type'))
            )
            node_categorical_indices['ship_method'].append(
                self._safe_encode(self.ship_method_encoder, event.get('ship_method'))
            )
        
        node_observable_features = np.array(node_observable_features, dtype=np.float32)
        node_realized_features = np.array(node_realized_features, dtype=np.float32)
        
        for key in node_categorical_indices:
            node_categorical_indices[key] = np.array(node_categorical_indices[key], dtype=np.int64)
        
        # === Package features ===
        package_features = np.array([
            package_data.get('weight', 0) or 0,
            package_data.get('length', 0) or 0,
            package_data.get('width', 0) or 0,
            package_data.get('height', 0) or 0
        ], dtype=np.float32).reshape(1, -1)
        package_features_scaled = self.package_feature_scaler.transform(package_features).flatten()
        
        # === Edge features ===
        edge_index = []
        edge_features = []
        
        for i in range(num_events - 1):
            edge_index.append([i, i + 1])
            edge_feat = self._extract_edge_features(
                events[i], events[i + 1], event_times[i]
            )
            edge_features.append(edge_feat)
        
        if edge_index:
            edge_index = np.array(edge_index, dtype=np.int64).T
            edge_features = np.array(edge_features, dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 8), dtype=np.float32)
        
        # === Build result ===
        result = {
            'node_observable_features': node_observable_features,
            'node_realized_features': node_realized_features,
            'node_categorical_indices': node_categorical_indices,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'package_features': package_features_scaled,
            'package_categorical': {
                'source_postal': self._safe_encode(self.postal_encoder, package_data.get('source_postal')),
                'dest_postal': self._safe_encode(self.postal_encoder, package_data.get('dest_postal')),
            },
            'num_nodes': num_events,
            'package_id': package_data.get('package_id', 'unknown'),
        }
        
        # === Labels ===
        if return_labels:
            labels = []
            for i in range(num_events - 1):
                transit_hours = (event_times[i+1] - event_times[i]).total_seconds() / 3600
                labels.append(transit_hours)
            
            if labels:
                labels_raw = np.array(labels, dtype=np.float32).reshape(-1, 1)
                labels_scaled = self.label_time_scaler.transform(labels_raw)
            else:
                labels_raw = np.zeros((0, 1), dtype=np.float32)
                labels_scaled = np.zeros((0, 1), dtype=np.float32)
            
            result['labels'] = labels_scaled
            result['labels_raw'] = labels_raw
        
        return result
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def inverse_transform_time(self, scaled_time):
        """Convert scaled time back to hours."""
        if scaled_time is None:
            return None
        
        if hasattr(scaled_time, 'numpy'):
            scaled_time = scaled_time.numpy()
        elif not isinstance(scaled_time, np.ndarray):
            scaled_time = np.array(scaled_time)
        
        if scaled_time.ndim == 0:
            return self.label_time_scaler.inverse_transform([[scaled_time]])[0, 0]
        if scaled_time.ndim == 1:
            return self.label_time_scaler.inverse_transform(scaled_time.reshape(-1, 1)).flatten()
        return self.label_time_scaler.inverse_transform(scaled_time)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of all feature components."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        # Observable: time_until_plan(1) + has_plan(1) + is_delivery(1) + plan_cyclical(8) + position(1) = 12
        observable_dim = 12
        
        # Realized: time_since_prev(1) + dwelling(2) + time_vs_plan(1) + actual_cyclical(8) + flags(2) + problems
        realized_dim = 14 + len(self.problem_types)
        
        # Edge: distance(2) + same_loc(1) + cross_region(1) + source_time(4) = 8
        edge_dim = 8
        
        # Package: weight, length, width, height = 4
        package_dim = 4
        
        return {
            'vocab_sizes': self.vocab_sizes.copy(),
            'observable_dim': observable_dim,
            'realized_dim': realized_dim,
            'edge_dim': edge_dim,
            'package_dim': package_dim,
            'num_problem_types': len(self.problem_types),
        }
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for embedding layers."""
        return self.vocab_sizes.copy()
    
    def save(self, path: str):
        """Save preprocessor to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'PackageLifecyclePreprocessor':
        """Load preprocessor from file."""
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {path}")
        return preprocessor
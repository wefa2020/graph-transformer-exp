"""
data/data_preprocessor.py - Causal Package Lifecycle Preprocessor with Time2Vec Support
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict
import json
import ast
import os
import io
import pickle


class PackageLifecyclePreprocessor:
    """
    Causal preprocessor for package lifecycle data with Time2Vec support.
    
    All vocabulary comes from config - no data collection during fit.
    Region lookup (location -> region mapping) comes from distance file.
    """
    
    def __init__(self, config, distance_df: pd.DataFrame = None, distance_file_path: str = None):
        """
        Args:
            config: Configuration object with data.vocab containing all vocabulary lists
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
        
        # === Scalers for continuous features ===
        self.time_since_prev_scaler = StandardScaler()
        self.dwelling_time_scaler = StandardScaler()
        self.time_delta_scaler = StandardScaler()
        self.elapsed_time_scaler = StandardScaler()
        self.edge_distance_scaler = StandardScaler()
        self.label_time_scaler = StandardScaler()
        self.package_feature_scaler = StandardScaler()
        
        # === Get vocabulary from config ===
        self.vocab = self._get_vocab_from_config(config)
        self.problem_type_to_idx = {pt: idx for idx, pt in enumerate(self.vocab['problem_types'])}
        
        self.fitted = False
        self.vocab_sizes = {}
        
        # === Unknown value tracking ===
        self.track_unknowns = False
        self.unknown_values = defaultdict(set)
        self.unknown_counts = defaultdict(int)
        
        # === Feature dimensions ===
        self.observable_time_dim = 6
        self.realized_time_dim = 6
    
    def _get_vocab_from_config(self, config) -> Dict[str, List[str]]:
        """Extract all vocabulary from config."""
        vocab = {
            'event_types': [],
            'problem_types': [],
            'zip_codes': [],
            'locations': [],
            'carriers': [],
            'leg_types': [],
            'ship_methods': [],
            'regions': [],
        }
        
        # Try new structure (config.data.vocab)
        if hasattr(config, 'data') and hasattr(config.data, 'vocab'):
            v = config.data.vocab
            vocab['event_types'] = getattr(v, 'event_types', []) or []
            vocab['problem_types'] = getattr(v, 'problem_types', []) or []
            vocab['zip_codes'] = getattr(v, 'zip_codes', []) or []
            vocab['locations'] = getattr(v, 'locations', []) or []
            vocab['carriers'] = getattr(v, 'carriers', []) or []
            vocab['leg_types'] = getattr(v, 'leg_types', []) or []
            vocab['ship_methods'] = getattr(v, 'ship_methods', []) or []
            vocab['regions'] = getattr(v, 'regions', []) or []
        
        # Fallback to old structure (config.data.event_types, etc.)
        elif hasattr(config, 'data'):
            vocab['event_types'] = getattr(config.data, 'event_types', []) or []
            vocab['problem_types'] = getattr(config.data, 'problem_types', []) or []
            vocab['zip_codes'] = getattr(config.data, 'zip_codes', []) or []
            vocab['locations'] = getattr(config.data, 'locations', []) or []
            vocab['carriers'] = getattr(config.data, 'carriers', []) or []
            vocab['leg_types'] = getattr(config.data, 'leg_types', []) or []
            vocab['ship_methods'] = getattr(config.data, 'ship_methods', []) or []
            vocab['regions'] = getattr(config.data, 'regions', []) or []
        
        # Convert all to strings
        for key in vocab:
            vocab[key] = [str(v) for v in vocab[key]]
        
        # Print summary
        print("=== Vocabulary from Config ===")
        for name, values in vocab.items():
            print(f"  {name}: {len(values)}")
        
        return vocab
    
    def enable_unknown_tracking(self, enable: bool = True):
        """Enable/disable tracking of unknown values."""
        self.track_unknowns = enable
        if enable:
            self.unknown_values = defaultdict(set)
            self.unknown_counts = defaultdict(int)
    
    def get_unknown_summary(self) -> Dict:
        """Get summary of all unknown values encountered."""
        return {
            'counts': dict(self.unknown_counts),
            'values': {k: list(v) for k, v in self.unknown_values.items()}
        }
    
    def print_unknown_summary(self):
        """Print a formatted summary of unknown values."""
        print("\n" + "=" * 70)
        print("UNKNOWN VALUES SUMMARY")
        print("=" * 70)
        
        if not self.unknown_counts:
            print("  No unknown values encountered.")
            return
        
        total_unknowns = sum(self.unknown_counts.values())
        print(f"\nTotal unknown encodings: {total_unknowns:,}")
        print("-" * 70)
        
        for category in sorted(self.unknown_counts.keys()):
            count = self.unknown_counts[category]
            values = self.unknown_values[category]
            
            print(f"\n{category}:")
            print(f"  Count: {count:,}")
            print(f"  Unique values ({len(values)}):")
            
            sorted_values = sorted(values, key=lambda x: str(x))
            if len(sorted_values) <= 20:
                for val in sorted_values:
                    print(f"    - '{val}'")
            else:
                for val in sorted_values[:10]:
                    print(f"    - '{val}'")
                print(f"    ... and {len(sorted_values) - 10} more")
        
        print("\n" + "=" * 70)
    
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
            
            if 'distance_miles' in df_dist.columns:
                dist_col = 'distance_miles'
                self.distance_unit = 'miles'
            elif 'distance_km' in df_dist.columns:
                dist_col = 'distance_km'
                self.distance_unit = 'km'
            else:
                print("Warning: No distance column found")
                return
            
            for _, row in df_dist.iterrows():
                loc1 = str(row['location_id_1']).strip()
                loc2 = str(row['location_id_2']).strip()
                
                try:
                    distance = float(row[dist_col])
                except (ValueError, TypeError):
                    continue
                
                if pd.isna(distance) or distance < 0:
                    continue
                
                self.distance_lookup[(loc1, loc2)] = distance
                self.distance_lookup[(loc2, loc1)] = distance
                
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
    
    def _get_location(self, event: Dict, events: List[Dict] = None, event_idx: int = None) -> str:
        """
        Get the primary location for an event.
        
        For DELIVERY events, the location is the sort_center of the previous event
        (which is the delivery station - the last facility before delivery).
        For all other events, the location is the event's own sort_center.
        
        Args:
            event: The current event dictionary
            events: List of all events in the lifecycle (needed for DELIVERY)
            event_idx: Index of current event in events list (needed for DELIVERY)
            
        Returns:
            Location string
        """
        event_type = str(event.get('event_type', ''))
        
        if event_type == 'DELIVERY':
            # Delivery station is the location of the previous event (last node before delivery)
            if events is not None and event_idx is not None and event_idx > 0:
                prev_event = events[event_idx - 1]
                prev_location = self._get_sort_center(prev_event)
                if prev_location != 'UNKNOWN':
                    return prev_location
            # Fallback to delivery_station field if available
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
        encoding = np.zeros(len(self.vocab['problem_types']), dtype=np.float32)
        problems = self._parse_problem_field(problem_value)
        
        if not problems:
            if 'NO_PROBLEM' in self.problem_type_to_idx:
                encoding[self.problem_type_to_idx['NO_PROBLEM']] = 1.0
        else:
            for problem in problems:
                if problem in self.problem_type_to_idx:
                    encoding[self.problem_type_to_idx[problem]] = 1.0
                elif self.track_unknowns:
                    self.unknown_values['problem_type'].add(problem)
                    self.unknown_counts['problem_type'] += 1
        
        return encoding
    
    def _safe_encode(self, encoder: LabelEncoder, value, category: str, 
                     default: str = 'UNKNOWN') -> int:
        """Safely encode a value, returning UNKNOWN index if not found."""
        original_value = value
        
        if value is None or value == '' or str(value) == 'nan':
            value = default
        else:
            value = str(value)
        
        if value not in encoder.classes_:
            if self.track_unknowns and value != default:
                self.unknown_values[category].add(original_value)
                self.unknown_counts[category] += 1
            value = default
        
        return int(encoder.transform([value])[0])
    
    def _get_plan_time_for_event(self, event: Dict, prev_event: Dict = None) -> Optional[str]:
        """Get plan_time for an event."""
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
    
    def _get_event_problem(self, event: Dict, events: List[Dict], 
                           event_idx: int) -> Tuple[np.ndarray, float]:
        """Get problem encoding for INDUCT/LINEHAUL events."""
        event_type = str(event.get('event_type', ''))
        
        if event_type not in ['INDUCT', 'LINEHAUL']:
            return np.zeros(len(self.vocab['problem_types']), dtype=np.float32), 0.0
        
        # Check direct problems (new format)
        direct_problems = self._parse_problem_field(event.get('problem'))
        if direct_problems:
            encoding = self._encode_problems(event.get('problem'))
            return encoding, 1.0
        
        # Fallback to EXIT problems (old format)
        current_sc = self._get_location(event, events, event_idx)
        if current_sc == 'UNKNOWN':
            return np.zeros(len(self.vocab['problem_types']), dtype=np.float32), 0.0
        
        for i in range(event_idx + 1, len(events)):
            next_event = events[i]
            next_type = str(next_event.get('event_type', ''))
            next_sc = self._get_sort_center(next_event)
            
            if next_type == 'EXIT' and next_sc == current_sc:
                exit_problems = self._parse_problem_field(next_event.get('problem'))
                if exit_problems:
                    encoding = self._encode_problems(next_event.get('problem'))
                    return encoding, 1.0
                else:
                    return np.zeros(len(self.vocab['problem_types']), dtype=np.float32), 0.0
            
            if next_sc != current_sc and next_sc != 'UNKNOWN':
                break
        
        return np.zeros(len(self.vocab['problem_types']), dtype=np.float32), 0.0
    
    # =========================================================================
    # FITTING (Uses Config Vocab - No Data Collection)
    # =========================================================================
    
    def fit(self, df: pd.DataFrame):
        """
        Fit encoders and scalers on training data.
        
        Encoders use vocabulary from config (no data collection).
        Scalers are fitted on actual data values.
        """
        print("\n=== Fitting Preprocessor ===")
        print("Using vocabulary from config (no data collection)")
        
        # === Fit encoders using config vocabulary ===
        special = ['PAD', 'UNKNOWN']
        
        self.event_type_encoder.fit(special + self.vocab['event_types'])
        self.location_encoder.fit(special + self.vocab['locations'])
        self.carrier_encoder.fit(special + self.vocab['carriers'])
        self.leg_type_encoder.fit(special + self.vocab['leg_types'])
        self.ship_method_encoder.fit(special + self.vocab['ship_methods'])
        self.postal_encoder.fit(special + self.vocab['zip_codes'])
        self.region_encoder.fit(special + self.vocab['regions'])
        
        # Set vocab sizes
        self.vocab_sizes = {
            'event_type': len(self.event_type_encoder.classes_),
            'location': len(self.location_encoder.classes_),
            'carrier': len(self.carrier_encoder.classes_),
            'leg_type': len(self.leg_type_encoder.classes_),
            'ship_method': len(self.ship_method_encoder.classes_),
            'postal': len(self.postal_encoder.classes_),
            'region': len(self.region_encoder.classes_),
        }
        
        print(f"\n=== Vocabulary Sizes ===")
        for name, size in self.vocab_sizes.items():
            print(f"  {name}: {size}")
        print(f"  problem_types: {len(self.vocab['problem_types'])}")
        
        # === Fit scalers on actual data values ===
        print("\nFitting scalers on data...")
        
        time_since_prev_vals = []
        dwelling_vals = []
        time_delta_vals = []
        elapsed_vals = []
        distance_vals = []
        label_vals = []
        
        for _, row in df.iterrows():
            events = row['events']
            event_times = []
            first_event_time = None
            
            for i, event in enumerate(events):
                event_time = self._parse_datetime(event['event_time'])
                if event_time is None:
                    continue
                
                if first_event_time is None:
                    first_event_time = event_time
                
                event_times.append(event_time)
                prev_event = events[i-1] if i > 0 else None
                
                # Elapsed time
                elapsed = (event_time - first_event_time).total_seconds() / 3600
                elapsed_vals.append([elapsed])
                
                # Time since previous
                if i > 0 and len(event_times) > 1:
                    time_since_prev = (event_time - event_times[-2]).total_seconds() / 3600
                    time_since_prev_vals.append([time_since_prev])
                
                # Dwelling time
                dwelling = (event.get('dwelling_seconds', 0) or 0) / 3600
                dwelling_vals.append([dwelling])
                
                # Time vs plan
                plan_time = self._get_plan_time_for_event(event, prev_event)
                plan_dt = self._parse_datetime(plan_time)
                if plan_dt:
                    time_vs_plan = (event_time - plan_dt).total_seconds() / 3600
                    time_vs_plan = max(-720, min(time_vs_plan, 720))
                    time_delta_vals.append([time_vs_plan])
                
                # Time until next plan
                if i < len(events) - 1:
                    next_event = events[i + 1]
                    next_plan_time = self._get_plan_time_for_event(next_event, event)
                    next_plan_dt = self._parse_datetime(next_plan_time)
                    if next_plan_dt:
                        time_until_plan = (next_plan_dt - event_time).total_seconds() / 3600
                        time_until_plan = max(-720, min(time_until_plan, 720))
                        time_delta_vals.append([time_until_plan])
                
                # Label
                if i < len(events) - 1:
                    next_time = self._parse_datetime(events[i+1]['event_time'])
                    if next_time:
                        label = (next_time - event_time).total_seconds() / 3600
                        label_vals.append([label])
            
            # Distances
            for i in range(len(events) - 1):
                from_loc = self._get_location(events[i], events, i)
                to_loc = self._get_location(events[i + 1], events, i + 1)
                dist, has_dist = self._get_distance(from_loc, to_loc)
                if has_dist and dist > 0:
                    distance_vals.append([dist])
        
        # Fit scalers
        self._fit_scaler(self.time_since_prev_scaler, time_since_prev_vals, 'time_since_prev')
        self._fit_scaler(self.dwelling_time_scaler, dwelling_vals, 'dwelling')
        self._fit_scaler(self.time_delta_scaler, time_delta_vals, 'time_delta')
        self._fit_scaler(self.elapsed_time_scaler, elapsed_vals, 'elapsed')
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
            'time_delta': self.time_delta_scaler,
            'elapsed_time': self.elapsed_time_scaler,
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
                                      first_event_time: datetime,
                                      events: List[Dict], event_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract OBSERVABLE features for an event."""
        event_type = str(event.get('event_type', 'UNKNOWN'))
        num_events = len(events)
        
        plan_time = self._get_plan_time_for_event(event, prev_event)
        plan_dt = self._parse_datetime(plan_time)
        
        has_plan = 0.0
        time_until_plan = 0.0
        
        if plan_dt and reference_time:
            has_plan = 1.0
            time_until_plan = (plan_dt - reference_time).total_seconds() / 3600
            time_until_plan = max(-720, min(time_until_plan, 720))
        
        time_until_plan_scaled = self.time_delta_scaler.transform([[time_until_plan]])[0, 0]
        elapsed_hours = (reference_time - first_event_time).total_seconds() / 3600 if reference_time else 0.0
        elapsed_scaled = self.elapsed_time_scaler.transform([[elapsed_hours]])[0, 0]
        
        time_ref = plan_dt if plan_dt else reference_time
        if time_ref is None:
            time_ref = first_event_time
        
        time_features = np.array([
            time_ref.hour + time_ref.minute / 60.0,
            time_ref.weekday(),
            time_ref.day,
            time_ref.month,
            elapsed_scaled,
            time_until_plan_scaled,
        ], dtype=np.float32)
        
        is_delivery = 1.0 if event_type == 'DELIVERY' else 0.0
        position = event_idx / max(1, num_events - 1)
        
        other_features = np.array([
            is_delivery,
            position,
            has_plan,
        ], dtype=np.float32)
        
        return time_features, other_features
    
    def _extract_realized_features(self, event: Dict, prev_event: Dict,
                                    prev_time: Optional[datetime],
                                    first_event_time: datetime,
                                    events: List[Dict], event_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract REALIZED features for an event."""
        event_type = str(event.get('event_type', ''))
        event_time = self._parse_datetime(event['event_time'])
        
        time_since_prev_scaled = 0.0
        if prev_time and event_time:
            time_since_prev = (event_time - prev_time).total_seconds() / 3600
            time_since_prev_scaled = self.time_since_prev_scaler.transform([[time_since_prev]])[0, 0]
        
        dwelling = (event.get('dwelling_seconds', 0) or 0) / 3600
        dwelling_scaled = self.dwelling_time_scaler.transform([[dwelling]])[0, 0]
        has_dwelling = 1.0 if dwelling > 0 else 0.0
        
        plan_time = self._get_plan_time_for_event(event, prev_event)
        plan_dt = self._parse_datetime(plan_time)
        time_vs_plan = 0.0
        if plan_dt and event_time:
            time_vs_plan = (event_time - plan_dt).total_seconds() / 3600
            time_vs_plan = max(-720, min(time_vs_plan, 720))
        time_vs_plan_scaled = self.time_delta_scaler.transform([[time_vs_plan]])[0, 0]
        
        elapsed_hours = (event_time - first_event_time).total_seconds() / 3600 if event_time and first_event_time else 0.0
        elapsed_scaled = self.elapsed_time_scaler.transform([[elapsed_hours]])[0, 0]
        
        if event_time:
            time_features = np.array([
                event_time.hour + event_time.minute / 60.0,
                event_time.weekday(),
                event_time.day,
                event_time.month,
                elapsed_scaled,
                time_vs_plan_scaled,
            ], dtype=np.float32)
        else:
            time_features = np.zeros(6, dtype=np.float32)
        
        missort = 0.0
        if event_type in ['INDUCT', 'LINEHAUL']:
            missort = float(event.get('missort', False))
        
        problem_encoding, has_problem = self._get_event_problem(event, events, event_idx)
        
        other_features = np.concatenate([
            [time_since_prev_scaled],
            [dwelling_scaled],
            [has_dwelling],
            [missort],
            [has_problem],
            problem_encoding,
        ]).astype(np.float32)
        
        return time_features, other_features
    
    def _extract_edge_features(self, source_event: Dict, target_event: Dict,
                                source_time: datetime,
                                events: List[Dict] = None,
                                source_idx: int = None,
                                target_idx: int = None) -> np.ndarray:
        """
        Extract edge features between two events.
        
        Args:
            source_event: Source event dictionary
            target_event: Target event dictionary
            source_time: Timestamp of source event
            events: Full list of events (for proper location resolution)
            source_idx: Index of source event in events list
            target_idx: Index of target event in events list
            
        Returns:
            Edge feature array
        """
        source_loc = self._get_location(source_event, events, source_idx)
        target_loc = self._get_location(target_event, events, target_idx)
        
        distance, has_distance = self._get_distance(source_loc, target_loc)
        distance_scaled = self.edge_distance_scaler.transform([[distance]])[0, 0]
        
        same_location = float(source_loc == target_loc and source_loc != 'UNKNOWN')
        
        source_region = self._get_region(source_loc)
        target_region = self._get_region(target_loc)
        cross_region = float(source_region != target_region and 
                            source_region != 'UNKNOWN' and target_region != 'UNKNOWN')
        
        edge_features = np.array([
            distance_scaled,
            float(has_distance),
            same_location,
            cross_region,
            source_time.hour + source_time.minute / 60.0,
            source_time.weekday(),
            source_time.day,
            source_time.month,
        ], dtype=np.float32)
        
        return edge_features
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    def process_lifecycle(self, package_data: Dict, return_labels: bool = True) -> Dict:
        """Process a package lifecycle with Time2Vec-ready features."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before processing")
        
        events = package_data['events']
        num_events = len(events)
        
        if num_events < 1:
            raise ValueError("Package must have at least 1 event")
        
        event_times = []
        for event in events:
            et = self._parse_datetime(event['event_time'])
            if et is None:
                raise ValueError("Invalid event_time")
            event_times.append(et)
        
        first_event_time = event_times[0]
        
        node_observable_time = []
        node_observable_other = []
        node_realized_time = []
        node_realized_other = []
        
        node_categorical_indices = {
            'event_type': [], 'location': [], 'postal': [], 'region': [],
            'carrier': [], 'leg_type': [], 'ship_method': [],
        }
        
        for i, event in enumerate(events):
            event_type = str(event.get('event_type', 'UNKNOWN'))
            prev_event = events[i-1] if i > 0 else None
            prev_time = event_times[i-1] if i > 0 else None
            reference_time = prev_time if i > 0 else event_times[0]
            
            obs_time, obs_other = self._extract_observable_features(
                event, prev_event, reference_time, first_event_time, events, i
            )
            node_observable_time.append(obs_time)
            node_observable_other.append(obs_other)
            
            real_time, real_other = self._extract_realized_features(
                event, prev_event, prev_time, first_event_time, events, i
            )
            node_realized_time.append(real_time)
            node_realized_other.append(real_other)
            
            location = self._get_location(event, events, i)
            postal = self._get_delivery_postal(event)
            region = self._get_region(location)
            
            node_categorical_indices['event_type'].append(
                self._safe_encode(self.event_type_encoder, event_type, 'event_type')
            )
            node_categorical_indices['location'].append(
                self._safe_encode(self.location_encoder, location, 'location')
            )
            node_categorical_indices['postal'].append(
                self._safe_encode(self.postal_encoder, postal, 'postal')
            )
            node_categorical_indices['region'].append(
                self._safe_encode(self.region_encoder, region, 'region')
            )
            node_categorical_indices['carrier'].append(
                self._safe_encode(self.carrier_encoder, event.get('carrier_id'), 'carrier')
            )
            node_categorical_indices['leg_type'].append(
                self._safe_encode(self.leg_type_encoder, event.get('leg_type'), 'leg_type')
            )
            node_categorical_indices['ship_method'].append(
                self._safe_encode(self.ship_method_encoder, event.get('ship_method'), 'ship_method')
            )
        
        node_observable_time = np.array(node_observable_time, dtype=np.float32)
        node_observable_other = np.array(node_observable_other, dtype=np.float32)
        node_realized_time = np.array(node_realized_time, dtype=np.float32)
        node_realized_other = np.array(node_realized_other, dtype=np.float32)
        
        for key in node_categorical_indices:
            node_categorical_indices[key] = np.array(node_categorical_indices[key], dtype=np.int64)
        
        package_features = np.array([
            package_data.get('weight', 0) or 0,
            package_data.get('length', 0) or 0,
            package_data.get('width', 0) or 0,
            package_data.get('height', 0) or 0
        ], dtype=np.float32).reshape(1, -1)
        package_features_scaled = self.package_feature_scaler.transform(package_features).flatten()
        
        edge_index = []
        edge_features = []
        
        for i in range(num_events - 1):
            edge_index.append([i, i + 1])
            edge_feat = self._extract_edge_features(
                events[i], events[i + 1], event_times[i],
                events=events, source_idx=i, target_idx=i + 1
            )
            edge_features.append(edge_feat)
        
        if edge_index:
            edge_index = np.array(edge_index, dtype=np.int64).T
            edge_features = np.array(edge_features, dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 8), dtype=np.float32)
        
        result = {
            'node_observable_time': node_observable_time,
            'node_realized_time': node_realized_time,
            'node_observable_other': node_observable_other,
            'node_realized_other': node_realized_other,
            'node_categorical_indices': node_categorical_indices,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'package_features': package_features_scaled,
            'package_categorical': {
                'source_postal': self._safe_encode(
                    self.postal_encoder, package_data.get('source_postal'), 'source_postal'
                ),
                'dest_postal': self._safe_encode(
                    self.postal_encoder, package_data.get('dest_postal'), 'dest_postal'
                ),
            },
            'num_nodes': num_events,
            'package_id': package_data.get('package_id', 'unknown'),
        }
        
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
        
        return {
            'vocab_sizes': self.vocab_sizes.copy(),
            'observable_time_dim': 6,
            'observable_other_dim': 3,
            'realized_time_dim': 6,
            'realized_other_dim': 5 + len(self.vocab['problem_types']),
            'edge_dim': 8,
            'package_dim': 4,
            'num_problem_types': len(self.vocab['problem_types']),
        }
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for embedding layers."""
        return self.vocab_sizes.copy()
    
    def get_postal_code_count(self) -> int:
        """Get the number of postal codes in the vocabulary."""
        return self.vocab_sizes.get('postal', 0)
    
    def get_zip_codes(self) -> List[str]:
        """Get the list of zip codes used."""
        if self.fitted:
            return [c for c in self.postal_encoder.classes_ if c not in ['PAD', 'UNKNOWN']]
        return self.vocab['zip_codes'].copy()
    
    def get_regions(self) -> List[str]:
        """Get the list of regions used."""
        if self.fitted:
            return [c for c in self.region_encoder.classes_ if c not in ['PAD', 'UNKNOWN']]
        return self.vocab['regions'].copy()
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    
    def save(self, path: str):
        """Save preprocessor to file (local or S3)."""
        if path.startswith('s3://'):
            import boto3
            
            buffer = io.BytesIO()
            pickle.dump(self, buffer)
            buffer.seek(0)
            
            path_clean = path.replace('s3://', '')
            bucket, key = path_clean.split('/', 1)
            boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            print(f"Preprocessor saved to {path}")
        else:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'PackageLifecyclePreprocessor':
        """Load preprocessor from file (local or S3)."""
        if path.startswith('s3://'):
            import boto3
            
            path_clean = path.replace('s3://', '')
            bucket, key = path_clean.split('/', 1)
            response = boto3.client('s3').get_object(Bucket=bucket, Key=key)
            buffer = io.BytesIO(response['Body'].read())
            preprocessor = pickle.load(buffer)
            print(f"Preprocessor loaded from {path}")
            return preprocessor
        else:
            with open(path, 'rb') as f:
                preprocessor = pickle.load(f)
            print(f"Preprocessor loaded from {path}")
            return preprocessor
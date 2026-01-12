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
    
    Time Features Output:
    - Raw time values for Time2Vec (hour, dow, dom, month, elapsed, time_delta)
    - These are processed by the model's Time2Vec layers
    
    Feature Types:
    1. OBSERVABLE features (known BEFORE event happens)
    2. REALIZED features (only known AFTER event happens)
    
    Problem Handling:
    - NEW FORMAT: Problems stored directly on INDUCT/LINEHAUL events
    - OLD FORMAT: Problems stored on EXIT events (backward compatibility)
    """
    
    def __init__(self, config, distance_df: pd.DataFrame = None, distance_file_path: str = None):
        """
        Args:
            config: Configuration object with data.event_types, data.problem_types, and data.zip_codes
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
        self.time_delta_scaler = StandardScaler()  # For time_until_plan and time_vs_plan
        self.elapsed_time_scaler = StandardScaler()
        self.edge_distance_scaler = StandardScaler()
        self.label_time_scaler = StandardScaler()
        self.package_feature_scaler = StandardScaler()
        
        # Event and problem types from config
        self.event_types = config.data.event_types
        self.problem_types = config.data.problem_types
        self.zip_codes = self._get_zip_codes_from_config(config)
        self.problem_type_to_idx = {}
        
        self.fitted = False
        self.vocab_sizes = {}
        
        # === Unknown value tracking ===
        self.track_unknowns = False
        self.unknown_values = defaultdict(set)
        self.unknown_counts = defaultdict(int)
        
        # === Feature dimensions (updated for Time2Vec) ===
        # Observable: [time_features(6), is_delivery(1), position(1), has_plan(1)] = 9
        # Realized: [time_features(6), time_since_prev(1), dwelling(2), missort(1), problem(1+N)] = 11 + N
        self.observable_time_dim = 6  # hour, dow, dom, month, elapsed, time_until_plan
        self.realized_time_dim = 6    # hour, dow, dom, month, elapsed, time_vs_plan
    
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
    
    def _get_zip_codes_from_config(self, config) -> List[str]:
        """Extract zip_codes from config with multiple fallback options."""
        zip_codes = []
        
        if hasattr(config, 'data'):
            if hasattr(config.data, 'zip_codes'):
                zip_codes = config.data.zip_codes
            elif isinstance(config.data, dict) and 'zip_codes' in config.data:
                zip_codes = config.data['zip_codes']
        elif hasattr(config, 'zip_codes'):
            zip_codes = config.zip_codes
        elif isinstance(config, dict):
            if 'data' in config and 'zip_codes' in config['data']:
                zip_codes = config['data']['zip_codes']
            elif 'zip_codes' in config:
                zip_codes = config['zip_codes']
        
        if zip_codes:
            zip_codes = [str(z) for z in zip_codes]
            print(f"Loaded {len(zip_codes)} zip codes from config")
        else:
            print("Warning: No zip_codes found in config, will collect from data during fit()")
        
        return zip_codes
    
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
    
    def _extract_raw_time_features(self, dt: datetime, elapsed_hours: float, 
                                    time_delta: float) -> np.ndarray:
        """
        Extract raw time features for Time2Vec.
        
        Args:
            dt: The datetime to extract features from
            elapsed_hours: Hours since journey start
            time_delta: Time difference (until_plan or vs_plan) in hours
        
        Returns:
            Array of [hour, dow, dom, month, elapsed, time_delta]
        """
        return np.array([
            dt.hour + dt.minute / 60.0,  # Hour as float (0-24)
            dt.weekday(),                 # Day of week (0-6)
            dt.day,                       # Day of month (1-31)
            dt.month,                     # Month (1-12)
            elapsed_hours,                # Hours since first event
            time_delta,                   # Time delta (scaled separately)
        ], dtype=np.float32)
    
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
    
    def _get_event_problem(self, event: Dict, events: List[Dict], 
                           event_idx: int) -> Tuple[np.ndarray, float]:
        """
        For INDUCT/LINEHAUL events, get problem encoding.
        
        Handles both data formats:
        - NEW FORMAT: Problems are stored directly on INDUCT/LINEHAUL events
        - OLD FORMAT: Problems are stored on EXIT events (backward compatibility)
        
        Logic:
        1. If event is not INDUCT/LINEHAUL, return no problems
        2. Check if INDUCT/LINEHAUL has problems directly → use them (new format)
        3. Otherwise, look for corresponding EXIT and use its problems (old format)
        
        Returns:
            Tuple of (problem_encoding, has_problem_flag)
        """
        event_type = str(event.get('event_type', ''))
        
        # Only process INDUCT and LINEHAUL events
        if event_type not in ['INDUCT', 'LINEHAUL']:
            return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
        
        # === STEP 1: Check if this event has problems directly (NEW FORMAT) ===
        direct_problems = self._parse_problem_field(event.get('problem'))
        if direct_problems:
            # Problems found directly on INDUCT/LINEHAUL - use them
            encoding = self._encode_problems(event.get('problem'))
            return encoding, 1.0
        
        # === STEP 2: Fallback to EXIT problems (OLD FORMAT) ===
        current_sc = self._get_sort_center(event)
        if current_sc == 'UNKNOWN':
            return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
        
        # Look for next EXIT at same sort center
        for i in range(event_idx + 1, len(events)):
            next_event = events[i]
            next_type = str(next_event.get('event_type', ''))
            next_sc = self._get_sort_center(next_event)
            
            if next_type == 'EXIT' and next_sc == current_sc:
                # Found corresponding EXIT event
                exit_problems = self._parse_problem_field(next_event.get('problem'))
                if exit_problems:
                    # EXIT has problems - old format
                    encoding = self._encode_problems(next_event.get('problem'))
                    return encoding, 1.0
                else:
                    # EXIT exists but no problems
                    return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
            
            # Stop searching if we've moved to a different sort center
            if next_sc != current_sc and next_sc != 'UNKNOWN':
                break
        
        # No problems found in either format
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
        all_regions = set()
        all_postals = set()
        
        if self.zip_codes:
            all_postals.update(self.zip_codes)
            print(f"Using {len(self.zip_codes)} postal codes from config.data.zip_codes")
        else:
            print("Collecting postal codes from data (no zip_codes in config)")
            for _, row in df.iterrows():
                source_postal = row.get('source_postal')
                dest_postal = row.get('dest_postal')
                if source_postal and str(source_postal) != 'nan':
                    all_postals.add(str(source_postal))
                if dest_postal and str(dest_postal) != 'nan':
                    all_postals.add(str(dest_postal))
                
                events = row['events']
                for event in events:
                    postal = self._get_delivery_postal(event)
                    if postal != 'UNKNOWN':
                        all_postals.add(postal)
        
        all_regions.update(self.region_lookup.values())
        
        for _, row in df.iterrows():
            events = row['events']
            for event in events:
                loc = self._get_location(event)
                if loc != 'UNKNOWN':
                    all_locations.add(loc)
                    region = self._get_region(loc)
                    if region != 'UNKNOWN':
                        all_regions.add(region)
                
                if event.get('carrier_id'):
                    all_carriers.add(str(event['carrier_id']))
                if event.get('leg_type'):
                    all_leg_types.add(str(event['leg_type']))
                if event.get('ship_method'):
                    all_ship_methods.add(str(event['ship_method']))
        
        # Fit encoders with special tokens (PAD=0, UNKNOWN=1)
        special = ['PAD', 'UNKNOWN']
        self.event_type_encoder.fit(special + self.event_types)
        self.location_encoder.fit(special + sorted(list(all_locations - {'UNKNOWN'})))
        self.carrier_encoder.fit(special + sorted(list(all_carriers - {'UNKNOWN'})))
        self.leg_type_encoder.fit(special + sorted(list(all_leg_types - {'UNKNOWN'})))
        self.ship_method_encoder.fit(special + sorted(list(all_ship_methods - {'UNKNOWN'})))
        self.postal_encoder.fit(special + sorted(list(all_postals - {'UNKNOWN'})))
        self.region_encoder.fit(special + sorted(list(all_regions - {'UNKNOWN'})))
        
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
        time_delta_vals = []  # Combined: time_until_plan and time_vs_plan
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
                
                # Time vs plan (realized)
                plan_time = self._get_plan_time_for_event(event, prev_event)
                plan_dt = self._parse_datetime(plan_time)
                if plan_dt:
                    time_vs_plan = (event_time - plan_dt).total_seconds() / 3600
                    time_vs_plan = max(-720, min(time_vs_plan, 720))
                    time_delta_vals.append([time_vs_plan])
                
                # Time until next plan (observable)
                if i < len(events) - 1:
                    next_event = events[i + 1]
                    next_plan_time = self._get_plan_time_for_event(next_event, event)
                    next_plan_dt = self._parse_datetime(next_plan_time)
                    if next_plan_dt:
                        time_until_plan = (next_plan_dt - event_time).total_seconds() / 3600
                        time_until_plan = max(-720, min(time_until_plan, 720))
                        time_delta_vals.append([time_until_plan])
                
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
        """
        Extract OBSERVABLE features for an event.
        
        Returns:
            Tuple of (time_features, other_features)
            - time_features: [6] for Time2Vec (hour, dow, dom, month, elapsed, time_until_plan)
            - other_features: [3] (is_delivery, position, has_plan)
        """
        event_type = str(event.get('event_type', 'UNKNOWN'))
        num_events = len(events)
        
        plan_time = self._get_plan_time_for_event(event, prev_event)
        plan_dt = self._parse_datetime(plan_time)
        
        # Calculate time until plan
        has_plan = 0.0
        time_until_plan = 0.0
        
        if plan_dt and reference_time:
            has_plan = 1.0
            time_until_plan = (plan_dt - reference_time).total_seconds() / 3600
            time_until_plan = max(-720, min(time_until_plan, 720))
        
        # Scale time_until_plan
        time_until_plan_scaled = self.time_delta_scaler.transform([[time_until_plan]])[0, 0]
        
        # Calculate elapsed time
        elapsed_hours = (reference_time - first_event_time).total_seconds() / 3600 if reference_time else 0.0
        elapsed_scaled = self.elapsed_time_scaler.transform([[elapsed_hours]])[0, 0]
        
        # Use plan_dt for time features if available, else reference_time
        time_ref = plan_dt if plan_dt else reference_time
        if time_ref is None:
            time_ref = first_event_time
        
        # Raw time features for Time2Vec
        time_features = np.array([
            time_ref.hour + time_ref.minute / 60.0,  # hour (0-24)
            time_ref.weekday(),                       # day of week (0-6)
            time_ref.day,                             # day of month (1-31)
            time_ref.month,                           # month (1-12)
            elapsed_scaled,                           # elapsed (scaled)
            time_until_plan_scaled,                   # time until plan (scaled)
        ], dtype=np.float32)
        
        # Other observable features
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
        """
        Extract REALIZED features for an event.
        
        Returns:
            Tuple of (time_features, other_features)
            - time_features: [6] for Time2Vec (hour, dow, dom, month, elapsed, time_vs_plan)
            - other_features: [4 + num_problems] (time_since_prev, dwelling, has_dwelling, missort, has_problem, problems...)
        """
        event_type = str(event.get('event_type', ''))
        event_time = self._parse_datetime(event['event_time'])
        
        # Time since previous
        time_since_prev_scaled = 0.0
        if prev_time and event_time:
            time_since_prev = (event_time - prev_time).total_seconds() / 3600
            time_since_prev_scaled = self.time_since_prev_scaler.transform([[time_since_prev]])[0, 0]
        
        # Dwelling
        dwelling = (event.get('dwelling_seconds', 0) or 0) / 3600
        dwelling_scaled = self.dwelling_time_scaler.transform([[dwelling]])[0, 0]
        has_dwelling = 1.0 if dwelling > 0 else 0.0
        
        # Time vs plan
        plan_time = self._get_plan_time_for_event(event, prev_event)
        plan_dt = self._parse_datetime(plan_time)
        time_vs_plan = 0.0
        if plan_dt and event_time:
            time_vs_plan = (event_time - plan_dt).total_seconds() / 3600
            time_vs_plan = max(-720, min(time_vs_plan, 720))
        time_vs_plan_scaled = self.time_delta_scaler.transform([[time_vs_plan]])[0, 0]
        
        # Elapsed time
        elapsed_hours = (event_time - first_event_time).total_seconds() / 3600 if event_time and first_event_time else 0.0
        elapsed_scaled = self.elapsed_time_scaler.transform([[elapsed_hours]])[0, 0]
        
        # Raw time features for Time2Vec (using actual event time)
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
        
        # Missort
        missort = 0.0
        if event_type in ['INDUCT', 'LINEHAUL']:
            missort = float(event.get('missort', False))
        
        # Problem encoding - handles both new and old data formats
        problem_encoding, has_problem = self._get_event_problem(event, events, event_idx)
        
        # Other realized features
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
                                source_time: datetime) -> np.ndarray:
        """Extract edge features."""
        source_loc = self._get_location(source_event)
        target_loc = self._get_location(target_event)
        
        distance, has_distance = self._get_distance(source_loc, target_loc)
        distance_scaled = self.edge_distance_scaler.transform([[distance]])[0, 0]
        
        same_location = float(source_loc == target_loc and source_loc != 'UNKNOWN')
        
        source_region = self._get_region(source_loc)
        target_region = self._get_region(target_loc)
        cross_region = float(source_region != target_region and 
                            source_region != 'UNKNOWN' and target_region != 'UNKNOWN')
        
        # Raw time features for edge (source time)
        edge_features = np.array([
            distance_scaled,
            float(has_distance),
            same_location,
            cross_region,
            source_time.hour + source_time.minute / 60.0,  # hour
            source_time.weekday(),                          # dow
            source_time.day,                                # dom
            source_time.month,                              # month
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
        
        # Feature arrays
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
            
            # Observable features
            obs_time, obs_other = self._extract_observable_features(
                event, prev_event, reference_time, first_event_time, events, i
            )
            node_observable_time.append(obs_time)
            node_observable_other.append(obs_other)
            
            # Realized features
            real_time, real_other = self._extract_realized_features(
                event, prev_event, prev_time, first_event_time, events, i
            )
            node_realized_time.append(real_time)
            node_realized_other.append(real_other)
            
            # Categorical indices
            location = self._get_location(event)
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
        
        # Convert to arrays
        node_observable_time = np.array(node_observable_time, dtype=np.float32)
        node_observable_other = np.array(node_observable_other, dtype=np.float32)
        node_realized_time = np.array(node_realized_time, dtype=np.float32)
        node_realized_other = np.array(node_realized_other, dtype=np.float32)
        
        for key in node_categorical_indices:
            node_categorical_indices[key] = np.array(node_categorical_indices[key], dtype=np.int64)
        
        # Package features
        package_features = np.array([
            package_data.get('weight', 0) or 0,
            package_data.get('length', 0) or 0,
            package_data.get('width', 0) or 0,
            package_data.get('height', 0) or 0
        ], dtype=np.float32).reshape(1, -1)
        package_features_scaled = self.package_feature_scaler.transform(package_features).flatten()
        
        # Edge features
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
        
        result = {
            # Time features (for Time2Vec)
            'node_observable_time': node_observable_time,    # [num_nodes, 6]
            'node_realized_time': node_realized_time,        # [num_nodes, 6]
            
            # Other features
            'node_observable_other': node_observable_other,  # [num_nodes, 3]
            'node_realized_other': node_realized_other,      # [num_nodes, 5 + num_problems]
            
            # Categorical indices
            'node_categorical_indices': node_categorical_indices,
            
            # Edge features
            'edge_index': edge_index,
            'edge_features': edge_features,
            
            # Package features
            'package_features': package_features_scaled,
            'package_categorical': {
                'source_postal': self._safe_encode(
                    self.postal_encoder, package_data.get('source_postal'), 'source_postal'
                ),
                'dest_postal': self._safe_encode(
                    self.postal_encoder, package_data.get('dest_postal'), 'dest_postal'
                ),
            },
            
            # Metadata
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
            'realized_other_dim': 5 + len(self.problem_types),
            'edge_dim': 8,
            'package_dim': 4,
            'num_problem_types': len(self.problem_types),
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
        return self.zip_codes.copy()
    
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
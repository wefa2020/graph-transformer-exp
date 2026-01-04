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
    Preprocess package lifecycle data for graph transformer with:
    - Categorical embeddings
    - Consistent time scaling
    - Lookahead features (next event info)
    - Enhanced edge features with distance and region from CSV
    - Problem features for INDUCT/LINEHAUL (from EXIT at same sort center)
    - Plan time features for each node
    - Proper handling of sort centers, delivery stations, and postal codes
    
    Plan Time Logic:
    - EXIT: plan_time = previous event's CPT (previous is INDUCT or LINEHAUL)
    - Other events: use their own plan_time
    
    Postal Code Logic:
    - Only used for to_postal when predicting DELIVERY time
    """
    
    def __init__(self, config, distance_df: pd.DataFrame = None, distance_file_path: str = None):
        """
        Args:
            config: Configuration object with data.event_types and data.problem_types
            distance_df: Pre-loaded DataFrame with distance data (for distributed training)
            distance_file_path: Path to location_distances_complete.csv (used if distance_df not provided)
        """
        self.config = config
        
        # Distance and region lookup
        self.distance_lookup = {}
        self.region_lookup = {}
        self.distance_unit = 'miles'
        
        # Store for serialization - only store file path, not the DataFrame
        self.distance_file_path = distance_file_path
        
        # Load distance data from DataFrame or file
        self._load_distance_data(distance_df=distance_df)
        
        # === Categorical Encoders ===
        self.event_type_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()  # For sort centers and delivery stations
        self.carrier_encoder = LabelEncoder()
        self.leg_type_encoder = LabelEncoder()
        self.ship_method_encoder = LabelEncoder()
        self.postal_encoder = LabelEncoder()  # For postal codes (DELIVERY only)
        self.region_encoder = LabelEncoder()
        
        # === Time Scalers (all continuous time features) ===
        self.time_since_prev_scaler = StandardScaler()
        self.dwelling_time_scaler = StandardScaler()
        self.plan_time_diff_scaler = StandardScaler()
        self.label_time_scaler = StandardScaler()
        
        # === Edge Feature Scalers ===
        self.edge_distance_scaler = StandardScaler()
        self.edge_next_plan_time_scaler = StandardScaler()
        
        # === Lookahead Feature Scalers ===
        self.next_plan_time_diff_scaler = StandardScaler()
        
        # === Package Feature Scaler ===
        self.package_feature_scaler = StandardScaler()
        self.plan_time_diff_scaler = StandardScaler()
        self.planned_remaining_scaler = StandardScaler()  # For node feature
        self.planned_duration_scaler = StandardScaler()   # For node feature
        self.planned_transit_scaler = StandardScaler()    # NEW: For planned transit time
        
        self.event_types = config.data.event_types
        self.problem_types = config.data.problem_types
        self.problem_type_to_idx = {}
        self.fitted = False
        
        self.vocab_sizes = {}
    
    # ==================== Distance Data Loading ====================
    
    def _load_distance_data(self, distance_df: pd.DataFrame = None):
        """
        Load distance and region lookup tables.
        
        Args:
            distance_df: Pre-loaded DataFrame (takes priority if provided)
        """
        df_dist = None
        
        # Priority 1: Use provided DataFrame
        if distance_df is not None:
            df_dist = distance_df
            print("Using provided distance DataFrame")
        
        # Priority 2: Load from file path
        elif self.distance_file_path is not None:
            if os.path.exists(self.distance_file_path):
                try:
                    df_dist = pd.read_csv(self.distance_file_path)
                    print(f"Loaded distance data from: {self.distance_file_path}")
                except Exception as e:
                    print(f"Error loading distance file: {e}")
            else:
                print(f"Warning: Distance file not found at {self.distance_file_path}")
        
        # Priority 3: Try default path
        else:
            default_path = os.path.join('data', 'location_distances_complete.csv')
            if os.path.exists(default_path):
                try:
                    df_dist = pd.read_csv(default_path)
                    self.distance_file_path = default_path
                    print(f"Loaded distance data from default path: {default_path}")
                except Exception as e:
                    print(f"Error loading default distance file: {e}")
            else:
                print("Warning: No distance data available. Distance features will be set to 0")
        
        # If no data available, return early
        if df_dist is None:
            print("Distance features will be set to 0")
            return
        
        # Process the DataFrame
        self._process_distance_dataframe(df_dist)
    
    def _process_distance_dataframe(self, df_dist: pd.DataFrame):
        """
        Process distance DataFrame and populate lookup tables.
        
        Args:
            df_dist: DataFrame with distance data
        """
        try:
            # Validate columns
            required_cols = ['location_id_1', 'location_id_2']
            if not all(col in df_dist.columns for col in required_cols):
                print(f"Warning: Expected columns {required_cols} not found")
                print(f"Found columns: {df_dist.columns.tolist()}")
                return
            
            # Determine distance column (prefer miles for US logistics)
            if 'distance_miles' in df_dist.columns:
                dist_col = 'distance_miles'
                self.distance_unit = 'miles'
            elif 'distance_km' in df_dist.columns:
                dist_col = 'distance_km'
                self.distance_unit = 'km'
            else:
                print("Warning: No distance column found")
                return
            
            print(f"Processing distances using '{dist_col}' column")
            
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
                
                # Store bidirectional distances
                self.distance_lookup[(loc1, loc2)] = distance
                self.distance_lookup[(loc2, loc1)] = distance
                
                # Store region info
                if 'super_region_1' in df_dist.columns:
                    region1 = row.get('super_region_1')
                    if pd.notna(region1) and str(region1).strip():
                        self.region_lookup[loc1] = str(region1).strip()
                
                if 'super_region_2' in df_dist.columns:
                    region2 = row.get('super_region_2')
                    if pd.notna(region2) and str(region2).strip():
                        self.region_lookup[loc2] = str(region2).strip()
            
            unique_pairs = len(self.distance_lookup) // 2
            print(f"Loaded {unique_pairs} unique distance pairs")
            print(f"Distance unit: {self.distance_unit}")
            print(f"Locations with region info: {len(self.region_lookup)}")
            
            if self.distance_lookup:
                distances = list(set(self.distance_lookup.values()))
                print(f"Distance stats ({self.distance_unit}):")
                print(f"  Min: {min(distances):.2f}, Max: {max(distances):.2f}")
                print(f"  Mean: {np.mean(distances):.2f}, Median: {np.median(distances):.2f}")
            
            if self.region_lookup:
                regions = set(self.region_lookup.values())
                print(f"Regions found: {sorted(regions)}")
                
        except Exception as e:
            print(f"Error processing distance data: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== Location Utility Methods ====================
    
    def _get_distance(self, from_location: str, to_location: str) -> Tuple[float, bool]:
        """
        Get distance between two locations from lookup
        
        Returns:
            Tuple of (distance, has_distance_flag)
        """
        if not self.distance_lookup:
            return 0.0, False
        
        from_loc = str(from_location).strip()
        to_loc = str(to_location).strip()
        
        # Same location
        if from_loc == to_loc:
            return 0.0, True
        
        # Lookup
        if (from_loc, to_loc) in self.distance_lookup:
            return self.distance_lookup[(from_loc, to_loc)], True
        
        return 0.0, False
    
    def _get_region(self, location: str) -> str:
        """Get region for a location"""
        if not self.region_lookup:
            return 'UNKNOWN'
        
        loc = str(location).strip()
        return self.region_lookup.get(loc, 'UNKNOWN')
    
    def _is_cross_region(self, from_location: str, to_location: str) -> Tuple[bool, bool]:
        """
        Check if transition is cross-region
        
        Returns:
            Tuple of (is_cross_region, has_region_info)
        """
        from_region = self._get_region(from_location)
        to_region = self._get_region(to_location)
        
        if from_region == 'UNKNOWN' or to_region == 'UNKNOWN':
            return False, False
        
        return from_region != to_region, True
    
    def _get_sort_center(self, event: Dict) -> str:
        """Get sort_center from event"""
        sort_center = event.get('sort_center')
        if sort_center and str(sort_center) != 'nan':
            return str(sort_center)
        return 'UNKNOWN'
    
    def _get_delivery_station(self, event: Dict) -> str:
        """Get delivery_station from event"""
        station = event.get('delivery_station')
        if station and str(station) != 'nan':
            return str(station)
        return 'UNKNOWN'
    
    def _get_delivery_postal(self, event: Dict) -> str:
        """Extract postal code from delivery_location (only for DELIVERY events)"""
        event_type = str(event.get('event_type', ''))
        if event_type != 'DELIVERY':
            return 'UNKNOWN'
        
        delivery_loc = event.get('delivery_location')
        if delivery_loc and isinstance(delivery_loc, dict):
            postal_id = delivery_loc.get('id')
            if postal_id:
                return str(postal_id)
        return 'UNKNOWN'
    
    def _get_from_to_locations(self, event: Dict, prev_event: Optional[Dict], 
                                events: List[Dict], event_idx: int) -> Tuple[str, str]:
        """
        Get from_location and to_location for an event.
        
        For DELIVERY: from=delivery_station, to=delivery_station (postal used separately)
        For others: from=previous_sort_center, to=current_sort_center
        
        Returns:
            Tuple of (from_location, to_location)
        """
        event_type = str(event.get('event_type', ''))
        
        if event_type == 'DELIVERY':
            # DELIVERY: from and to are both delivery_station
            # postal_code is handled separately as to_postal
            delivery_station = self._get_delivery_station(event)
            return delivery_station, delivery_station
        
        # For non-DELIVERY events: to_location is current sort_center
        to_loc = self._get_sort_center(event)
        
        # from_location depends on previous event
        if prev_event is not None:
            prev_type = str(prev_event.get('event_type', ''))
            if prev_type == 'DELIVERY':
                # Previous was delivery - from delivery_station
                from_loc = self._get_delivery_station(prev_event)
            else:
                # Previous was sort center event
                from_loc = self._get_sort_center(prev_event)
        else:
            # First event: from_location is same as to_location
            from_loc = to_loc
        
        return from_loc, to_loc
    
    # ==================== Parsing Utility Methods ====================
    
    def _parse_problem_field(self, problem_value) -> List[str]:
        """Parse problem field which can be None, JSON string, or list"""
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
        
        return []
    
    def _parse_datetime(self, time_value: Union[str, datetime, None]) -> Optional[datetime]:
        """Parse time value to datetime object"""
        if time_value is None:
            return None
        
        if isinstance(time_value, datetime):
            return time_value
        
        if isinstance(time_value, str):
            if time_value == 'null' or time_value.strip() == '':
                return None
            try:
                return datetime.fromisoformat(str(time_value).replace('Z', '+00:00'))
            except Exception:
                return None
        
        return None
    
    def _calculate_time_vs_plan(self, event_time: Union[str, datetime], 
                                plan_time: Union[str, datetime, None]) -> float:
        """Calculate time difference between event_time and plan_time in hours"""
        event_dt = self._parse_datetime(event_time)
        plan_dt = self._parse_datetime(plan_time)
        
        if event_dt is None or plan_dt is None:
            return 0.0
        
        try:
            diff_hours = (event_dt - plan_dt).total_seconds() / 3600
            diff_hours = max(-720, min(diff_hours, 720))
            return float(diff_hours)
        except Exception:
            return 0.0
    
    def _calculate_time_until_plan(self, current_time: Union[str, datetime],
                                   plan_time: Union[str, datetime, None]) -> float:
        """Calculate time until planned time in hours"""
        current_dt = self._parse_datetime(current_time)
        plan_dt = self._parse_datetime(plan_time)
        
        if current_dt is None or plan_dt is None:
            return 0.0
        
        try:
            diff_hours = (plan_dt - current_dt).total_seconds() / 3600
            diff_hours = max(-720, min(diff_hours, 720))
            return float(diff_hours)
        except Exception:
            return 0.0
    
    def _get_plan_time_for_event(self, event: Dict, prev_event: Dict = None) -> Optional[str]:
        """
        Get the appropriate plan_time for an event.
        
        Logic:
        - EXIT: plan_time = previous event's CPT (previous is INDUCT or LINEHAUL)
        - Other events: use their own plan_time
        """
        event_type = str(event.get('event_type', ''))
        
        # For EXIT events, use previous event's CPT
        if event_type == 'EXIT' and prev_event is not None:
            prev_type = str(prev_event.get('event_type', ''))
            if prev_type in ['INDUCT', 'LINEHAUL']:
                cpt = prev_event.get('cpt')
                if cpt and cpt != 'null':
                    return cpt
        
        # For other events, use their own plan_time
        plan_time = event.get('plan_time')
        if plan_time and plan_time != 'null':
            return plan_time
        
        return None
    
    def _get_next_plan_time(self, current_event: Dict, next_event: Dict) -> Optional[str]:
        """
        Get the plan_time for the next event (used in lookahead features).
        
        Logic:
        - If next event is EXIT: use current event's CPT
        - Otherwise: use next event's plan_time
        """
        next_type = str(next_event.get('event_type', ''))
        
        # If next event is EXIT, use current event's CPT
        if next_type == 'EXIT':
            current_type = str(current_event.get('event_type', ''))
            if current_type in ['INDUCT', 'LINEHAUL']:
                cpt = current_event.get('cpt')
                if cpt and cpt != 'null':
                    return cpt
        
        # Otherwise use next event's plan_time
        plan_time = next_event.get('plan_time')
        if plan_time and plan_time != 'null':
            return plan_time
        
        return None
    
    def _get_exit_problem_for_event(self, event: Dict, events: List[Dict], 
                                     event_idx: int) -> Tuple[np.ndarray, float]:
        """
        For INDUCT/LINEHAUL events, get the problem from the next EXIT at same sort center.
        Problems are only associated with INDUCT/LINEHAUL and used to predict EXIT time.
        
        Returns:
            Tuple of (problem_encoding, has_problem_flag)
        """
        event_type = str(event.get('event_type', ''))
        
        # Only INDUCT and LINEHAUL can have problems
        if event_type not in ['INDUCT', 'LINEHAUL']:
            return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
        
        current_sc = self._get_sort_center(event)
        if current_sc == 'UNKNOWN':
            return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
        
        # Look for next EXIT at same sort center
        for i in range(event_idx + 1, len(events)):
            next_event = events[i]
            next_type = str(next_event.get('event_type', ''))
            next_sc = self._get_sort_center(next_event)
            
            if next_type == 'EXIT' and next_sc == current_sc:
                problem_encoding = self._encode_problems(next_event.get('problem'))
                problems = self._parse_problem_field(next_event.get('problem'))
                has_problem = 1.0 if problems else 0.0
                return problem_encoding, has_problem
            
            # If we've moved to a different sort center without EXIT, stop looking
            if next_sc != current_sc and next_sc != 'UNKNOWN':
                break
        
        return np.zeros(len(self.problem_types), dtype=np.float32), 0.0
    
    def _encode_problems(self, problem_value) -> np.ndarray:
        """Create multi-hot encoding for problem types"""
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
    
    def _safe_encode(self, encoder: LabelEncoder, value: Optional[str], default: str = 'UNKNOWN') -> int:
        """Safely encode a value, returning UNKNOWN index if not found"""
        if value is None or value == '' or str(value) == 'nan':
            value = default
        else:
            value = str(value)
        
        if value not in encoder.classes_:
            value = default
        
        return int(encoder.transform([value])[0])
    
    def _extract_time_features(self, event_time: datetime) -> Dict[str, float]:
        """Extract cyclical time features from datetime"""
        hour = event_time.hour
        day_of_week = event_time.weekday()
        day_of_month = event_time.day
        month = event_time.month
        
        return {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dow_sin': np.sin(2 * np.pi * day_of_week / 7),
            'dow_cos': np.cos(2 * np.pi * day_of_week / 7),
            'dom_sin': np.sin(2 * np.pi * day_of_month / 31),
            'dom_cos': np.cos(2 * np.pi * day_of_month / 31),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
        }
    
    def _extract_plan_time_features(self, plan_time: Optional[str]) -> Tuple[Dict[str, float], float]:
        """
        Extract cyclical time features from plan_time.
        
        Returns:
            Tuple of (time_features_dict, has_plan_time_flag)
        """
        plan_dt = self._parse_datetime(plan_time)
        
        if plan_dt is None:
            # Return zeros if no plan time
            return {
                'plan_hour_sin': 0.0,
                'plan_hour_cos': 0.0,
                'plan_dow_sin': 0.0,
                'plan_dow_cos': 0.0,
                'plan_dom_sin': 0.0,
                'plan_dom_cos': 0.0,
                'plan_month_sin': 0.0,
                'plan_month_cos': 0.0,
            }, 0.0
        
        hour = plan_dt.hour
        day_of_week = plan_dt.weekday()
        day_of_month = plan_dt.day
        month = plan_dt.month
        
        return {
            'plan_hour_sin': np.sin(2 * np.pi * hour / 24),
            'plan_hour_cos': np.cos(2 * np.pi * hour / 24),
            'plan_dow_sin': np.sin(2 * np.pi * day_of_week / 7),
            'plan_dow_cos': np.cos(2 * np.pi * day_of_week / 7),
            'plan_dom_sin': np.sin(2 * np.pi * day_of_month / 31),
            'plan_dom_cos': np.cos(2 * np.pi * day_of_month / 31),
            'plan_month_sin': np.sin(2 * np.pi * month / 12),
            'plan_month_cos': np.cos(2 * np.pi * month / 12),
        }, 1.0
    
    # ==================== Fitting ====================
    
    def fit(self, df: pd.DataFrame):
        """Fit encoders and scalers on training data"""
        
        all_locations = set()  # sort centers and delivery stations
        all_carriers = set()
        all_leg_types = set()
        all_ship_methods = set()
        all_postals = set()  # postal codes (DELIVERY only)
        all_regions = set()
        
        # Collect regions from distance file
        all_regions.update(self.region_lookup.values())
        
        # === Collect all categorical values ===
        for _, row in df.iterrows():
            # Package level postal codes
            source_postal = row.get('source_postal')
            dest_postal = row.get('dest_postal')
            
            if source_postal and str(source_postal) != 'nan':
                all_postals.add(str(source_postal))
            if dest_postal and str(dest_postal) != 'nan':
                all_postals.add(str(dest_postal))
            
            events = row['events']
            for i, event in enumerate(events):
                # Collect sort center
                sort_center = self._get_sort_center(event)
                if sort_center != 'UNKNOWN':
                    all_locations.add(sort_center)
                    region = self._get_region(sort_center)
                    if region != 'UNKNOWN':
                        all_regions.add(region)
                
                # Collect delivery station
                delivery_station = self._get_delivery_station(event)
                if delivery_station != 'UNKNOWN':
                    all_locations.add(delivery_station)
                    region = self._get_region(delivery_station)
                    if region != 'UNKNOWN':
                        all_regions.add(region)
                
                # Collect delivery postal codes (DELIVERY events only)
                delivery_postal = self._get_delivery_postal(event)
                if delivery_postal != 'UNKNOWN':
                    all_postals.add(delivery_postal)
                
                if 'carrier_id' in event and event['carrier_id']:
                    all_carriers.add(str(event['carrier_id']))
                
                if 'leg_type' in event and event['leg_type']:
                    all_leg_types.add(str(event['leg_type']))
                
                if 'ship_method' in event and event['ship_method']:
                    all_ship_methods.add(str(event['ship_method']))
        
        # Add special tokens (PAD=0, UNKNOWN=1)
        special_tokens = ['PAD', 'UNKNOWN']
        
        all_locations = special_tokens + sorted(list(all_locations - {'UNKNOWN'}))
        all_carriers = special_tokens + sorted(list(all_carriers - {'UNKNOWN'}))
        all_leg_types = special_tokens + sorted(list(all_leg_types - {'UNKNOWN'}))
        all_ship_methods = special_tokens + sorted(list(all_ship_methods - {'UNKNOWN'}))
        all_postals = special_tokens + sorted(list(all_postals - {'UNKNOWN'}))
        all_regions = special_tokens + sorted(list(all_regions - {'UNKNOWN'}))
        
        # Fit encoders
        self.event_type_encoder.fit(['PAD'] + self.event_types)
        self.location_encoder.fit(all_locations)
        self.carrier_encoder.fit(all_carriers)
        self.leg_type_encoder.fit(all_leg_types)
        self.ship_method_encoder.fit(all_ship_methods)
        self.postal_encoder.fit(all_postals)
        self.region_encoder.fit(all_regions)
        
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
        
        # === Collect values for all scalers ===
        time_since_prev_vals = []
        dwelling_time_vals = []
        plan_time_diff_vals = []
        label_time_vals = []
        next_plan_time_diff_vals = []
        
        edge_distance_vals = []
        edge_next_plan_time_vals = []
        
        for _, row in df.iterrows():
            events = row['events']
            event_times = []
            
            # First pass: collect event times and features
            for i, event in enumerate(events):
                event_time = self._parse_datetime(event['event_time'])
                if event_time:
                    event_times.append(event_time)
                    
                    prev_event = events[i-1] if i > 0 else None
                    plan_time = self._get_plan_time_for_event(event, prev_event)
                    delay = self._calculate_time_vs_plan(event['event_time'], plan_time)
                    plan_time_diff_vals.append([delay])
                    
                    # Time since previous
                    if i > 0 and len(event_times) > 1:
                        time_since_prev = (event_time - event_times[-2]).total_seconds() / 3600
                        time_since_prev_vals.append([time_since_prev])
                    
                    # Dwelling time
                    dwelling = event.get('dwelling_seconds', 0) or 0
                    dwelling_time_vals.append([dwelling / 3600])
                    
                    # Label (time to next event)
                    if i < len(events) - 1:
                        next_event_time = self._parse_datetime(events[i+1]['event_time'])
                        if next_event_time:
                            label_time = (next_event_time - event_time).total_seconds() / 3600
                            label_time_vals.append([label_time])
                    
                    # Next plan time diff (lookahead)
                    if i < len(events) - 1:
                        next_event = events[i+1]
                        next_plan_time = self._get_next_plan_time(event, next_event)
                        if next_plan_time:
                            next_plan_diff = self._calculate_time_until_plan(
                                event['event_time'], next_plan_time
                            )
                            next_plan_time_diff_vals.append([next_plan_diff])
            
            # Second pass: edge features
            for i in range(len(events) - 1):
                if i < len(event_times) - 1:
                    event_from = events[i]
                    event_to = events[i + 1]
                    
                    # Get edge locations
                    edge_from_loc, edge_to_loc = self._get_edge_locations(event_from, event_to)
                    
                    distance, has_dist = self._get_distance(edge_from_loc, edge_to_loc)
                    if has_dist and distance > 0:
                        edge_distance_vals.append([distance])
                    
                    # Next plan time (plan time of destination node)
                    next_plan_time = self._get_next_plan_time(event_from, event_to)
                    if next_plan_time:
                        time_until_next_plan = self._calculate_time_until_plan(
                            event_from['event_time'], next_plan_time
                        )
                        edge_next_plan_time_vals.append([time_until_next_plan])
        
        # === Fit all scalers ===
        self._fit_scaler(self.time_since_prev_scaler, time_since_prev_vals, 'time_since_prev')
        self._fit_scaler(self.dwelling_time_scaler, dwelling_time_vals, 'dwelling_time')
        self._fit_scaler(self.plan_time_diff_scaler, plan_time_diff_vals, 'plan_time_diff')
        self._fit_scaler(self.label_time_scaler, label_time_vals, 'label_time')
        self._fit_scaler(self.next_plan_time_diff_scaler, next_plan_time_diff_vals, 'next_plan_time_diff')
        
        self._fit_scaler(self.edge_distance_scaler, edge_distance_vals, 'edge_distance')
        self._fit_scaler(self.edge_next_plan_time_scaler, edge_next_plan_time_vals, 'edge_next_plan_time')
        
        # Package feature scaler
        package_features = df[['weight', 'length', 'width', 'height']].fillna(0).values
        self.package_feature_scaler.fit(package_features)
        
        self._print_scaler_stats()
        
        self.fitted = True
        return self
    
    def _fit_scaler(self, scaler: StandardScaler, values: List, name: str):
        """Fit a scaler with fallback for empty data"""
        if values:
            scaler.fit(np.array(values))
        else:
            print(f"Warning: No data for {name} scaler, using default")
            scaler.fit(np.array([[0.0]]))
    
    def _print_scaler_stats(self):
        """Print statistics for all fitted scalers"""
        print("\n=== Scaler Statistics ===")
        
        scalers = {
            'time_since_prev': self.time_since_prev_scaler,
            'dwelling_time': self.dwelling_time_scaler,
            'plan_time_diff': self.plan_time_diff_scaler,
            'label_time': self.label_time_scaler,
            'next_plan_time_diff': self.next_plan_time_diff_scaler,
            'edge_distance': self.edge_distance_scaler,
            'edge_next_plan_time': self.edge_next_plan_time_scaler,
        }
        
        for name, scaler in scalers.items():
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                print(f"  {name}: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    
    # ==================== Edge Location Logic ====================
    
    def _get_edge_locations(self, event_from: Dict, event_to: Dict) -> Tuple[str, str]:
        """
        Get from/to locations for an edge connecting two events.
        
        Logic:
        - Sort center events (INDUCT, EXIT, LINEHAUL): use sort_center
        - DELIVERY: use delivery_station
        
        Returns:
            Tuple of (edge_from_location, edge_to_location)
        """
        from_type = str(event_from.get('event_type', ''))
        to_type = str(event_to.get('event_type', ''))
        
        # Determine edge_from_location
        if from_type == 'DELIVERY':
            edge_from_location = self._get_delivery_station(event_from)
        else:
            edge_from_location = self._get_sort_center(event_from)
        
        # Determine edge_to_location
        if to_type == 'DELIVERY':
            edge_to_location = self._get_delivery_station(event_to)
        else:
            edge_to_location = self._get_sort_center(event_to)
        
        return edge_from_location, edge_to_location
    
    # ==================== Feature Extraction ====================
    
    def _extract_edge_features(self, event_from: Dict, event_to: Dict,
                               time_from: datetime, time_to: datetime,
                               events: List[Dict], from_idx: int,
                               prev_event: Dict = None) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Extract edge features with proper handling of sort centers, delivery stations, and postal codes.
        
        Returns:
            Tuple of (continuous_features, categorical_indices)
        """
        from_type = str(event_from.get('event_type', ''))
        to_type = str(event_to.get('event_type', ''))
        
        # === Get edge locations ===
        edge_from_loc, edge_to_loc = self._get_edge_locations(event_from, event_to)
        
        # Get postal code only for DELIVERY target
        edge_to_postal = self._get_delivery_postal(event_to) if to_type == 'DELIVERY' else 'UNKNOWN'
        
        # === Flags for event types ===
        from_is_delivery = 1.0 if from_type == 'DELIVERY' else 0.0
        to_is_delivery = 1.0 if to_type == 'DELIVERY' else 0.0
        
        # === Continuous Features ===
        
        # 1. Distance from lookup (scaled)
        distance, has_distance = self._get_distance(edge_from_loc, edge_to_loc)
        distance_scaled = self.edge_distance_scaler.transform([[distance]])[0, 0]
        has_distance_flag = 1.0 if has_distance else 0.0
        
        # 2. Cross-region flag
        is_cross_region, has_region_info = self._is_cross_region(edge_from_loc, edge_to_loc)
        cross_region_flag = 1.0 if is_cross_region else 0.0
        has_region_flag = 1.0 if has_region_info else 0.0
        
        # 3. Next node plan time (scaled)
        # For EXIT target: use source event's CPT
        # For other targets: use target's plan_time
        next_plan_time = self._get_next_plan_time(event_from, event_to)
        time_until_next_plan = 0.0
        has_next_plan = 0.0
        if next_plan_time:
            time_until_next_plan = self._calculate_time_until_plan(
                event_from['event_time'], next_plan_time
            )
            has_next_plan = 1.0
        next_plan_time_scaled = self.edge_next_plan_time_scaler.transform([[time_until_next_plan]])[0, 0]
        
        # 4. Same location flag
        same_location = float(edge_from_loc == edge_to_loc and edge_from_loc != 'UNKNOWN')
        
        # 5. Missort flag (from source event)
        has_missort = 0.0
        if from_type in ['INDUCT', 'LINEHAUL']:
            has_missort = float(event_from.get('missort', False))
        
        # 6. Problem flag (for INDUCT/LINEHAUL predicting EXIT)
        problem_encoding, has_problem = self._get_exit_problem_for_event(event_from, events, from_idx)
        
        # 7. Time features from source node
        time_features_from = self._extract_time_features(time_from)
        
        continuous_features = np.concatenate([
            [distance_scaled, has_distance_flag],           # 2
            [cross_region_flag, has_region_flag],           # 2
            [next_plan_time_scaled, has_next_plan],         # 2
            [same_location],                                 # 1
            [from_is_delivery, to_is_delivery],             # 2 (event type flags)
            [has_missort, has_problem],                     # 2
            problem_encoding,                                # len(problem_types)
            # Time features from source
            [time_features_from['hour_sin']],               # 1
            [time_features_from['hour_cos']],               # 1
            [time_features_from['dow_sin']],                # 1
            [time_features_from['dow_cos']],                # 1
        ], dtype=np.float32)
        
        # === Categorical Indices ===
        carrier_from = event_from.get('carrier_id')
        carrier_to = event_to.get('carrier_id')
        ship_method_from = event_from.get('ship_method')
        ship_method_to = event_to.get('ship_method')
        
        from_region = self._get_region(edge_from_loc)
        to_region = self._get_region(edge_to_loc)
        
        categorical_indices = {
            # Location (sort_center or delivery_station)
            'from_location': self._safe_encode(self.location_encoder, edge_from_loc),
            'to_location': self._safe_encode(self.location_encoder, edge_to_loc),
            # Postal code (only for DELIVERY target)
            'to_postal': self._safe_encode(self.postal_encoder, edge_to_postal),
            # Region
            'from_region': self._safe_encode(self.region_encoder, from_region),
            'to_region': self._safe_encode(self.region_encoder, to_region),
            # Carrier and ship method
            'carrier_from': self._safe_encode(self.carrier_encoder, carrier_from),
            'carrier_to': self._safe_encode(self.carrier_encoder, carrier_to),
            'ship_method_from': self._safe_encode(self.ship_method_encoder, ship_method_from),
            'ship_method_to': self._safe_encode(self.ship_method_encoder, ship_method_to),
        }
        
        return continuous_features, categorical_indices
    
    def _extract_lookahead_features(self, current_event: Dict, next_event: Dict,
                                    current_time: datetime,
                                    events: List[Dict], current_idx: int) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Extract lookahead features (information about next event for prediction).
        
        Next plan time logic:
        - If next event is EXIT: use current event's CPT
        - Otherwise: use next event's plan_time
        
        Returns:
            Tuple of (continuous_features, categorical_indices)
        """
        next_event_type = str(next_event.get('event_type', 'UNKNOWN'))
        
        # Get edge locations
        edge_from_loc, edge_to_loc = self._get_edge_locations(current_event, next_event)
        
        # Get postal code only for DELIVERY
        next_postal = self._get_delivery_postal(next_event) if next_event_type == 'DELIVERY' else 'UNKNOWN'
        
        # === Continuous Features ===
        
        # 1. Time until next planned time (scaled)
        # For EXIT: use current event's CPT
        # For others: use next event's plan_time
        next_plan_time = self._get_next_plan_time(current_event, next_event)
        time_until_plan = 0.0
        has_next_plan = 0.0
        if next_plan_time:
            time_until_plan = self._calculate_time_until_plan(
                current_event['event_time'], next_plan_time
            )
            has_next_plan = 1.0
        time_until_plan_scaled = self.next_plan_time_diff_scaler.transform([[time_until_plan]])[0, 0]
        
        # 2. Distance to next location (scaled)
        distance_to_next, has_distance = self._get_distance(edge_from_loc, edge_to_loc)
        distance_to_next_scaled = self.edge_distance_scaler.transform([[distance_to_next]])[0, 0]
        has_distance_flag = 1.0 if has_distance else 0.0
        
        # 3. Cross-region flag for next transition
        is_cross_region, _ = self._is_cross_region(edge_from_loc, edge_to_loc)
        cross_region_flag = 1.0 if is_cross_region else 0.0
        
        # 4. Next is delivery flag
        next_is_delivery = 1.0 if next_event_type == 'DELIVERY' else 0.0
        
        # 5. Next event problem encoding (for INDUCT/LINEHAUL from EXIT at same SC)
        next_problem_encoding, next_has_problem = self._get_exit_problem_for_event(
            next_event, events, current_idx + 1
        )
        
        # 6. Next event missort flag (if INDUCT or LINEHAUL)
        next_missort = 0.0
        if next_event_type in ['INDUCT', 'LINEHAUL']:
            next_missort = float(next_event.get('missort', False))
        
        continuous_features = np.concatenate([
            [time_until_plan_scaled, has_next_plan],           # 2
            [distance_to_next_scaled, has_distance_flag],      # 2
            [cross_region_flag],                               # 1
            [next_is_delivery],                                # 1
            [next_has_problem, next_missort],                  # 2
            next_problem_encoding,                              # len(problem_types)
        ])
        
        # === Categorical Indices ===
        next_carrier = next_event.get('carrier_id')
        next_leg_type = next_event.get('leg_type')
        next_ship_method = next_event.get('ship_method')
        next_region = self._get_region(edge_to_loc)
        
        categorical_indices = {
            'next_event_type': self._safe_encode(self.event_type_encoder, next_event_type),
            'next_location': self._safe_encode(self.location_encoder, edge_to_loc),
            'next_postal': self._safe_encode(self.postal_encoder, next_postal),
            'next_region': self._safe_encode(self.region_encoder, next_region),
            'next_carrier': self._safe_encode(self.carrier_encoder, next_carrier),
            'next_leg_type': self._safe_encode(self.leg_type_encoder, next_leg_type),
            'next_ship_method': self._safe_encode(self.ship_method_encoder, next_ship_method),
        }
        
        return continuous_features, categorical_indices
    
    # ==================== Main Processing ====================
    
    def process_lifecycle(self, package_data: Dict, return_labels: bool = True) -> Dict:
        """Process a single package lifecycle into graph features"""
        
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before processing")
        
        events = package_data['events']
        num_events = len(package_data['events'])
        
        if num_events < 1:
            raise ValueError("Package must have at least 1 event")
        
        # Calculate dimensions
        lookahead_cont_dim = 2 + 2 + 1 + 1 + 2 + len(self.problem_types)  # 8 + len(problem_types)
        edge_cont_dim = 15 + len(self.problem_types)
        
        # === Initialize Storage ===
        node_continuous_features = []
        node_categorical_indices = {
            'event_type': [],
            'from_location': [],  # sort_center or delivery_station
            'to_location': [],    # sort_center or delivery_station
            'to_postal': [],      # postal code (DELIVERY only)
            'from_region': [],
            'to_region': [],
            'carrier': [],
            'leg_type': [],
            'ship_method': [],
        }
        
        lookahead_categorical_indices = {
            'next_event_type': [],
            'next_location': [],
            'next_postal': [],
            'next_region': [],
            'next_carrier': [],
            'next_leg_type': [],
            'next_ship_method': [],
        }
        
        event_times = []
        
        # Package-level categorical
        source_postal_idx = self._safe_encode(
            self.postal_encoder, package_data.get('source_postal')
        )
        dest_postal_idx = self._safe_encode(
            self.postal_encoder, package_data.get('dest_postal')
        )
        
        # === Process Each Event ===
        for i, event in enumerate(events):
            event_type = str(event.get('event_type', 'UNKNOWN'))
            prev_event = events[i-1] if i > 0 else None
            is_delivery = event_type == 'DELIVERY'
            
            # --- Get locations ---
            from_loc, to_loc = self._get_from_to_locations(event, prev_event, events, i)
            
            # Get postal code only for DELIVERY
            to_postal = self._get_delivery_postal(event) if is_delivery else 'UNKNOWN'
            
            # --- Categorical Features ---
            event_type_idx = self._safe_encode(self.event_type_encoder, event_type)
            node_categorical_indices['event_type'].append(event_type_idx)
            
            from_location_idx = self._safe_encode(self.location_encoder, from_loc)
            node_categorical_indices['from_location'].append(from_location_idx)
            
            to_location_idx = self._safe_encode(self.location_encoder, to_loc)
            node_categorical_indices['to_location'].append(to_location_idx)
            
            to_postal_idx = self._safe_encode(self.postal_encoder, to_postal)
            node_categorical_indices['to_postal'].append(to_postal_idx)
            
            from_region = self._get_region(from_loc)
            from_region_idx = self._safe_encode(self.region_encoder, from_region)
            node_categorical_indices['from_region'].append(from_region_idx)
            
            to_region = self._get_region(to_loc)
            to_region_idx = self._safe_encode(self.region_encoder, to_region)
            node_categorical_indices['to_region'].append(to_region_idx)
            
            carrier = event.get('carrier_id')
            carrier_idx = self._safe_encode(self.carrier_encoder, carrier)
            node_categorical_indices['carrier'].append(carrier_idx)
            
            leg_type = event.get('leg_type')
            leg_type_idx = self._safe_encode(self.leg_type_encoder, leg_type)
            node_categorical_indices['leg_type'].append(leg_type_idx)
            
            ship_method = event.get('ship_method')
            ship_method_idx = self._safe_encode(self.ship_method_encoder, ship_method)
            node_categorical_indices['ship_method'].append(ship_method_idx)
            
            # --- Event Time ---
            event_time = self._parse_datetime(event['event_time'])
            if event_time is None:
                raise ValueError(f"Invalid event_time for event {i}")
            event_times.append(event_time)
            
            # --- Time Features (scaled) ---
            if i == 0:
                time_since_prev_scaled = 0.0
            else:
                time_since_prev = (event_time - event_times[i-1]).total_seconds() / 3600
                time_since_prev_scaled = self.time_since_prev_scaler.transform([[time_since_prev]])[0, 0]
            
            # Position (normalized 0-1)
            position = i / max(1, num_events - 1)
            
            # Dwelling time (scaled)
            dwelling_hours = (event.get('dwelling_seconds', 0) or 0) / 3600
            dwelling_scaled = self.dwelling_time_scaler.transform([[dwelling_hours]])[0, 0]
            has_dwelling = 1.0 if dwelling_hours > 0 else 0.0
            
            # Plan time diff (scaled) - how late/early vs plan
            # For EXIT: plan_time = prev event's CPT
            plan_time = self._get_plan_time_for_event(event, prev_event)
            time_vs_plan = self._calculate_time_vs_plan(event['event_time'], plan_time)
            time_vs_plan_scaled = self.plan_time_diff_scaler.transform([[time_vs_plan]])[0, 0]
            has_plan_time = 1.0 if plan_time is not None else 0.0
            
            # Plan time cyclical features (when the event was planned)
            plan_time_features, has_plan_time_features = self._extract_plan_time_features(plan_time)
            
            # Is delivery flag
            is_delivery_flag = 1.0 if is_delivery else 0.0
            
            # Missort flag (only for INDUCT/LINEHAUL)
            missort_flag = 0.0
            if event_type in ['INDUCT', 'LINEHAUL']:
                missort_flag = float(event.get('missort', False))
            
            # Problem encoding (only for INDUCT/LINEHAUL, from EXIT at same SC)
            problem_encoding, has_problem = self._get_exit_problem_for_event(event, events, i)
            
            # Actual event time cyclical features
            time_features = self._extract_time_features(event_time)
            
            # --- Lookahead Features ---
            if i < num_events - 1:
                next_event = events[i + 1]
                lookahead_cont, lookahead_cat = self._extract_lookahead_features(
                    event, next_event, event_time, events, i
                )
                
                for key, val in lookahead_cat.items():
                    lookahead_categorical_indices[key].append(val)
            else:
                # Last event - use zeros and PAD indices
                lookahead_cont = np.zeros(lookahead_cont_dim, dtype=np.float32)
                for key in lookahead_categorical_indices:
                    lookahead_categorical_indices[key].append(0)  # PAD index
            
            # --- Combine Continuous Features ---
            cont_features = np.concatenate([
                # Time features (scaled)
                [time_since_prev_scaled, position],                            # 2
                [dwelling_scaled, has_dwelling],                               # 2
                [time_vs_plan_scaled, has_plan_time],                         # 2
                # Flags
                [is_delivery_flag],                                            # 1
                [missort_flag, has_problem],                                   # 2
                problem_encoding,                                              # len(problem_types)
                # Actual event time cyclical features
                [time_features['hour_sin'], time_features['hour_cos']],       # 2
                [time_features['dow_sin'], time_features['dow_cos']],         # 2
                [time_features['dom_sin'], time_features['dom_cos']],         # 2
                [time_features['month_sin'], time_features['month_cos']],     # 2
                # Plan time cyclical features (when event was planned)
                [has_plan_time_features],                                      # 1
                [plan_time_features['plan_hour_sin'], plan_time_features['plan_hour_cos']],   # 2
                [plan_time_features['plan_dow_sin'], plan_time_features['plan_dow_cos']],     # 2
                [plan_time_features['plan_dom_sin'], plan_time_features['plan_dom_cos']],     # 2
                [plan_time_features['plan_month_sin'], plan_time_features['plan_month_cos']], # 2
                # Lookahead features
                lookahead_cont,                                                # lookahead_cont_dim
            ])
            
            node_continuous_features.append(cont_features)
        
        # Convert to numpy arrays
        node_continuous_features = np.array(node_continuous_features, dtype=np.float32)
        
        for key in node_categorical_indices:
            node_categorical_indices[key] = np.array(node_categorical_indices[key], dtype=np.int64)
        for key in lookahead_categorical_indices:
            lookahead_categorical_indices[key] = np.array(lookahead_categorical_indices[key], dtype=np.int64)
        
        # --- Package Features (scaled) ---
        package_features = np.array([
            package_data.get('weight', 0) or 0,
            package_data.get('length', 0) or 0,
            package_data.get('width', 0) or 0,
            package_data.get('height', 0) or 0
        ], dtype=np.float32).reshape(1, -1)
        
        package_features_scaled = self.package_feature_scaler.transform(package_features).flatten()
        package_features_expanded = np.tile(package_features_scaled, (num_events, 1))
        
        node_continuous_features = np.concatenate(
            [node_continuous_features, package_features_expanded], axis=1
        )
        
        # === Edge Features ===
        edge_index = []
        edge_continuous_features = []
        edge_categorical_indices = {
            'from_location': [],
            'to_location': [],
            'to_postal': [],
            'from_region': [],
            'to_region': [],
            'carrier_from': [],
            'carrier_to': [],
            'ship_method_from': [],
            'ship_method_to': [],
        }
        
        if num_events > 1:
            for i in range(num_events - 1):
                edge_index.append([i, i+1])
                
                prev_event = events[i-1] if i > 0 else None
                edge_cont, edge_cat = self._extract_edge_features(
                    events[i], events[i+1],
                    event_times[i], event_times[i+1],
                    events, i,
                    prev_event
                )
                
                edge_continuous_features.append(edge_cont)
                for key, val in edge_cat.items():
                    edge_categorical_indices[key].append(val)
            
            edge_index = np.array(edge_index, dtype=np.int64).T
            edge_continuous_features = np.array(edge_continuous_features, dtype=np.float32)
            
            for key in edge_categorical_indices:
                edge_categorical_indices[key] = np.array(edge_categorical_indices[key], dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_continuous_features = np.zeros((0, edge_cont_dim), dtype=np.float32)
            for key in edge_categorical_indices:
                edge_categorical_indices[key] = np.zeros((0,), dtype=np.int64)
        
        # === Build Result ===
        result = {
            'node_continuous_features': node_continuous_features,
            'node_categorical_indices': node_categorical_indices,
            'lookahead_categorical_indices': lookahead_categorical_indices,
            'package_categorical': {
                'source_postal': source_postal_idx,
                'dest_postal': dest_postal_idx,
            },
            'edge_index': edge_index,
            'edge_continuous_features': edge_continuous_features,
            'edge_categorical_indices': edge_categorical_indices,
            'num_nodes': num_events,
            'package_id': package_data['package_id'],
        }
        
        # === Labels (scaled) ===
        if return_labels:
            labels = []
            for i in range(num_events - 1):
                # Transit time = next_event_time - current_event_time
                transit_hours = (event_times[i+1] - event_times[i]).total_seconds() / 3600
                labels.append(transit_hours)
            
            if labels:
                labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
                if self.config.data.normalize_time:
                    labels = self.label_time_scaler.transform(labels)
            else:
                labels = np.zeros((0, 1), dtype=np.float32)
            
            result['labels'] = labels
            
            label_mask = np.zeros(num_events, dtype=bool)
            label_mask[:-1] = True
            result['label_mask'] = label_mask
        
        return result
    
    # ==================== Utility Methods ====================
    
    def inverse_transform_time(self, scaled_time):
        """Convert scaled time back to hours
        
        Args:
            scaled_time: Scaled time values (can be 1D or 2D array)
            
        Returns:
            Time in hours (same shape as input)
        """
        if scaled_time is None:
            return None
        
        # Convert to numpy if needed
        if hasattr(scaled_time, 'numpy'):
            scaled_time = scaled_time.numpy()
        elif not isinstance(scaled_time, np.ndarray):
            scaled_time = np.array(scaled_time)
        
        # Handle scalar
        if scaled_time.ndim == 0:
            scaled_time = scaled_time.reshape(1, 1)
            result = self.label_time_scaler.inverse_transform(scaled_time)
            return result.item()
        
        # Handle 1D array
        if scaled_time.ndim == 1:
            scaled_time_2d = scaled_time.reshape(-1, 1)
            result = self.label_time_scaler.inverse_transform(scaled_time_2d)
            return result.flatten()
        
        # Already 2D
        return self.label_time_scaler.inverse_transform(scaled_time)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of all feature components"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        # Lookahead continuous: plan(2) + distance(2) + cross_region(1) + is_delivery(1) + flags(2) + problems
        lookahead_cont_dim = 2 + 2 + 1 + 1 + 2 + len(self.problem_types)
        
        # Node continuous: 
        # time(2) + dwelling(2) + plan_diff(2) + is_delivery(1) + flags(2) + problems 
        # + actual_time(8) + plan_time(9) + lookahead + package(4)
        node_continuous_dim = (
            2 +  # time features (time_since_prev, position)
            2 +  # dwelling
            2 +  # plan time diff
            1 +  # is_delivery flag
            2 +  # flags (missort, has_problem)
            len(self.problem_types) +  # problems
            8 +  # actual event time cyclical
            9 +  # plan time cyclical (has_plan + 8)
            lookahead_cont_dim +  # lookahead
            4    # package features
        )
        
        # Edge continuous: distance(2) + region(2) + plan(2) + same(1) + delivery_flags(2) + flags(2) + problem + time(4)
        edge_continuous_dim = 15 + len(self.problem_types)
        
        return {
            'vocab_sizes': self.vocab_sizes.copy(),
            'node_continuous_dim': node_continuous_dim,
            'edge_continuous_dim': edge_continuous_dim,
            'problem_types_dim': len(self.problem_types),
            'num_node_categorical': 9,  # event_type, from_location, to_location, to_postal, from_region, to_region, carrier, leg_type, ship_method
            'num_lookahead_categorical': 7,  # next_event_type, next_location, next_postal, next_region, next_carrier, next_leg_type, next_ship_method
            'num_edge_categorical': 9,  # from_location, to_location, to_postal, from_region, to_region, carrier_from, carrier_to, ship_method_from, ship_method_to
            'lookahead_continuous_dim': lookahead_cont_dim,
        }
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for embedding layers"""
        return self.vocab_sizes.copy()
    
    def get_distance_coverage(self) -> Dict[str, float]:
        """Get statistics about distance data coverage"""
        if not self.distance_lookup:
            return {'coverage': 0.0, 'num_pairs': 0}
        
        distances = list(set(self.distance_lookup.values()))
        return {
            'num_pairs': len(self.distance_lookup) // 2,
            'min_distance': min(distances),
            'max_distance': max(distances),
            'mean_distance': np.mean(distances),
            'unit': self.distance_unit,
        }
    
    def save(self, path: str):
        """Save preprocessor to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'PackageLifecyclePreprocessor':
        """Load preprocessor from file"""
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {path}")
        return preprocessor
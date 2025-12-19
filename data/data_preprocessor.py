import numpy as np
import pandas as pd
from datetime import datetime
<<<<<<< HEAD
from typing import Dict, List, Set, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import ast
=======
from typing import Dict, List, Union, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import ast
import os
import pickle

>>>>>>> 43a4a96 (large set 1)

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
    
    def __init__(self, config, distance_file_path: str = None):
        """
        Args:
            config: Configuration object with data.event_types and data.problem_types
            distance_file_path: Path to location_distances_complete.csv
        """
        self.config = config
        
        # Distance and region lookup
        self.distance_lookup = {}
        self.region_lookup = {}
        self.distance_unit = 'miles'
        
        self.distance_file_path = distance_file_path or os.path.join(
            'data', 'location_distances_complete.csv'
        )
        self._load_distance_data()
        
        # === Categorical Encoders ===
        self.event_type_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()  # For sort centers and delivery stations
        self.carrier_encoder = LabelEncoder()
        self.leg_type_encoder = LabelEncoder()
        self.ship_method_encoder = LabelEncoder()
<<<<<<< HEAD
        self.location_encoder = LabelEncoder()
        
        self.time_scaler = StandardScaler()  # Now scales TRANSIT TIMES (event gaps)
=======
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
>>>>>>> 43a4a96 (large set 1)
        self.package_feature_scaler = StandardScaler()
        self.plan_time_diff_scaler = StandardScaler()
        self.planned_remaining_scaler = StandardScaler()  # For node feature
        self.planned_duration_scaler = StandardScaler()   # For node feature
        self.planned_transit_scaler = StandardScaler()    # NEW: For planned transit time
        
        self.event_types = config.data.event_types
        self.problem_types = config.data.problem_types
<<<<<<< HEAD
        self.max_route_length = config.data.max_route_length
        self.fitted = False
        
        # Track filtered packages
        self.filter_stats = {
            'invalid_event_time': 0,
            'exit_before_induct_linehaul': 0,
            'linehaul_exit_too_close': 0,
            'invalid_sort_center_structure': 0,
            'exit_problem_no_prev': 0,
            'exit_problem_invalid_prev': 0,
            'event_plan_time_diff_too_large': 0,
            'total_processed': 0,
            'total_valid': 0
        }
    
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
=======
        self.problem_type_to_idx = {}
        self.fitted = False
        
        self.vocab_sizes = {}
    
    # ==================== Distance Data Loading ====================
    
    def _load_distance_data(self):
        """Load distance and region lookup tables from CSV"""
        try:
            if not os.path.exists(self.distance_file_path):
                print(f"Warning: Distance file not found at {self.distance_file_path}")
                print("Distance features will be set to 0")
                return
            
            df_dist = pd.read_csv(self.distance_file_path)
            
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
            
            print(f"Loading distances using '{dist_col}' column")
            
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
            print(f"Error loading distance file: {e}")
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
>>>>>>> 43a4a96 (large set 1)
                parsed = json.loads(problem_value)
                if isinstance(parsed, list):
                    return [str(p) for p in parsed]
                return [str(parsed)]
            except json.JSONDecodeError:
                try:
<<<<<<< HEAD
                    # Try ast.literal_eval for Python literals
=======
>>>>>>> 43a4a96 (large set 1)
                    parsed = ast.literal_eval(problem_value)
                    if isinstance(parsed, list):
                        return [str(p) for p in parsed]
                    return [str(parsed)]
                except:
<<<<<<< HEAD
                    # If all parsing fails, return as single item
=======
>>>>>>> 43a4a96 (large set 1)
                    if problem_value.strip():
                        return [problem_value.strip()]
                    return []
        
        return []
    
<<<<<<< HEAD
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
=======
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
>>>>>>> 43a4a96 (large set 1)
                return None
        
        return None
    
<<<<<<< HEAD
    def _extract_route(self, events: List[Dict]) -> List[str]:
        """
        Extract the planned route from events (sequence of unique locations)
        Deduplicates consecutive same locations
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of unique location identifiers in order (e.g., ['SC_123', 'SC_456', 'DS_789'])
        """
        route = []
        
        for event in events:
            event_type = event['event_type']
            sort_center = event.get('sort_center')
            
            if not sort_center or sort_center == 'null':
                continue
            
            # Create location identifier based on event type
            if event_type in ['INDUCT', 'EXIT', 'LINEHAUL']:
                location = f"SC_{sort_center}"
            elif event_type == 'DELIVERY':
                location = f"DS_{sort_center}"
            else:
                continue
            
            # Add deduplication logic - only add if different from last location
            if not route or route[-1] != location:
                route.append(location)
        
        return route
    
    def _encode_route(self, route: List[str]) -> np.ndarray:
        """
        Encode route as indices with padding/truncation
        
        Args:
            route: List of location strings
            
        Returns:
            Array of indices with shape (max_route_length,)
        """
        # Encode locations to indices
        encoded = []
        for loc in route[:self.max_route_length]:  # Truncate if too long
            if loc in self.location_encoder.classes_:
                encoded.append(self.location_encoder.transform([loc])[0])
            else:
                # Unknown location
                encoded.append(self.location_encoder.transform(['UNKNOWN'])[0])
        
        # Pad with PADDING token if too short
        padding_idx = self.location_encoder.transform(['PADDING'])[0]
        while len(encoded) < self.max_route_length:
            encoded.append(padding_idx)
        
        return np.array(encoded, dtype=np.int32)
    
    def _extract_temporal_features(self, event_time: datetime) -> np.ndarray:
        """
        Extract temporal features from event_time
        
        Args:
            event_time: datetime object
            
        Returns:
            Array of [month_sin, month_cos, dow_sin, dow_cos, hour_sin, hour_cos]
        """
        # Month of year (1-12) - cyclical encoding
        month = event_time.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Day of week (0-6, Monday=0) - cyclical encoding
        dow = event_time.weekday()
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        
        # Hour of day (0-23) - cyclical encoding
        hour = event_time.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        return np.array([month_sin, month_cos, dow_sin, dow_cos, hour_sin, hour_cos], dtype=np.float32)
    
    def _extract_next_plan_temporal_features(self, next_plan_time: Union[str, datetime, None]) -> np.ndarray:
        """
        Extract temporal features from next_plan_time
        
        Args:
            next_plan_time: datetime object, string, or None
            
        Returns:
            Array of [month_sin, month_cos, dow_sin, dow_cos, hour_sin, hour_cos, has_next_plan]
        """
        # Parse next_plan_time
        next_plan_dt = self._parse_datetime(next_plan_time)
        
        if next_plan_dt is None:
            # No next_plan_time - return zeros with flag = 0
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Extract temporal features from next_plan_time
        temporal_features = self._extract_temporal_features(next_plan_dt)
        
        # Has next_plan flag
        has_next_plan = 1.0
        
        return np.concatenate([temporal_features, [has_next_plan]]).astype(np.float32)
    
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
        # Parse both times to datetime objects
=======
    def _calculate_time_vs_plan(self, event_time: Union[str, datetime], 
                                plan_time: Union[str, datetime, None]) -> float:
        """Calculate time difference between event_time and plan_time in hours"""
>>>>>>> 43a4a96 (large set 1)
        event_dt = self._parse_datetime(event_time)
        plan_dt = self._parse_datetime(plan_time)
        
        if event_dt is None or plan_dt is None:
            return 0.0
        
        try:
            diff_hours = (event_dt - plan_dt).total_seconds() / 3600
<<<<<<< HEAD
            
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
=======
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
>>>>>>> 43a4a96 (large set 1)
        """
        event_type = str(event.get('event_type', ''))
        
        # For EXIT events, use previous event's CPT
        if event_type == 'EXIT' and prev_event is not None:
<<<<<<< HEAD
            cpt = prev_event.get('cpt')
            if cpt and cpt != 'null':
                return cpt
=======
            prev_type = str(prev_event.get('event_type', ''))
            if prev_type in ['INDUCT', 'LINEHAUL']:
                cpt = prev_event.get('cpt')
                if cpt and cpt != 'null':
                    return cpt
>>>>>>> 43a4a96 (large set 1)
        
        # For other events, use their own plan_time
        plan_time = event.get('plan_time')
        if plan_time and plan_time != 'null':
            return plan_time
        
        return None
<<<<<<< HEAD
=======
    
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
>>>>>>> 43a4a96 (large set 1)
    
    def fit(self, df: pd.DataFrame):
        """Fit encoders and scalers on training data"""
        
        all_locations = set()  # sort centers and delivery stations
        all_carriers = set()
        all_leg_types = set()
        all_ship_methods = set()
<<<<<<< HEAD
        all_locations = set()
        
        # Reset filter stats
        self.filter_stats = {
            'invalid_event_time': 0,
            'exit_before_induct_linehaul': 0,
            'linehaul_exit_too_close': 0,
            'invalid_sort_center_structure': 0,
            'exit_problem_no_prev': 0,
            'exit_problem_invalid_prev': 0,
            'event_plan_time_diff_too_large': 0,
            'total_processed': 0,
            'total_valid': 0
        }
    
        for _, row in df.iterrows():
            self.filter_stats['total_processed'] += 1
            events = row['events']
            
            # Extract route locations (now deduplicated)
            route = self._extract_route(events)
            all_locations.update(route)
            
            for event in events:
                if 'sort_center' in event and event['sort_center']:
                    all_sort_centers.add(str(event['sort_center']))
                    
=======
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
                
>>>>>>> 43a4a96 (large set 1)
                if 'carrier_id' in event and event['carrier_id']:
                    all_carriers.add(str(event['carrier_id']))
                
                if 'leg_type' in event and event['leg_type']:
                    all_leg_types.add(str(event['leg_type']))
                
                if 'ship_method' in event and event['ship_method']:
                    all_ship_methods.add(str(event['ship_method']))
        
<<<<<<< HEAD
        # Add special tokens
        all_sort_centers.add('UNKNOWN')
        all_carriers.add('UNKNOWN')
        all_leg_types.add('UNKNOWN')
        all_ship_methods.add('UNKNOWN')
        all_locations.update(['UNKNOWN', 'PADDING'])
        
        # Fit encoders
        self.event_type_encoder.fit(self.event_types)
        self.sort_center_encoder.fit(sorted(list(all_sort_centers)))
        self.carrier_encoder.fit(sorted(list(all_carriers)))
        self.leg_type_encoder.fit(sorted(list(all_leg_types)))
        self.ship_method_encoder.fit(sorted(list(all_ship_methods)))
        self.location_encoder.fit(sorted(list(all_locations)))
        
        # Store problem types for multi-hot encoding
        self.problem_type_to_idx = {pt: idx for idx, pt in enumerate(self.problem_types)}
        
        print(f"Found {len(all_sort_centers)} sort centers")
        print(f"Found {len(all_carriers)} carriers")
        print(f"Found {len(all_leg_types)} leg types")
        print(f"Found {len(all_ship_methods)} ship methods")
        print(f"Found {len(all_locations)} route locations")
        print(f"Found {len(self.problem_types)} problem types")
        
        # Print filter statistics
        print(f"\n{'='*60}")
        print(f"PACKAGE FILTERING STATISTICS")
        print(f"{'='*60}")
        print(f"Total packages processed: {self.filter_stats['total_processed']}")
        print(f"Valid packages: {self.filter_stats['total_valid']}")
        print(f"Filtered packages: {self.filter_stats['total_processed'] - self.filter_stats['total_valid']}")
        print(f"\nFiltering reasons:")
        print(f"  - Invalid event time: {self.filter_stats['invalid_event_time']}")
        print(f"  - EXIT before INDUCT/LINEHAUL: {self.filter_stats['exit_before_induct_linehaul']}")
        print(f"  - LINEHAUL-EXIT < 5 min: {self.filter_stats['linehaul_exit_too_close']}")
        print(f"  - Invalid sort center structure: {self.filter_stats['invalid_sort_center_structure']}")
        print(f"  - EXIT problem no prev event: {self.filter_stats['exit_problem_no_prev']}")
        print(f"  - EXIT problem invalid prev: {self.filter_stats['exit_problem_invalid_prev']}")
        print(f"  - Event-Plan time diff > 24h: {self.filter_stats['event_plan_time_diff_too_large']}")
        print(f"{'='*60}\n")
    
        # ====================================================================
        # FIT TIME SCALER (TARGET) - TRANSIT TIMES BETWEEN EVENTS
        # ====================================================================
        transit_times = []
        for _, row in df.iterrows():
            events = row['events']
            for i in range(len(events) - 1):
                try:
                    # Current event time
                    current_time = self._parse_datetime(events[i]['event_time'])
                    # Next event time
                    next_time = self._parse_datetime(events[i+1]['event_time'])
                    
                    if current_time and next_time:
                        # Transit time = next_time - current_time
                        transit_hours = (next_time - current_time).total_seconds() / 3600
                        if transit_hours >= 0:  # Only positive transit times
                            transit_times.append([transit_hours])
                except Exception as e:
                    continue
        
        if transit_times:
            self.time_scaler.fit(np.array(transit_times))
            print(f"Fitted time_scaler (TARGET - TRANSIT TIME) on {len(transit_times)} samples")
            print(f"  Mean: {np.mean(transit_times):.2f} hours")
            print(f"  Std: {np.std(transit_times):.2f} hours")
            print(f"  Min: {np.min(transit_times):.2f} hours")
            print(f"  Max: {np.max(transit_times):.2f} hours")
            print(f"  Median: {np.median(transit_times):.2f} hours")
            print(f"  25th percentile: {np.percentile(transit_times, 25):.2f} hours")
            print(f"  75th percentile: {np.percentile(transit_times, 75):.2f} hours")
            print(f"  95th percentile: {np.percentile(transit_times, 95):.2f} hours")
        # ====================================================================
        
        # Fit plan time difference scaler
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
        
        # Fit planned_remaining scaler (next_plan_time - current_event_time)
        # This will be a NODE feature now
        planned_remaining_times = []
        for _, row in df.iterrows():
            events = row['events']
            for i in range(len(events) - 1):
                # Get current event time
                current_event_time = self._parse_datetime(events[i].get('event_time'))
                
                # Get plan_time for next node (node i+1)
                next_plan_time = self._parse_datetime(
                    self._get_plan_time_for_event(events[i+1], events[i])
                )
                
                # Calculate: next_plan_time - current_event_time
                if current_event_time is not None and next_plan_time is not None:
                    diff = (next_plan_time - current_event_time).total_seconds() / 3600
                    planned_remaining_times.append([diff])
                else:
                    planned_remaining_times.append([0.0])
        
        if planned_remaining_times:
            self.planned_remaining_scaler.fit(np.array(planned_remaining_times))
            print(f"\nFitted planned_remaining scaler on {len(planned_remaining_times)} samples")
            print(f"  Mean: {np.mean(planned_remaining_times):.2f} hours")
            print(f"  Std: {np.std(planned_remaining_times):.2f} hours")
            print(f"  Min: {np.min(planned_remaining_times):.2f} hours")
            print(f"  Max: {np.max(planned_remaining_times):.2f} hours")
        
        # Fit planned duration scaler (next_plan_time - current_plan_time)
        # This will be a NODE feature now
        planned_durations = []
        for _, row in df.iterrows():
            events = row['events']
            for i in range(len(events) - 1):
                # Get plan_time for current node
                current_plan_time = self._parse_datetime(
                    self._get_plan_time_for_event(events[i], events[i-1] if i > 0 else None)
                )
                
                # Get plan_time for next node
                next_plan_time = self._parse_datetime(
                    self._get_plan_time_for_event(events[i+1], events[i])
                )
                
                # Calculate planned duration between events
                if current_plan_time is not None and next_plan_time is not None:
                    diff = (next_plan_time - current_plan_time).total_seconds() / 3600
                    planned_durations.append([diff])
                else:
                    planned_durations.append([0.0])
        
        if planned_durations:
            self.planned_duration_scaler.fit(np.array(planned_durations))
            print(f"\nFitted planned_duration scaler on {len(planned_durations)} samples")
            print(f"  Mean: {np.mean(planned_durations):.2f} hours")
            print(f"  Std: {np.std(planned_durations):.2f} hours")
            print(f"  Min: {np.min(planned_durations):.2f} hours")
            print(f"  Max: {np.max(planned_durations):.2f} hours")
        
        # ====================================================================
        # NEW: Fit planned_transit scaler (planned transit time between events)
        # ====================================================================
        planned_transit_times = []
        for _, row in df.iterrows():
            events = row['events']
            for i in range(len(events) - 1):
                # Get plan_time for current node
                current_plan_time = self._parse_datetime(
                    self._get_plan_time_for_event(events[i], events[i-1] if i > 0 else None)
                )
                
                # Get plan_time for next node
                next_plan_time = self._parse_datetime(
                    self._get_plan_time_for_event(events[i+1], events[i])
                )
                
                # Calculate planned transit time
                if current_plan_time is not None and next_plan_time is not None:
                    planned_transit = (next_plan_time - current_plan_time).total_seconds() / 3600
                    planned_transit_times.append([planned_transit])
                else:
                    planned_transit_times.append([0.0])
        
        if planned_transit_times:
            self.planned_transit_scaler.fit(np.array(planned_transit_times))
            print(f"\nFitted planned_transit scaler on {len(planned_transit_times)} samples")
            print(f"  Mean: {np.mean(planned_transit_times):.2f} hours")
            print(f"  Std: {np.std(planned_transit_times):.2f} hours")
            print(f"  Min: {np.min(planned_transit_times):.2f} hours")
            print(f"  Max: {np.max(planned_transit_times):.2f} hours")
            print(f"  Median: {np.median(planned_transit_times):.2f} hours")
            print(f"  25th percentile: {np.percentile(planned_transit_times, 25):.2f} hours")
            print(f"  75th percentile: {np.percentile(planned_transit_times, 75):.2f} hours")
            print(f"  95th percentile: {np.percentile(planned_transit_times, 95):.2f} hours")
        # ====================================================================
=======
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
>>>>>>> 43a4a96 (large set 1)
        
        self._fit_scaler(self.edge_distance_scaler, edge_distance_vals, 'edge_distance')
        self._fit_scaler(self.edge_next_plan_time_scaler, edge_next_plan_time_vals, 'edge_next_plan_time')
        
        # Package feature scaler
        package_features = df[['weight', 'length', 'width', 'height']].fillna(0).values
        self.package_feature_scaler.fit(package_features)
        
        self._print_scaler_stats()
        
        self.fitted = True
        return self
    
<<<<<<< HEAD
    def get_filter_stats(self) -> Dict:
        """Get filtering statistics"""
        return self.filter_stats.copy()
    
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
=======
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
>>>>>>> 43a4a96 (large set 1)
    
    def process_lifecycle(self, package_data: Dict, return_labels: bool = True) -> Dict:
        """Process a single package lifecycle into graph features"""
        
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before processing")
        
        events = package_data['events']
        num_events = len(package_data['events'])
        
<<<<<<< HEAD
        # Validate minimum events
        if num_events < 1:
            raise ValueError("Package must have at least 1 event")
        
        # Extract and encode route (now deduplicated)
        route = self._extract_route(events)
        route_encoded = self._encode_route(route)
        
        node_features = []
=======
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
        
>>>>>>> 43a4a96 (large set 1)
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
            
<<<<<<< HEAD
            # Current sort center
            sort_center = event.get('sort_center', 'UNKNOWN')
            sort_center = str(sort_center) if sort_center else 'UNKNOWN'
            if sort_center not in self.sort_center_encoder.classes_:
                sort_center = 'UNKNOWN'
            sort_center_idx = self.sort_center_encoder.transform([sort_center])[0]
            
            # Next node features (to_sort_center, carrier, leg_type, ship_method)
            # For last node, use UNKNOWN
            if i < num_events - 1:
                next_event = events[i + 1]
                
                # to_sort_center: next node's sort center
                to_sort_center = next_event.get('sort_center', 'UNKNOWN')
                to_sort_center = str(to_sort_center) if to_sort_center else 'UNKNOWN'
                if to_sort_center not in self.sort_center_encoder.classes_:
                    to_sort_center = 'UNKNOWN'
                to_sort_center_idx = self.sort_center_encoder.transform([to_sort_center])[0]
                
                # Carrier from next event
                carrier = next_event.get('carrier_id', 'UNKNOWN')
                carrier = str(carrier) if carrier else 'UNKNOWN'
                if carrier not in self.carrier_encoder.classes_:
                    carrier = 'UNKNOWN'
                carrier_idx = self.carrier_encoder.transform([carrier])[0]
                
                # Leg type from next event
                leg_type = next_event.get('leg_type', 'UNKNOWN')
                leg_type = str(leg_type) if leg_type else 'UNKNOWN'
                if leg_type not in self.leg_type_encoder.classes_:
                    leg_type = 'UNKNOWN'
                leg_type_idx = self.leg_type_encoder.transform([leg_type])[0]
                
                # Ship method from next event
                ship_method = next_event.get('ship_method', 'UNKNOWN')
                ship_method = str(ship_method) if ship_method else 'UNKNOWN'
                if ship_method not in self.ship_method_encoder.classes_:
                    ship_method = 'UNKNOWN'
                ship_method_idx = self.ship_method_encoder.transform([ship_method])[0]
                
                # Is different sort center flag
                is_different_sc = float(sort_center != to_sort_center)
                
                # Hops remaining to end
                hops_remaining = num_events - i - 1
                hops_remaining_normalized = hops_remaining / max(1, num_events - 1)
                
                # Next event type (one-hot)
                next_event_type = str(next_event['event_type'])
                next_event_type_idx = self.event_type_encoder.transform([next_event_type])[0]
                next_event_type_onehot = np.zeros(len(self.event_types))
                next_event_type_onehot[next_event_type_idx] = 1
                
            else:
                # Last node - use UNKNOWN for all next node features
                to_sort_center_idx = self.sort_center_encoder.transform(['UNKNOWN'])[0]
                carrier_idx = self.carrier_encoder.transform(['UNKNOWN'])[0]
                leg_type_idx = self.leg_type_encoder.transform(['UNKNOWN'])[0]
                ship_method_idx = self.ship_method_encoder.transform(['UNKNOWN'])[0]
                is_different_sc = 0.0
                hops_remaining_normalized = 0.0
                next_event_type_onehot = np.zeros(len(self.event_types))
            
            # Event time
=======
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
>>>>>>> 43a4a96 (large set 1)
            event_time = self._parse_datetime(event['event_time'])
            if event_time is None:
                raise ValueError(f"Invalid event_time for event {i}")
            event_times.append(event_time)
            
<<<<<<< HEAD
            # Extract temporal features (month, day of week, hour)
            temporal_features = self._extract_temporal_features(event_time)
            
            # Get appropriate plan_time (use CPT from previous event for EXIT)
            prev_event = events[i-1] if i > 0 else None
            plan_time = self._get_plan_time_for_event(event, prev_event)
            
            # Calculate time vs plan (delay)
            time_vs_plan = self._calculate_time_vs_plan(event['event_time'], plan_time)
            
            # Normalize time_vs_plan
            time_vs_plan_scaled = self.plan_time_diff_scaler.transform([[time_vs_plan]])[0, 0]
            
            # Has plan time flag
            has_plan_time = 1.0 if plan_time is not None else 0.0
            
            # Get next_plan_time directly from event
            next_plan_time = event.get('next_plan_time')
            
            # Extract temporal features from next_plan_time (for prediction)
            next_plan_temporal_features = self._extract_next_plan_temporal_features(next_plan_time)
            
=======
            # --- Time Features (scaled) ---
>>>>>>> 43a4a96 (large set 1)
            if i == 0:
                time_since_prev_scaled = 0.0
            else:
                time_since_prev = (event_time - event_times[i-1]).total_seconds() / 3600
                time_since_prev_scaled = self.time_since_prev_scaler.transform([[time_since_prev]])[0, 0]
            
<<<<<<< HEAD
            # Positional encoding
=======
            # Position (normalized 0-1)
>>>>>>> 43a4a96 (large set 1)
            position = i / max(1, num_events - 1)
            
            # Dwelling time (scaled)
            dwelling_hours = (event.get('dwelling_seconds', 0) or 0) / 3600
            dwelling_scaled = self.dwelling_time_scaler.transform([[dwelling_hours]])[0, 0]
            has_dwelling = 1.0 if dwelling_hours > 0 else 0.0
            
<<<<<<< HEAD
            # Missort flag (for INDUCT and LINEHAUL events)
=======
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
>>>>>>> 43a4a96 (large set 1)
            missort_flag = 0.0
            if event_type in ['INDUCT', 'LINEHAUL']:
                missort_flag = float(event.get('missort', False))
            
<<<<<<< HEAD
            # Problem encoding (for INDUCT and LINEHAUL events)
            problem_encoding = np.zeros(len(self.problem_types), dtype=np.float32)
            if event_type in ['INDUCT', 'LINEHAUL']:
                problem_encoding = self._encode_problems(event.get('problem'))
            
            # Is first/last event flags
            is_first_event = 1.0 if i == 0 else 0.0
            is_last_event = 1.0 if i == num_events - 1 else 0.0
            
            # ============================================================
            # Add "edge features" as NODE features (for non-last nodes)
            # ============================================================
            if i < num_events - 1:
                # Get current event time
                current_event_time = self._parse_datetime(events[i].get('event_time'))
                
                # Get plan_time for next node (node i+1)
                next_plan_time_dt = self._parse_datetime(
                    self._get_plan_time_for_event(events[i+1], events[i])
                )
                
                # Get plan_time for current node
                current_plan_time = self._parse_datetime(
                    self._get_plan_time_for_event(events[i], events[i-1] if i > 0 else None)
                )
                
                # Calculate: next_plan_time - current_event_time
                if current_event_time is not None and next_plan_time_dt is not None:
                    planned_remaining = (next_plan_time_dt - current_event_time).total_seconds() / 3600
                else:
                    planned_remaining = 0.0
                
                # Calculate: next_plan_time - current_plan_time
                if current_plan_time is not None and next_plan_time_dt is not None:
                    planned_duration = (next_plan_time_dt - current_plan_time).total_seconds() / 3600
                else:
                    planned_duration = 0.0
                
                # NEW: Calculate planned transit time (same as planned_duration)
                # This is conceptually an edge feature but stored on node
                planned_transit_time = planned_duration
                
                # Scale these features
                planned_remaining_scaled = self.planned_remaining_scaler.transform([[planned_remaining]])[0, 0]
                planned_duration_scaled = self.planned_duration_scaler.transform([[planned_duration]])[0, 0]
                planned_transit_scaled = self.planned_transit_scaler.transform([[planned_transit_time]])[0, 0]
                
                # Has next plan time flag
                has_next_plan = 1.0 if next_plan_time_dt is not None else 0.0
            else:
                # Last node - no next event
                planned_remaining_scaled = 0.0
                planned_duration_scaled = 0.0
                planned_transit_scaled = 0.0
                has_next_plan = 0.0
            # ============================================================
            
            # Combine features
            features = np.concatenate([
                event_type_onehot,  # Current event type one-hot
                [sort_center_idx, to_sort_center_idx, carrier_idx, leg_type_idx, ship_method_idx],  # Categorical indices
                [time_since_start, time_since_prev, position, dwelling_time],  # Time features
                temporal_features,  # 6 temporal features (current time)
                [time_vs_plan_scaled, has_plan_time],  # Plan time features
                next_plan_temporal_features,  # 7 features (6 temporal + 1 flag) - for prediction
                [missort_flag],  # Missort flag
                problem_encoding,  # Problem multi-hot encoding
                [is_different_sc, hops_remaining_normalized, is_first_event, is_last_event],  # Additional features
                next_event_type_onehot,  # Next event type one-hot
                [planned_remaining_scaled, planned_duration_scaled, planned_transit_scaled, has_next_plan],  # Edge features as node features (NEW: added planned_transit_scaled)
=======
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
>>>>>>> 43a4a96 (large set 1)
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
        
<<<<<<< HEAD
        package_features = self.package_feature_scaler.transform(package_features).flatten()
        
        # Add route encoding to package features
        package_features_with_route = np.concatenate([package_features, route_encoded.astype(np.float32)])
        
        # Expand to all nodes
        package_features_expanded = np.tile(package_features_with_route, (num_events, 1))
        node_features = np.concatenate([node_features, package_features_expanded], axis=1)
        
        # Edge index (NO FEATURES)
        edge_index = []
=======
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
>>>>>>> 43a4a96 (large set 1)
        
        if num_events > 1:
            for i in range(num_events - 1):
                edge_index.append([i, i+1])
<<<<<<< HEAD
            
            edge_index = np.array(edge_index, dtype=np.int64).T
            # ✅ EASY FIX: Edge features are just constant 0
            num_edges = edge_index.shape[1]
            edge_features = np.zeros((num_edges, 1), dtype=np.float32)  # Single constant feature
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 1), dtype=np.float32)  # Single constant feature
=======
                
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
>>>>>>> 43a4a96 (large set 1)
        
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
<<<<<<< HEAD
            'edge_features': edge_features,  # No edge features
=======
            'edge_continuous_features': edge_continuous_features,
            'edge_categorical_indices': edge_categorical_indices,
>>>>>>> 43a4a96 (large set 1)
            'num_nodes': num_events,
            'package_id': package_data['package_id'],
            'route': route,
            'route_encoded': route_encoded
        }
        
<<<<<<< HEAD
        # ====================================================================
        # Labels: TRANSIT TIME to next event (actual time gap)
        # ====================================================================
=======
        # === Labels (scaled) ===
>>>>>>> 43a4a96 (large set 1)
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
            
<<<<<<< HEAD
            # Create mask with SAME length as nodes
            label_mask = np.zeros(num_events, dtype=bool)
            label_mask[:-1] = True
            result['label_mask'] = label_mask
        # ====================================================================
        
        return result

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
            'location_vocab': len(self.location_encoder.classes_),
            'problem_types_dim': len(self.problem_types),
            'time_features_dim': 4,
            'temporal_features_dim': 6,
            'plan_time_features_dim': 2,
            'next_plan_temporal_features_dim': 7,
            'missort_dim': 1,
            'additional_features_dim': 4,  # is_different_sc, hops_remaining, is_first, is_last
            'edge_as_node_features_dim': 4,  # planned_remaining, planned_duration, planned_transit, has_next_plan (NEW: +1)
            'package_features_dim': 4,
            'route_features_dim': self.max_route_length,
            'edge_features_dim': 1,  # No edge features anymore
            'total_node_features': (
                len(self.event_types) +  # Current event type
                5 +  # sort_center_idx, to_sort_center_idx, carrier_idx, leg_type_idx, ship_method_idx
                4 +  # time_since_start, time_since_prev, position, dwelling_time
                6 +  # temporal features
                2 +  # time_vs_plan_scaled, has_plan_time
                7 +  # next_plan_time temporal features + flag (for prediction)
                1 +  # missort_flag
                len(self.problem_types) +  # problem_encoding
                4 +  # is_different_sc, hops_remaining, is_first, is_last
                len(self.event_types) +  # Next event type
                4 +  # Edge features as node features (planned_remaining, planned_duration, planned_transit, has_next_plan) (NEW: +1)
                4 +  # package physical features
                self.max_route_length  # route encoding
            )
        }
        
=======
            label_mask = np.zeros(num_events, dtype=bool)
            label_mask[:-1] = True
            result['label_mask'] = label_mask
        
        return result
    
    # ==================== Utility Methods ====================
    
>>>>>>> 43a4a96 (large set 1)
    def inverse_transform_time(self, scaled_time: np.ndarray) -> np.ndarray:
        """Convert scaled transit time back to hours"""
        if self.config.data.normalize_time:
            return self.label_time_scaler.inverse_transform(scaled_time)
        return scaled_time
    
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
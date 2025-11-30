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
        self.location_encoder = LabelEncoder()
        
        self.time_scaler = StandardScaler()  # Now scales TRANSIT TIMES (event gaps)
        self.package_feature_scaler = StandardScaler()
        self.plan_time_diff_scaler = StandardScaler()
        self.planned_remaining_scaler = StandardScaler()  # For node feature
        self.planned_duration_scaler = StandardScaler()   # For node feature
        self.planned_transit_scaler = StandardScaler()    # NEW: For planned transit time
        
        self.event_types = config.data.event_types
        self.problem_types = config.data.problem_types
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
        
        # For EXIT events, use previous event's CPT
        if event_type == 'EXIT' and prev_event is not None:
            cpt = prev_event.get('cpt')
            if cpt and cpt != 'null':
                return cpt
        
        # For other events, use their own plan_time
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
                    
                if 'carrier_id' in event and event['carrier_id']:
                    all_carriers.add(str(event['carrier_id']))
                    
                if 'leg_type' in event and event['leg_type']:
                    all_leg_types.add(str(event['leg_type']))
                
                if 'ship_method' in event and event['ship_method']:
                    all_ship_methods.add(str(event['ship_method']))
        
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
        
        # Fit package feature scaler
        package_features = df[['weight', 'length', 'width', 'height']].fillna(0).values
        self.package_feature_scaler.fit(package_features)
        
        self.fitted = True
        return self
    
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
    
    def process_lifecycle(self, package_data: Dict, return_labels: bool = True) -> Dict:
        """Process a single package lifecycle into graph features"""
        
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before processing")
        
        events = package_data['events']
        num_events = len(package_data['events'])
        
        # Validate minimum events
        if num_events < 1:
            raise ValueError("Package must have at least 1 event")
        
        # Extract and encode route (now deduplicated)
        route = self._extract_route(events)
        route_encoded = self._encode_route(route)
        
        node_features = []
        event_times = []
        
        # Process node features
        for i, event in enumerate(events):
            # Event type (one-hot)
            event_type = str(event['event_type'])
            event_type_idx = self.event_type_encoder.transform([event_type])[0]
            event_type_onehot = np.zeros(len(self.event_types))
            event_type_onehot[event_type_idx] = 1
            
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
            event_time = self._parse_datetime(event['event_time'])
            if event_time is None:
                raise ValueError(f"Invalid event_time for event {i}")
            event_times.append(event_time)
            
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
        
        # Add route encoding to package features
        package_features_with_route = np.concatenate([package_features, route_encoded.astype(np.float32)])
        
        # Expand to all nodes
        package_features_expanded = np.tile(package_features_with_route, (num_events, 1))
        node_features = np.concatenate([node_features, package_features_expanded], axis=1)
        
        # Edge index (NO FEATURES)
        edge_index = []
        
        if num_events > 1:
            for i in range(num_events - 1):
                edge_index.append([i, i+1])
            
            edge_index = np.array(edge_index, dtype=np.int64).T
            # ✅ EASY FIX: Edge features are just constant 0
            num_edges = edge_index.shape[1]
            edge_features = np.zeros((num_edges, 1), dtype=np.float32)  # Single constant feature
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 1), dtype=np.float32)  # Single constant feature
        
        result = {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,  # No edge features
            'num_nodes': num_events,
            'package_id': package_data['package_id'],
            'route': route,
            'route_encoded': route_encoded
        }
        
        # ====================================================================
        # Labels: TRANSIT TIME to next event (actual time gap)
        # ====================================================================
        if return_labels:
            labels = []
            for i in range(num_events - 1):
                # Transit time = next_event_time - current_event_time
                transit_hours = (event_times[i+1] - event_times[i]).total_seconds() / 3600
                labels.append(transit_hours)
            
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
        
    def inverse_transform_time(self, scaled_time: np.ndarray) -> np.ndarray:
        """Convert scaled transit time back to hours"""
        if self.config.data.normalize_time:
            return self.time_scaler.inverse_transform(scaled_time)
        return scaled_time
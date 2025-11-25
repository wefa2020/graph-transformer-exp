import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Optional
import json
import ast


class PackageEventValidator:
    """
    Validator and preprocessor for package lifecycle events.
    
    Validates event sequences according to business rules and preprocesses
    events for downstream consumption.
    
    Validation Rules:
    1. All events must have valid event_time
    2. Within each sort center, EXIT must come after INDUCT/LINEHAUL
    3. If EXIT follows LINEHAUL, time difference must be >= 5 minutes
    4. Each sort center must have exactly one INDUCT/LINEHAUL and one EXIT
    5. At each event, abs(event_time - plan_time) must be <= 7 hours
    
    Preprocessing:
    - Events are sorted by sort center and time
    - Problems from EXIT events are moved to previous INDUCT/LINEHAUL events
    - next_plan_time is added to each event:
        * INDUCT/LINEHAUL: next_plan_time = cpt
        * EXIT: next_plan_time = next node's plan_time
        * DELIVERY: next_plan_time = plan_time
    """
    
    def __init__(self, max_time_vs_plan_hours: float = 7.0, min_linehaul_exit_minutes: float = 5.0):
        """
        Initialize validator
        
        Args:
            max_time_vs_plan_hours: Maximum allowed hours difference between event_time and plan_time
            min_linehaul_exit_minutes: Minimum minutes between LINEHAUL and EXIT events
        """
        self.max_time_vs_plan_hours = max_time_vs_plan_hours
        self.min_linehaul_exit_minutes = min_linehaul_exit_minutes
        
        # Statistics tracking
        self.stats = {
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
        
        self._reset_stats()
    
    def _reset_stats(self):
        """Reset statistics counters"""
        for key in self.stats:
            self.stats[key] = 0
    
    @staticmethod
    def parse_datetime(time_value: Union[str, datetime, None]) -> Union[datetime, None]:
        """
        Parse time value which can be:
        - datetime object (already parsed)
        - ISO string
        - None
        
        Args:
            time_value: Time value to parse
            
        Returns:
            datetime object or None
        """
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
    
    @staticmethod
    def parse_problem_field(problem_value) -> List[str]:
        """
        Parse problem field which can be:
        - None
        - "[\"WRONG_NODE\"]" (JSON string)
        - ["WRONG_NODE"] (list)
        
        Args:
            problem_value: Problem value to parse
            
        Returns:
            List of problem strings
        """
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
    
    def validate_and_preprocess_events(self, events: List[Dict]) -> List[Dict]:
        """
        Validate and preprocess events with sorting and problem field movement
        
        Sorting logic:
        1. Group events by sort center (DELIVERY events use "DELIVERY" as their group key)
        2. Order sort centers by their earliest event time (ascending)
        3. Within each sort center, sort events by event_time (ascending)
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Modified list of events with proper ordering and problems moved to previous nodes
            
        Raises:
            ValueError: If validation rules are violated
        """
        if not events:
            return []
        
        # STEP 1: Parse and validate all event times
        for event in events:
            event_time = self.parse_datetime(event.get('event_time'))
            if event_time is None:
                raise ValueError(
                    f"Invalid or missing event_time in event: {event.get('event_type', 'UNKNOWN')} "
                    f"at sort_center: {event.get('sort_center', 'UNKNOWN')}"
                )
            event['_parsed_time'] = event_time
        
        # STEP 2: Group events by sort center
        sort_center_events = {}
        
        for event in events:
            event_type = event.get('event_type')
            
            if event_type == 'DELIVERY':
                sc_key = 'DELIVERY'
            else:
                sort_center = event.get('sort_center')
                
                if not sort_center or sort_center == 'null':
                    sc_key = 'UNKNOWN'
                else:
                    sc_key = str(sort_center)
            
            if sc_key not in sort_center_events:
                sort_center_events[sc_key] = []
            
            sort_center_events[sc_key].append(event)
        
        # STEP 3: Sort events within each sort center by event_time
        for sc_key in sort_center_events:
            sort_center_events[sc_key].sort(key=lambda e: e['_parsed_time'])
        
        # STEP 4: Validate sort center structure
        self._validate_sort_center_structure(sort_center_events)
        
        # STEP 5: Validate event sequences within sort centers
        self._validate_event_sequences_within_sc(sort_center_events)
        
        # STEP 6: Order sort centers by their earliest event time
        sort_center_order = sorted(
            sort_center_events.keys(),
            key=lambda sc: sort_center_events[sc][0]['_parsed_time']
        )
        
        # STEP 7: Reconstruct sorted event list
        sorted_events = []
        for sc_key in sort_center_order:
            sorted_events.extend(sort_center_events[sc_key])
        
        # STEP 8: Validate global event sequence
        self._validate_global_event_sequence(sorted_events)
        
        # STEP 9: Validate time vs plan
        self._validate_time_vs_plan(sorted_events)
        
        # STEP 10: Clean up temporary parsed time field
        for event in sorted_events:
            del event['_parsed_time']
        
        # STEP 11: Create copies and move problems
        processed_events = self._move_exit_problems_to_prev(sorted_events)
        
        # STEP 12: Add next_plan_time to each event
        processed_events = self._add_next_plan_time(processed_events)
        
        return processed_events
    
    def _validate_sort_center_structure(self, sort_center_events: Dict[str, List[Dict]]):
        """
        Validate that each sort center has exactly one INDUCT/LINEHAUL and one EXIT
        
        Args:
            sort_center_events: Dictionary mapping sort center keys to event lists
            
        Raises:
            ValueError: If structure is invalid
        """
        for sc_key, sc_events in sort_center_events.items():
            if sc_key in ['DELIVERY', 'UNKNOWN']:
                continue
            
            induct_linehaul_count = 0
            exit_count = 0
            
            for event in sc_events:
                event_type = event.get('event_type')
                if event_type in ['INDUCT', 'LINEHAUL']:
                    induct_linehaul_count += 1
                elif event_type == 'EXIT':
                    exit_count += 1
            
            if induct_linehaul_count != 1:
                raise ValueError(
                    f"Invalid sort center structure at '{sc_key}': "
                    f"Found {induct_linehaul_count} INDUCT/LINEHAUL events, expected exactly 1. "
                    f"Package has invalid event structure."
                )
            
            if exit_count != 1:
                raise ValueError(
                    f"Invalid sort center structure at '{sc_key}': "
                    f"Found {exit_count} EXIT events, expected exactly 1. "
                    f"Package has invalid event structure."
                )
    
    def _validate_event_sequences_within_sc(self, sort_center_events: Dict[str, List[Dict]]):
        """
        Validate that EXIT comes after INDUCT/LINEHAUL within each sort center
        
        Args:
            sort_center_events: Dictionary mapping sort center keys to event lists
            
        Raises:
            ValueError: If sequence is invalid
        """
        for sc_key, sc_events in sort_center_events.items():
            if sc_key in ['DELIVERY', 'UNKNOWN']:
                continue
            
            induct_linehaul_times = []
            exit_times = []
            
            for event in sc_events:
                event_type = event.get('event_type')
                if event_type in ['INDUCT', 'LINEHAUL']:
                    induct_linehaul_times.append(event['_parsed_time'])
                elif event_type == 'EXIT':
                    exit_times.append(event['_parsed_time'])
            
            if induct_linehaul_times and exit_times:
                max_induct_linehaul_time = max(induct_linehaul_times)
                min_exit_time = min(exit_times)
                
                if min_exit_time <= max_induct_linehaul_time:
                    raise ValueError(
                        f"Invalid event sequence at sort center '{sc_key}': "
                        f"EXIT event at {min_exit_time} is not after all INDUCT/LINEHAUL events "
                        f"(latest INDUCT/LINEHAUL at {max_induct_linehaul_time}). "
                        f"Package has invalid event timing."
                    )
    
    def _validate_global_event_sequence(self, sorted_events: List[Dict]):
        """
        Validate global event sequence (e.g., LINEHAUL-EXIT timing)
        
        Args:
            sorted_events: List of sorted events
            
        Raises:
            ValueError: If sequence is invalid
        """
        for i in range(1, len(sorted_events)):
            prev_event = sorted_events[i-1]
            curr_event = sorted_events[i]
            
            prev_type = prev_event.get('event_type')
            curr_type = curr_event.get('event_type')
            
            if prev_type == 'LINEHAUL' and curr_type == 'EXIT':
                time_diff = (curr_event['_parsed_time'] - prev_event['_parsed_time']).total_seconds() / 60
                
                if time_diff < self.min_linehaul_exit_minutes:
                    raise ValueError(
                        f"Invalid event sequence: EXIT event follows LINEHAUL with only {time_diff:.2f} minutes gap "
                        f"(minimum {self.min_linehaul_exit_minutes} minutes required). "
                        f"LINEHAUL at {prev_event['_parsed_time']}, EXIT at {curr_event['_parsed_time']}. "
                        f"Package has unrealistic timing."
                    )
    
    def _validate_time_vs_plan(self, sorted_events: List[Dict]):
        """
        Validate that abs(event_time - plan_time) <= max_time_vs_plan_hours
        
        Args:
            sorted_events: List of sorted events
            
        Raises:
            ValueError: If time difference is too large
        """
        for i, event in enumerate(sorted_events):
            event_time = event['_parsed_time']
            event_type = event.get('event_type')
            
            # Get appropriate plan_time
            if event_type == 'EXIT' and i > 0:
                prev_event = sorted_events[i-1]
                plan_time_value = prev_event.get('cpt')
            else:
                plan_time_value = event.get('plan_time')
            
            plan_time = self.parse_datetime(plan_time_value)
            
            if plan_time is not None:
                time_diff_hours = abs((event_time - plan_time).total_seconds() / 3600)
                
                if time_diff_hours > self.max_time_vs_plan_hours:
                    raise ValueError(
                        f"Event time vs plan time difference too large at event {i} "
                        f"(type: {event_type}): {time_diff_hours:.2f} hours "
                        f"(maximum {self.max_time_vs_plan_hours} hours allowed). "
                        f"Event time: {event_time}, Plan time: {plan_time}. "
                        f"Package has unrealistic delay or advancement."
                    )
    
    def _move_exit_problems_to_prev(self, sorted_events: List[Dict]) -> List[Dict]:
        """
        Move problems from EXIT events to previous INDUCT/LINEHAUL events
        
        Args:
            sorted_events: List of sorted events
            
        Returns:
            List of processed events with problems moved
            
        Raises:
            ValueError: If EXIT problem cannot be moved
        """
        processed_events = []
        for event in sorted_events:
            event_copy = event.copy()
            processed_events.append(event_copy)
        
        for i, event in enumerate(processed_events):
            if event['event_type'] == 'EXIT':
                problem_value = event.get('problem')
                problems = self.parse_problem_field(problem_value)
                
                if problems:
                    if i == 0:
                        raise ValueError(
                            f"EXIT event at index {i} has problem {problems} but no previous event. "
                            f"Package has invalid event sequence."
                        )
                    
                    prev_event_type = processed_events[i-1]['event_type']
                    if prev_event_type not in ['INDUCT', 'LINEHAUL']:
                        raise ValueError(
                            f"EXIT event at index {i} has problem {problems} but previous event "
                            f"type is '{prev_event_type}'. Expected INDUCT or LINEHAUL. "
                            f"Cannot assign problem to incompatible event type."
                        )
                    
                    processed_events[i-1]['problem'] = problem_value
                    processed_events[i].pop('problem', None)
        
        return processed_events
    
    def _add_next_plan_time(self, sorted_events: List[Dict]) -> List[Dict]:
        """
        Add next_plan_time to each event based on event type:
        - INDUCT/LINEHAUL: next_plan_time = cpt
        - EXIT: next_plan_time = next node's plan_time
        - DELIVERY: next_plan_time = plan_time
        
        Args:
            sorted_events: List of sorted events
            
        Returns:
            List of events with next_plan_time added
        """
        processed_events = []
        
        for i, event in enumerate(sorted_events):
            event_copy = event.copy()
            event_type = event_copy.get('event_type')
            
            if event_type in ['INDUCT', 'LINEHAUL']:
                # For INDUCT/LINEHAUL: next_plan_time = cpt
                next_plan_time = event_copy.get('cpt')
                event_copy['next_plan_time'] = next_plan_time
                
            elif event_type == 'EXIT':
                # For EXIT: next_plan_time = next node's plan_time
                if i + 1 < len(sorted_events):
                    next_event = sorted_events[i + 1]
                    next_plan_time = next_event.get('plan_time')
                    event_copy['next_plan_time'] = next_plan_time
                else:
                    # If EXIT is the last event, set to None
                    event_copy['next_plan_time'] = None
                    
            elif event_type == 'DELIVERY':
                # For DELIVERY: next_plan_time = plan_time
                next_plan_time = event_copy.get('plan_time')
                event_copy['next_plan_time'] = next_plan_time
                
            else:
                # For any other event type, set to None
                event_copy['next_plan_time'] = None
            
            processed_events.append(event_copy)
        
        return processed_events
    
    def _categorize_error(self, error_msg: str) -> str:
        """
        Categorize error message into error type
        
        Args:
            error_msg: Error message string
            
        Returns:
            Error category string
        """
        if 'Invalid or missing event_time' in error_msg:
            return 'invalid_event_time'
        elif 'EXIT event at' in error_msg and 'is not after all INDUCT/LINEHAUL' in error_msg:
            return 'exit_before_induct_linehaul'
        elif 'EXIT event follows LINEHAUL' in error_msg and 'minutes gap' in error_msg:
            return 'linehaul_exit_too_close'
        elif 'Invalid sort center structure' in error_msg:
            return 'invalid_sort_center_structure'
        elif 'EXIT event' in error_msg and 'has problem' in error_msg and 'no previous event' in error_msg:
            return 'exit_problem_no_prev'
        elif 'EXIT event' in error_msg and 'has problem' in error_msg:
            return 'exit_problem_invalid_prev'
        elif 'Event time vs plan time difference too large' in error_msg:
            return 'event_plan_time_diff_too_large'
        else:
            return 'unknown_error'
    
    def filter_packages(self, packages: List[Dict], verbose: bool = True, reset_stats: bool = True) -> Dict:
        """
        Filter packages based on event validation rules
        
        Args:
            packages: List of package dictionaries, each containing:
                - package_id: str
                - events: List[Dict] - list of event dictionaries
                - other package metadata (optional)
            verbose: Whether to print statistics
            reset_stats: Whether to reset statistics before processing
            
        Returns:
            Dictionary containing:
                - valid_packages: List of valid package dictionaries with preprocessed events
                - invalid_packages: List of invalid package dictionaries with error messages
                - stats: Dictionary with filtering statistics
                
        Example:
            >>> validator = PackageEventValidator()
            >>> packages = [
            ...     {'package_id': 'PKG001', 'events': [...], 'weight': 10},
            ...     {'package_id': 'PKG002', 'events': [...], 'weight': 15}
            ... ]
            >>> result = validator.filter_packages(packages)
            >>> valid = result['valid_packages']
            >>> stats = result['stats']
        """
        
        if reset_stats:
            self._reset_stats()
        
        valid_packages = []
        invalid_packages = []
        
        for package in packages:
            self.stats['total_processed'] += 1
            package_id = package.get('package_id', f'unknown_{self.stats["total_processed"]}')
            
            try:
                # Validate and preprocess events
                preprocessed_events = self.validate_and_preprocess_events(package['events'])
                
                # Create a copy of the package with preprocessed events
                valid_package = package.copy()
                valid_package['events'] = preprocessed_events
                valid_packages.append(valid_package)
                
                self.stats['total_valid'] += 1
                
            except ValueError as e:
                error_msg = str(e)
                error_category = self._categorize_error(error_msg)
                
                # Update statistics
                if error_category in self.stats:
                    self.stats[error_category] += 1
                
                # Store invalid package with error information
                invalid_package = {
                    'package_id': package_id,
                    'error_category': error_category,
                    'error_message': error_msg,
                    'original_package': package
                }
                invalid_packages.append(invalid_package)
        
        # Print statistics if verbose
        if verbose:
            self.print_stats()
        
        return {
            'valid_packages': valid_packages,
            'invalid_packages': invalid_packages,
            'stats': self.stats.copy()
        }
    
    def filter_dataframe(self, df: pd.DataFrame, verbose: bool = True, reset_stats: bool = True) -> Dict:
        """
        Filter packages from a pandas DataFrame
        
        Args:
            df: DataFrame with columns including 'package_id' and 'events'
            verbose: Whether to print statistics
            reset_stats: Whether to reset statistics before processing
            
        Returns:
            Dictionary containing:
                - valid_df: DataFrame with valid packages
                - invalid_df: DataFrame with invalid packages and error info
                - stats: Dictionary with filtering statistics
        """
        packages = df.to_dict('records')
        result = self.filter_packages(packages, verbose=verbose, reset_stats=reset_stats)
        
        valid_df = pd.DataFrame(result['valid_packages']) if result['valid_packages'] else pd.DataFrame()
        invalid_df = pd.DataFrame(result['invalid_packages']) if result['invalid_packages'] else pd.DataFrame()
        
        return {
            'valid_df': valid_df,
            'invalid_df': invalid_df,
            'stats': result['stats']
        }
    
    def print_stats(self):
        """Print filtering statistics"""
        print(f"\n{'='*60}")
        print(f"PACKAGE FILTERING STATISTICS")
        print(f"{'='*60}")
        print(f"Total packages processed: {self.stats['total_processed']}")
        print(f"Valid packages: {self.stats['total_valid']}")
        print(f"Filtered packages: {self.stats['total_processed'] - self.stats['total_valid']}")
        
        if self.stats['total_processed'] > 0:
            valid_pct = (self.stats['total_valid'] / self.stats['total_processed']) * 100
            print(f"Valid percentage: {valid_pct:.2f}%")
        
        print(f"\nFiltering reasons:")
        print(f"  - Invalid event time: {self.stats['invalid_event_time']}")
        print(f"  - EXIT before INDUCT/LINEHAUL: {self.stats['exit_before_induct_linehaul']}")
        print(f"  - LINEHAUL-EXIT < {self.min_linehaul_exit_minutes} min: {self.stats['linehaul_exit_too_close']}")
        print(f"  - Invalid sort center structure: {self.stats['invalid_sort_center_structure']}")
        print(f"  - EXIT problem no prev event: {self.stats['exit_problem_no_prev']}")
        print(f"  - EXIT problem invalid prev: {self.stats['exit_problem_invalid_prev']}")
        print(f"  - Event-Plan time diff > {self.max_time_vs_plan_hours}h: {self.stats['event_plan_time_diff_too_large']}")
        print(f"{'='*60}\n")
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return self.stats.copy()
    
    def get_error_summary(self) -> pd.DataFrame:
        """
        Get a summary of errors as a DataFrame
        
        Returns:
            DataFrame with error categories and counts
        """
        error_categories = [
            'invalid_event_time',
            'exit_before_induct_linehaul',
            'linehaul_exit_too_close',
            'invalid_sort_center_structure',
            'exit_problem_no_prev',
            'exit_problem_invalid_prev',
            'event_plan_time_diff_too_large'
        ]
        
        summary = []
        for category in error_categories:
            if self.stats[category] > 0:
                summary.append({
                    'error_category': category,
                    'count': self.stats[category],
                    'percentage': (self.stats[category] / max(1, self.stats['total_processed'])) * 100
                })
        
        return pd.DataFrame(summary)


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage with next_plan_time
    print("="*60)
    print("Example 1: Basic usage with next_plan_time")
    print("="*60)
    
    sample_packages = [
        {
            'package_id': 'PKG001',
            'weight': 10.5,
            'events': [
                {
                    'event_type': 'INDUCT',
                    'event_time': '2024-01-01T10:00:00Z',
                    'plan_time': '2024-01-01T10:00:00Z',
                    'cpt': '2024-01-01T11:00:00Z',
                    'sort_center': 'SC1',
                    'carrier_id': 'C1'
                },
                {
                    'event_type': 'EXIT',
                    'event_time': '2024-01-01T11:00:00Z',
                    'sort_center': 'SC1',
                    'carrier_id': 'C1'
                },
                {
                    'event_type': 'LINEHAUL',
                    'event_time': '2024-01-01T12:00:00Z',
                    'plan_time': '2024-01-01T12:00:00Z',
                    'cpt': '2024-01-01T14:00:00Z',
                    'sort_center': 'SC2',
                    'carrier_id': 'C1'
                },
                {
                    'event_type': 'EXIT',
                    'event_time': '2024-01-01T14:30:00Z',
                    'sort_center': 'SC2',
                    'carrier_id': 'C1'
                },
                {
                    'event_type': 'DELIVERY',
                    'event_time': '2024-01-01T16:00:00Z',
                    'plan_time': '2024-01-01T16:00:00Z',
                    'carrier_id': 'C1'
                }
            ]
        },
        {
            'package_id': 'PKG002',
            'weight': 15.0,
            'events': [
                {
                    'event_type': 'INDUCT',
                    'event_time': '2024-01-01T10:00:00Z',
                    'plan_time': '2024-01-01T10:00:00Z',
                    'cpt': '2024-01-01T11:00:00Z',
                    'sort_center': 'SC1'
                },
                {
                    'event_type': 'EXIT',
                    'event_time': '2024-01-01T09:00:00Z',  # Invalid: EXIT before INDUCT
                    'sort_center': 'SC1'
                }
            ]
        }
    ]
    
    # Create validator
    validator = PackageEventValidator(max_time_vs_plan_hours=7.0, min_linehaul_exit_minutes=5.0)
    
    # Filter packages
    result = validator.filter_packages(sample_packages, verbose=True)
    
    print(f"\nValid packages: {len(result['valid_packages'])}")
    print(f"Invalid packages: {len(result['invalid_packages'])}")
    
    # Access individual results and show next_plan_time
    for pkg in result['valid_packages']:
        print(f"\n  ✓ Valid: {pkg['package_id']}")
        for i, event in enumerate(pkg['events']):
            print(f"    Event {i}: {event['event_type']}")
            print(f"      - event_time: {event.get('event_time')}")
            print(f"      - plan_time: {event.get('plan_time')}")
            print(f"      - cpt: {event.get('cpt')}")
            print(f"      - next_plan_time: {event.get('next_plan_time')}")
    
    for pkg in result['invalid_packages']:
        print(f"  ✗ Invalid: {pkg['package_id']} - {pkg['error_category']}")
    
    # Example 2: Using with DataFrame
    print("\n" + "="*60)
    print("Example 2: DataFrame usage")
    print("="*60)
    
    df = pd.DataFrame(sample_packages)
    result_df = validator.filter_dataframe(df, verbose=False)
    
    print(f"Valid DataFrame shape: {result_df['valid_df'].shape}")
    print(f"Invalid DataFrame shape: {result_df['invalid_df'].shape}")
    
    # Example 3: Get error summary
    print("\n" + "="*60)
    print("Example 3: Error summary")
    print("="*60)
    
    error_summary = validator.get_error_summary()
    if not error_summary.empty:
        print(error_summary.to_string(index=False))
    else:
        print("No errors found!")
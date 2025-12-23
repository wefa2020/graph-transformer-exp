import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional
from gremlin_python.driver import client, serializer
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing
from multiprocessing import Manager
from tqdm import tqdm
import time

# Global variable for process-local Neptune client
_process_client = None
_process_endpoint = None


def init_worker(endpoint: str):
    """Initialize Neptune client for each worker process"""
    global _process_client, _process_endpoint
    _process_endpoint = endpoint
    _process_client = client.Client(
        f'wss://{endpoint}/gremlin',
        'g',
        message_serializer=serializer.GraphSONSerializersV2d0()
    )


def get_process_client():
    """Get or create client for current process"""
    global _process_client, _process_endpoint
    if _process_client is None and _process_endpoint is not None:
        _process_client = client.Client(
            f'wss://{_process_endpoint}/gremlin',
            'g',
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
    return _process_client


def process_single_package(args: Tuple[str, str]) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """
    Process a single package - designed to run in separate process
    
    Args:
        args: Tuple of (package_id, endpoint)
    
    Returns:
        Tuple of (package_data, invalid_reason, package_id)
        - package_data: Dict if valid, None if invalid
        - invalid_reason: 'missing_time', 'invalid_sequence', 'missing_events', or None
        - package_id: The package ID (for tracking invalid packages)
    """
    package_id, endpoint = args
    
    # Get or create client for this process
    gremlin_client = get_process_client()
    if gremlin_client is None:
        # Fallback: create new client if initializer wasn't called
        gremlin_client = client.Client(
            f'wss://{endpoint}/gremlin',
            'g',
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
    
    try:
        # Extract package edges
        package_data = _extract_package_edges(gremlin_client, package_id)
        
        if package_data is None:
            return None, 'missing_time', package_id
        
        # Validate sequence
        is_valid, invalid_reason = _validate_package_sequence(package_data)
        if not is_valid:
            return None, invalid_reason, package_id
        
        # Deduplicate events
        package_data = _deduplicate_events(package_data)
        
        return package_data, None, package_id
        
    except Exception as e:
        print(f"Error processing package {package_id}: {e}")
        return None, 'error', package_id


def _extract_package_edges(gremlin_client, package_id: str) -> Optional[Dict]:
    """Extract all edges for a package including Problem and Missort"""
    try:
        # Escape package_id for query
        escaped_id = package_id.replace("'", "\\'")
        
        # Get package properties
        package_query = f"""
        g.V()
         .has('Package', 'id', '{escaped_id}')
         .elementMap()
        """
        
        result_set = gremlin_client.submit(package_query)
        results = result_set.all().result()
        
        if not results:
            return None
        
        package_props = results[0]
        
        package_data = {
            'package_id': package_id,
            'source_postal': package_props.get('source_postal_code'),
            'dest_postal': package_props.get('destination_postal_code'),
            'pdd': package_props.get('pdd'),
            'weight': package_props.get('weight', 0),
            'length': package_props.get('length', 0),
            'width': package_props.get('width', 0),
            'height': package_props.get('height', 0),
            'events': []
        }
        
        # Get Induct edges
        induct_query = f"""
        g.V()
         .has('Package', 'id', '{escaped_id}')
         .outE('Induct')
         .project('edge', 'sort_center')
         .by(elementMap())
         .by(inV().values('id'))
        """
        
        result_set = gremlin_client.submit(induct_query)
        induct_edges = result_set.all().result()
        
        # Get Exit202 edges
        exit_query = f"""
        g.V()
         .has('Package', 'id', '{escaped_id}')
         .outE('Exit202')
         .project('edge', 'sort_center')
         .by(elementMap())
         .by(inV().values('id'))
        """
        
        result_set = gremlin_client.submit(exit_query)
        exit_edges = result_set.all().result()
        
        # Get LineHaul edges
        linehaul_query = f"""
        g.V()
         .has('Package', 'id', '{escaped_id}')
         .outE('LineHaul')
         .project('edge', 'sort_center')
         .by(elementMap())
         .by(inV().values('id'))
        """
        
        result_set = gremlin_client.submit(linehaul_query)
        linehaul_edges = result_set.all().result()
        
        # Get Problem edges
        problem_query = f"""
        g.V()
         .has('Package', 'id', '{escaped_id}')
         .outE('Problem')
         .project('edge', 'sort_center')
         .by(elementMap())
         .by(inV().values('id'))
        """
        
        result_set = gremlin_client.submit(problem_query)
        problem_edges = result_set.all().result()
        
        # Get Missort edges
        missort_query = f"""
        g.V()
         .has('Package', 'id', '{escaped_id}')
         .outE('Missort')
         .project('edge', 'sort_center')
         .by(elementMap())
         .by(inV().values('id'))
        """
        
        result_set = gremlin_client.submit(missort_query)
        missort_edges = result_set.all().result()
        
        # Get Delivery edge
        delivery_query = f"""
        g.V()
         .has('Package', 'id', '{escaped_id}')
         .outE('Delivery')
         .project('edge', 'delivery_node')
         .by(elementMap())
         .by(inV().elementMap())
        """
        
        result_set = gremlin_client.submit(delivery_query)
        delivery_edges = result_set.all().result()
        
        # Build lookup dictionaries for Problem and Missort
        problems_by_sc = defaultdict(list)
        missorts_by_sc = defaultdict(set)
        
        for edge_data in problem_edges:
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            if event_time:
                problems_by_sc[edge_data['sort_center']].append({
                    'event_time': event_time,
                    'container_problems': edge.get('container_problems', '')
                })
        
        # Sort problems by time for each sort center
        for sc in problems_by_sc:
            problems_by_sc[sc].sort(key=lambda x: x['event_time'])
        
        for edge_data in missort_edges:
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            if event_time:
                missorts_by_sc[edge_data['sort_center']].add(event_time)
        
        # Process Induct edges
        for edge_data in induct_edges:
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            
            if not event_time:
                return None
            
            sort_center = edge_data['sort_center']
            has_missort = sort_center in missorts_by_sc and len(missorts_by_sc[sort_center]) > 0
            
            package_data['events'].append({
                'event_type': 'INDUCT',
                'sort_center': sort_center,
                'event_time': event_time,
                'plan_time': edge.get('plan_time'),
                'cpt': edge.get('cpt'),
                'leg_type': edge.get('leg_type'),
                'carrier_id': edge.get('carrier_id'),
                'load_id': edge.get('load_id'),
                'ship_method': edge.get('ship_method'),
                'missort': has_missort
            })
        
        # Process Exit202 edges
        for edge_data in exit_edges:
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            
            if not event_time:
                return None
            
            sort_center = edge_data['sort_center']
            
            # Find last problem before this exit event
            problem_info = None
            if sort_center in problems_by_sc:
                relevant_problems = [
                    p for p in problems_by_sc[sort_center]
                    if p['event_time'] <= event_time
                ]
                if relevant_problems:
                    problem_info = relevant_problems[-1]['container_problems']
            
            package_data['events'].append({
                'event_type': 'EXIT',
                'sort_center': sort_center,
                'event_time': event_time,
                'dwelling_seconds': edge.get('dwelling_seconds'),
                'leg_type': edge.get('leg_type'),
                'carrier_id': edge.get('carrier_id'),
                'problem': problem_info
            })
        
        # Process LineHaul edges
        for edge_data in linehaul_edges:
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            
            if not event_time:
                return None
            
            sort_center = edge_data['sort_center']
            has_missort = sort_center in missorts_by_sc and len(missorts_by_sc[sort_center]) > 0
            
            package_data['events'].append({
                'event_type': 'LINEHAUL',
                'sort_center': sort_center,
                'event_time': event_time,
                'plan_time': edge.get('plan_time'),
                'cpt': edge.get('cpt'),
                'leg_type': edge.get('leg_type'),
                'carrier_id': edge.get('carrier_id'),
                'ship_method': edge.get('ship_method'),
                'missort': has_missort
            })
        
        # Process Delivery edge
        for edge_data in delivery_edges:
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            
            if not event_time:
                return None
            
            delivery_node = edge_data.get('delivery_node', {})
            
            package_data['events'].append({
                'event_type': 'DELIVERY',
                'event_time': event_time,
                'plan_time': edge.get('plan_time'),
                'delivery_station': edge.get('delivery_station'),
                'ship_method': edge.get('ship_method'),
                'delivery_location': {
                    'id': delivery_node.get('id'),
                    'city': delivery_node.get('city'),
                    'county': delivery_node.get('county'),
                    'state': delivery_node.get('state_id'),
                    'lat': delivery_node.get('lat'),
                    'lng': delivery_node.get('lng')
                }
            })
        
        return package_data
        
    except Exception as e:
        print(f"Error extracting edges for package {package_id}: {e}")
        return None


def _validate_package_sequence(package_data: Dict) -> Tuple[bool, Optional[str]]:
    """Validate that package has proper event sequence"""
    events = package_data['events']
    
    if not events:
        return False, 'missing_events'
    
    event_types = set(e['event_type'] for e in events)
    
    # Must have INDUCT, EXIT, and DELIVERY
    required_events = {'INDUCT', 'EXIT', 'DELIVERY'}
    if not required_events.issubset(event_types):
        return False, 'missing_events'
    
    # Check that we have at least 3 events (minimum lifecycle)
    if len(events) < 3:
        return False, 'missing_events'
    
    # Validate sort center events have matching EXIT/LINEHAUL pairs
    sort_center_events = defaultdict(list)
    for event in events:
        if event['event_type'] in ['INDUCT', 'EXIT', 'LINEHAUL']:
            sc = event.get('sort_center')
            if sc:
                sort_center_events[sc].append(event['event_type'])
    
    # Each sort center should have balanced events
    for sc, sc_events in sort_center_events.items():
        has_entry = 'INDUCT' in sc_events or 'LINEHAUL' in sc_events
        has_exit = 'EXIT' in sc_events
        
        if not (has_entry and has_exit):
            return False, 'invalid_sequence'
    
    return True, None


def _deduplicate_events(package_data: Dict) -> Dict:
    """Deduplicate events"""
    events = package_data['events']
    
    # Sort events by time
    events.sort(key=lambda x: x['event_time'])
    
    # Deduplicate
    seen = set()
    deduped_events = []
    
    for event in events:
        if event['event_type'] == 'DELIVERY':
            if 'DELIVERY' not in seen:
                deduped_events.append(event)
                seen.add('DELIVERY')
        else:
            sc = event.get('sort_center', '')
            key = (sc, event['event_type'])
            
            if key not in seen:
                deduped_events.append(event)
                seen.add(key)
    
    package_data['events'] = deduped_events
    package_data['num_events'] = len(deduped_events)
    
    return package_data


class NeptuneDataExtractor:
    """Extract package lifecycle data from Neptune using Gremlin Client with batch processing"""
    
    def __init__(self, endpoint: str, use_iam: bool = False, max_workers: int = 30):
        self.endpoint = endpoint
        self.use_iam = use_iam
        self.max_workers = max_workers
        
        # Single client for non-parallel operations
        self.main_client = client.Client(
            f'wss://{endpoint}/gremlin',
            'g',
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        
        # Tracking invalid packages (using Manager for process-safe sharing)
        self.manager = Manager()
        self.invalid_packages = {
            'missing_time': self.manager.list(),
            'invalid_sequence': self.manager.list(),
            'missing_events': self.manager.list(),
            'error': self.manager.list()
        }
    
    def count_delivered_packages(
        self,
        start_date: str,
        end_date: str
    ) -> int:
        """Count total number of delivered packages in date range"""
        print("Counting total packages...")
        
        try:
            query = f"""
            g.V()
             .hasLabel('Package')
             .has('is_delivered', 'True')
             .has('has_cycle', false)
             .has('is_return', false)
             .has('delivered_date', gte(datetime('{start_date}')))
             .has('delivered_date', lte(datetime('{end_date}')))
             .count()
            """
            
            result_set = self.main_client.submit(query)
            count = result_set.all().result()[0]
            
            print(f"Total packages to process: {count}")
            return count
            
        except Exception as e:
            print(f"Error counting packages: {e}")
            return 0
    
    def extract_delivered_packages_batch(
        self,
        start_date: str,
        end_date: str,
        skip: int = 0,
        limit: int = 1000
    ) -> List[str]:
        """Extract a batch of delivered package IDs in date range"""
        try:
            query = f"""
            g.V()
             .hasLabel('Package')
             .has('is_delivered', 'True')
             .has('has_cycle', false)
             .has('is_return', false)
             .has('delivered_date', gte(datetime('{start_date}')))
             .has('delivered_date', lte(datetime('{end_date}')))
             .range({skip}, {skip + limit})
             .values('id')
            """
            
            result_set = self.main_client.submit(query)
            package_ids = result_set.all().result()
            
            return package_ids
            
        except Exception as e:
            print(f"Error extracting package batch (skip={skip}): {e}")
            return []
    
    def process_package_batch(self, package_ids: List[str]) -> List[Dict]:
        """Process a batch of packages with multi-processing"""
        
        valid_lifecycles = []
        
        # Prepare arguments for worker processes
        args_list = [(pkg_id, self.endpoint) for pkg_id in package_ids]
        
        # Use ProcessPoolExecutor with initializer
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=init_worker,
            initargs=(self.endpoint,)
        ) as executor:
            # Submit all tasks
            future_to_package = {
                executor.submit(process_single_package, args): args[0]
                for args in args_list
            }
            
            # Process results with progress bar
            with tqdm(total=len(package_ids), desc="Processing batch") as pbar:
                for future in as_completed(future_to_package):
                    try:
                        result, invalid_reason, package_id = future.result()
                        
                        if result is not None:
                            valid_lifecycles.append(result)
                        elif invalid_reason:
                            # Track invalid packages
                            if invalid_reason in self.invalid_packages:
                                self.invalid_packages[invalid_reason].append(package_id)
                                
                    except Exception as e:
                        package_id = future_to_package[future]
                        print(f"\nError processing package {package_id}: {e}")
                        self.invalid_packages['error'].append(package_id)
                    
                    pbar.update(1)
        
        return valid_lifecycles
    
    def save_batch_results(
        self,
        valid_lifecycles: List[Dict],
        batch_num: int,
        output_dir: str
    ):
        """Save batch results to file"""
        
        if not valid_lifecycles:
            return
        
        output_file = os.path.join(output_dir, f'package_lifecycles_batch_{batch_num}.json')
        
        with open(output_file, 'w') as f:
            json.dump(valid_lifecycles, f, indent=2, default=str)
        
        print(f"  ✓ Saved batch {batch_num}: {len(valid_lifecycles)} packages")
    
    def extract_lifecycles(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = 'data/graph-data',
        query_batch_size: int = 10000,
        save_batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Main method to extract package lifecycles with batch processing
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            output_dir: Directory to save results
            query_batch_size: Number of packages to query from Neptune at once
            save_batch_size: Number of packages per output file
        """
        
        print("="*80)
        print("NEPTUNE DATA EXTRACTION (Gremlin Client - Process Pool Mode)")
        print("="*80)
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Query Batch Size: {query_batch_size}")
        print(f"Max Workers (Processes): {self.max_workers}")
        print(f"Output Directory: {output_dir}")
        print("="*80)
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Count total packages
        try:
            total_count = self.count_delivered_packages(start_date, end_date)
        except:
            print("Could not count total packages, proceeding with batch extraction...")
            total_count = None
        
        # Extract and process in batches
        all_valid_lifecycles = []
        batch_num = 0
        skip = 0
        
        print(f"\nExtracting packages in batches of {query_batch_size}...")
        
        while True:
            batch_num += 1
            
            print(f"\n{'='*80}")
            print(f"BATCH {batch_num} (offset: {skip})")
            print(f"{'='*80}")
            
            # Step 1: Get batch of package IDs
            print(f"Fetching package IDs (skip={skip}, limit={query_batch_size})...")
            package_ids = self.extract_delivered_packages_batch(
                start_date=start_date,
                end_date=end_date,
                skip=skip,
                limit=query_batch_size
            )
            
            if not package_ids:
                print("No more packages to process")
                break
            
            print(f"Retrieved {len(package_ids)} package IDs")
            
            # Step 2: Process this batch
            print(f"Processing {len(package_ids)} packages...")
            valid_lifecycles = self.process_package_batch(package_ids)
            
            print(f"✓ Processed {len(valid_lifecycles)} valid packages from this batch")
            
            # Step 3: Save batch results
            self.save_batch_results(valid_lifecycles, batch_num, output_dir)
            
            # Add to overall results
            all_valid_lifecycles.extend(valid_lifecycles)
            
            # Print progress
            print(f"\nProgress Summary:")
            print(f"  - Batches processed: {batch_num}")
            print(f"  - Packages queried: {skip + len(package_ids)}")
            print(f"  - Valid lifecycles: {len(all_valid_lifecycles)}")
            if total_count:
                progress = ((skip + len(package_ids)) / total_count) * 100
                print(f"  - Overall progress: {progress:.1f}%")
            
            # Check if we got fewer packages than requested (last batch)
            if len(package_ids) < query_batch_size:
                print("\nReached last batch")
                break
            
            # Move to next batch
            skip += query_batch_size
            
            # Small delay to avoid overwhelming Neptune
            time.sleep(0.5)
        
        # Print final statistics
        print(f"\n{'='*80}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total batches processed: {batch_num}")
        print(f"Total valid packages: {len(all_valid_lifecycles)}")
        
        # Print invalid package statistics
        print("\nInvalid Packages Summary:")
        print(f"  - Missing time fields: {len(self.invalid_packages['missing_time'])}")
        print(f"  - Invalid sequence: {len(self.invalid_packages['invalid_sequence'])}")
        print(f"  - Missing events: {len(self.invalid_packages['missing_events'])}")
        print(f"  - Errors: {len(self.invalid_packages['error'])}")
        
        total_invalid = (
            len(self.invalid_packages['missing_time']) +
            len(self.invalid_packages['invalid_sequence']) +
            len(self.invalid_packages['missing_events']) +
            len(self.invalid_packages['error'])
        )
        print(f"  - Total invalid: {total_invalid}")
        
        total_processed = len(all_valid_lifecycles) + total_invalid
        if total_processed > 0:
            print(f"  - Success rate: {len(all_valid_lifecycles)/total_processed*100:.2f}%")
        
        # Save complete dataset
        print(f"\nSaving complete dataset...")
        complete_file = os.path.join(output_dir, 'package_lifecycles.json')
        with open(complete_file, 'w') as f:
            json.dump(all_valid_lifecycles, f, indent=2, default=str)
        print(f"  ✓ Saved complete dataset: {len(all_valid_lifecycles)} packages")
        
        # Save invalid package IDs
        invalid_file = os.path.join(output_dir, 'invalid_packages.json')
        with open(invalid_file, 'w') as f:
            json.dump({
                'missing_time': list(self.invalid_packages['missing_time']),
                'invalid_sequence': list(self.invalid_packages['invalid_sequence']),
                'missing_events': list(self.invalid_packages['missing_events']),
                'error': list(self.invalid_packages['error'])
            }, f, indent=2)
        print(f"  ✓ Saved invalid package IDs")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_valid_lifecycles)
        
        # Save CSV summary
        if len(df) > 0:
            summary_file = os.path.join(output_dir, 'package_summary.csv')
            summary_df = df[['package_id', 'source_postal', 'dest_postal', 'weight', 'num_events']].copy()
            summary_df.to_csv(summary_file, index=False)
            print(f"  ✓ Saved summary CSV")
        
        print(f"\n{'='*80}")
        print(f"Output directory: {output_dir}")
        print(f"Main file: {complete_file}")
        print(f"{'='*80}")
        
        return df
    
    def load_from_json(self, json_path: str, output_dir: str = '/home/ubuntu/graph-transformer-exp/data/graph-data') -> pd.DataFrame:
        """Load extracted data from JSON file"""
        
        with open(f"{output_dir}/{json_path}", 'r') as f:
            data = json.load(f)

        validator = PackageEventValidator(max_time_vs_plan_hours=800.0, min_linehaul_exit_minutes=5.0)
        result = validator.filter_packages(data, verbose=True)

        return pd.DataFrame(result['valid_packages'])
    
    def close(self):
        """Close all connections"""
        try:
            self.main_client.close()
        except:
            pass
        
        # Shutdown manager
        try:
            self.manager.shutdown()
        except:
            pass


# Helper functions
def datetime_to_epoch_ms(dt_str: str) -> int:
    """Convert ISO datetime string to epoch milliseconds"""
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    return int(dt.timestamp() * 1000)


def epoch_ms_to_datetime(epoch_ms: int) -> str:
    """Convert epoch milliseconds to ISO datetime string"""
    dt = datetime.fromtimestamp(epoch_ms / 1000.0)
    return dt.isoformat() + 'Z'


# Example usage
if __name__ == "__main__":
    # Configuration
    NEPTUNE_ENDPOINT = "swa-shipgraph-neptune-instance-prod-us-east-1-read-replica.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182"
    START_DATE = "2025-11-10T00:00:00Z"
    END_DATE = "2025-12-10T00:00:00Z"
    
    # Initialize extractor
    extractor = NeptuneDataExtractor(
        endpoint=NEPTUNE_ENDPOINT,
        max_workers=30  # Number of processes
    )
    
    try:
        # Extract lifecycles with batch processing
        df = extractor.extract_lifecycles(
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir='data/graph-data',
            query_batch_size=10000,
            save_batch_size=1000
        )
        
        print(f"\nExtracted {len(df)} package lifecycles")
        if len(df) > 0:
            print("\nSample data:")
            print(df.head())
            print(f"\nColumns: {df.columns.tolist()}")
            
            # Print sample event with missort/problem info
            if len(df) > 0:
                sample_events = df.iloc[0]['events']
                print("\nSample events with missort/problem info:")
                for event in sample_events[:3]:
                    print(f"  {event['event_type']} at {event.get('sort_center', 'N/A')}")
                    if 'missort' in event:
                        print(f"    - Missort: {event['missort']}")
                    if 'problem' in event:
                        print(f"    - Problem: {event['problem']}")
        
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user")
        print("Partial results have been saved to data/graph-data/")
    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        extractor.close()
        print("\nClosed all connections")
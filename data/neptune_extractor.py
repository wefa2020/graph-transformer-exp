import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional
from gremlin_python.driver import client, serializer
from gremlin_python.driver.protocol import GremlinServerError
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing
from multiprocessing import Manager
from tqdm import tqdm
import time
import random
from functools import wraps

# Global variable for process-local Neptune client
_process_client = None
_process_endpoint = None
_process_id = None


def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def init_worker(endpoint: str, worker_id: int = 0):
    """Initialize Neptune client for each worker process"""
    global _process_client, _process_endpoint, _process_id
    _process_endpoint = endpoint
    _process_id = worker_id
    _process_client = None  # Lazy initialization


def get_process_client():
    """Get or create client for current process with proper timeout settings"""
    global _process_client, _process_endpoint
    if _process_client is None and _process_endpoint is not None:
        try:
            _process_client = client.Client(
                f'wss://{_process_endpoint}/gremlin',
                'g',
                message_serializer=serializer.GraphSONSerializersV2d0(),
                # Connection settings to prevent timeouts
                pool_size=2,
                max_workers=2,
                # Timeout settings (in seconds)
                connection_timeout=30,
                # Keep connection alive
                # Enable compression for large results
            )
        except Exception as e:
            print(f"Failed to create client: {e}")
            _process_client = None
    return _process_client


def close_process_client():
    """Close the process-local client"""
    global _process_client
    if _process_client is not None:
        try:
            _process_client.close()
        except:
            pass
        _process_client = None


@retry_with_backoff(max_retries=3, base_delay=0.5, max_delay=10.0)
def execute_query(gremlin_client, query: str, timeout_ms: int = 120000):
    """Execute a Gremlin query with retry logic"""
    try:
        result_set = gremlin_client.submit(query)
        return result_set.all().result()
    except Exception as e:
        error_str = str(e).lower()
        # Check for timeout-related errors
        if 'timeout' in error_str or 'timed out' in error_str:
            # Close and recreate client on timeout
            close_process_client()
        raise


def process_single_package(args: Tuple[str, str]) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """
    Process a single package - designed to run in separate process
    """
    package_id, endpoint = args
    
    gremlin_client = get_process_client()
    if gremlin_client is None:
        return None, 'connection_error', package_id
    
    try:
        # Extract package edges using OPTIMIZED single query
        package_data = _extract_package_edges_optimized(gremlin_client, package_id)
        
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
        error_str = str(e).lower()
        if 'timeout' in error_str:
            return None, 'timeout', package_id
        print(f"Error processing package {package_id}: {e}")
        return None, 'error', package_id


def _extract_package_edges_optimized(gremlin_client, package_id: str) -> Optional[Dict]:
    """
    Extract all edges for a package using a SINGLE optimized query
    This reduces 7 queries to 2 queries, dramatically improving performance
    """
    try:
        escaped_id = package_id.replace("'", "\\'")
        
        # OPTIMIZED: Single query to get package properties AND all edges
        combined_query = f"""
        g.V()
         .has('Package', 'id', '{escaped_id}')
         .project('package', 'induct', 'exit', 'linehaul', 'problem', 'missort', 'delivery')
         .by(elementMap())
         .by(outE('Induct').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
         .by(outE('Exit202').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
         .by(outE('LineHaul').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
         .by(outE('Problem').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
         .by(outE('Missort').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
         .by(outE('Delivery').project('edge', 'node').by(elementMap()).by(inV().elementMap()).fold())
        """
        
        results = execute_query(gremlin_client, combined_query)
        
        if not results:
            return None
        
        data = results[0]
        package_props = data['package']
        
        package_data = {
            'package_id': package_id,
            'source_postal': package_props.get('source_postal_code'),
            'dest_postal': package_props.get('destination_postal_code'),
            'leg_plan': package_props.get('leg_plan'),
            'pdd': package_props.get('pdd'),
            'weight': package_props.get('weight', 0),
            'length': package_props.get('length', 0),
            'width': package_props.get('width', 0),
            'height': package_props.get('height', 0),
            'events': []
        }
        
        # Build lookup dictionaries for Problem and Missort
        problems_by_sc = defaultdict(list)
        missorts_by_sc = defaultdict(set)
        
        for edge_data in data.get('problem', []):
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            if event_time:
                problems_by_sc[edge_data['sc']].append({
                    'event_time': event_time,
                    'container_problems': edge.get('container_problems', '')
                })
        
        for sc in problems_by_sc:
            problems_by_sc[sc].sort(key=lambda x: x['event_time'])
        
        for edge_data in data.get('missort', []):
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            if event_time:
                missorts_by_sc[edge_data['sc']].add(event_time)
        
        # Process Induct edges
        for edge_data in data.get('induct', []):
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            if not event_time:
                return None
            
            sort_center = edge_data['sc']
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
        for edge_data in data.get('exit', []):
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            if not event_time:
                return None
            
            sort_center = edge_data['sc']
            
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
        for edge_data in data.get('linehaul', []):
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            if not event_time:
                return None
            
            sort_center = edge_data['sc']
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
        for edge_data in data.get('delivery', []):
            edge = edge_data['edge']
            event_time = edge.get('event_time')
            if not event_time:
                return None
            
            delivery_node = edge_data.get('node', {})
            
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
        raise


def _validate_package_sequence(package_data: Dict) -> Tuple[bool, Optional[str]]:
    """Validate that package has proper event sequence"""
    events = package_data['events']
    
    if not events:
        return False, 'missing_events'
    
    event_types = set(e['event_type'] for e in events)
    
    required_events = {'INDUCT', 'EXIT', 'DELIVERY'}
    if not required_events.issubset(event_types):
        return False, 'missing_events'
    
    if len(events) < 3:
        return False, 'missing_events'
    
    sort_center_events = defaultdict(list)
    for event in events:
        if event['event_type'] in ['INDUCT', 'EXIT', 'LINEHAUL']:
            sc = event.get('sort_center')
            if sc:
                sort_center_events[sc].append(event['event_type'])
    
    for sc, sc_events in sort_center_events.items():
        has_entry = 'INDUCT' in sc_events or 'LINEHAUL' in sc_events
        has_exit = 'EXIT' in sc_events
        
        if not (has_entry and has_exit):
            return False, 'invalid_sequence'
    
    return True, None


def _deduplicate_events(package_data: Dict) -> Dict:
    """Deduplicate events"""
    events = package_data['events']
    events.sort(key=lambda x: x['event_time'])
    
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


def process_package_batch_worker(args):
    """Worker function for processing a batch of packages in a single process"""
    package_ids, endpoint = args
    results = []
    
    for pkg_id in package_ids:
        result = process_single_package((pkg_id, endpoint))
        results.append(result)
    
    return results


class NeptuneDataExtractor:
    """Extract package lifecycle data from Neptune with optimized parallel processing"""
    
    def __init__(self, endpoint: str, use_iam: bool = False, max_workers: int = None):
        self.endpoint = endpoint
        self.use_iam = use_iam
        
        # Use ALL CPUs if not specified
        if max_workers is None:
            self.max_workers = os.cpu_count() or 4
        else:
            self.max_workers = max_workers
        
        print(f"Initializing with {self.max_workers} workers (CPUs available: {os.cpu_count()})")
        
        # Main client for non-parallel operations
        self.main_client = client.Client(
            f'wss://{endpoint}/gremlin',
            'g',
            message_serializer=serializer.GraphSONSerializersV2d0(),
            pool_size=4,
            max_workers=4,
        )
        
        # Manager for process-safe sharing
        self.manager = Manager()
        self.invalid_packages = {
            'missing_time': self.manager.list(),
            'invalid_sequence': self.manager.list(),
            'missing_events': self.manager.list(),
            'timeout': self.manager.list(),
            'connection_error': self.manager.list(),
            'error': self.manager.list()
        }
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'failed_queries': 0,
            'retry_count': 0
        }
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=15.0)
    def count_delivered_packages(self, start_date: str, end_date: str) -> int:
        """Count total number of delivered packages in date range"""
        print("Counting total packages...")
        
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
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=15.0)
    def extract_delivered_packages_batch(
        self,
        start_date: str,
        end_date: str,
        skip: int = 0,
        limit: int = 1000
    ) -> List[str]:
        """Extract a batch of delivered package IDs in date range"""
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
    
    def process_package_batch(self, package_ids: List[str]) -> List[Dict]:
        """Process a batch of packages with multi-processing - OPTIMIZED VERSION"""
        
        valid_lifecycles = []
        
        # Calculate optimal chunk size for each worker
        # Each worker gets a mini-batch to process
        chunk_size = max(1, len(package_ids) // (self.max_workers * 4))
        chunk_size = min(chunk_size, 50)  # Cap at 50 packages per mini-batch
        
        # Split packages into chunks
        chunks = [
            package_ids[i:i + chunk_size] 
            for i in range(0, len(package_ids), chunk_size)
        ]
        
        # Prepare arguments
        args_list = [(chunk, self.endpoint) for chunk in chunks]
        
        # Use ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=init_worker,
            initargs=(self.endpoint,)
        ) as executor:
            # Submit chunk processing tasks
            futures = []
            for chunk in chunks:
                # Submit individual packages within each chunk
                for pkg_id in chunk:
                    future = executor.submit(
                        process_single_package, 
                        (pkg_id, self.endpoint)
                    )
                    futures.append(future)
            
            # Process results with progress bar
            with tqdm(total=len(package_ids), desc="Processing batch") as pbar:
                for future in as_completed(futures):
                    try:
                        result, invalid_reason, package_id = future.result(timeout=60)
                        
                        if result is not None:
                            valid_lifecycles.append(result)
                        elif invalid_reason:
                            if invalid_reason in self.invalid_packages:
                                self.invalid_packages[invalid_reason].append(package_id)
                                
                    except TimeoutError:
                        self.invalid_packages['timeout'].append('unknown')
                    except Exception as e:
                        print(f"\nError: {e}")
                        self.invalid_packages['error'].append('unknown')
                    
                    pbar.update(1)
        
        return valid_lifecycles
    
    def process_package_batch_threaded(self, package_ids: List[str]) -> List[Dict]:
        """
        Alternative: Process using ThreadPoolExecutor 
        Better for I/O bound operations like Neptune queries
        """
        valid_lifecycles = []
        
        # For threads, we can use more workers since they share memory
        # and are better for I/O-bound tasks
        num_threads = self.max_workers 
        
        # Create a shared client for all threads in this batch
        batch_client = client.Client(
            f'wss://{self.endpoint}/gremlin',
            'g',
            message_serializer=serializer.GraphSONSerializersV2d0(),
            pool_size=num_threads,
            max_workers=num_threads,
        )
        
        def process_package_thread(package_id: str) -> Tuple[Optional[Dict], Optional[str], str]:
            """Process a single package in a thread"""
            try:
                package_data = _extract_package_edges_optimized(batch_client, package_id)
                
                if package_data is None:
                    return None, 'missing_time', package_id
                
                is_valid, invalid_reason = _validate_package_sequence(package_data)
                if not is_valid:
                    return None, invalid_reason, package_id
                
                package_data = _deduplicate_events(package_data)
                return package_data, None, package_id
                
            except Exception as e:
                error_str = str(e).lower()
                if 'timeout' in error_str:
                    return None, 'timeout', package_id
                return None, 'error', package_id
        
        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {
                    executor.submit(process_package_thread, pkg_id): pkg_id 
                    for pkg_id in package_ids
                }
                
                with tqdm(total=len(package_ids), desc="Processing batch (threaded)") as pbar:
                    for future in as_completed(futures):
                        try:
                            result, invalid_reason, package_id = future.result(timeout=120)
                            
                            if result is not None:
                                valid_lifecycles.append(result)
                            elif invalid_reason:
                                if invalid_reason in self.invalid_packages:
                                    self.invalid_packages[invalid_reason].append(package_id)
                                    
                        except TimeoutError:
                            pkg_id = futures[future]
                            self.invalid_packages['timeout'].append(pkg_id)
                        except Exception as e:
                            pkg_id = futures[future]
                            self.invalid_packages['error'].append(pkg_id)
                        
                        pbar.update(1)
        finally:
            batch_client.close()
        
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
        output_dir: str,
        query_batch_size: int = 5000,
        save_batch_size: int = 1000,
        use_threads: bool = True  # New parameter - threads are better for I/O
    ) -> pd.DataFrame:
        """
        Main method to extract package lifecycles with batch processing
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            output_dir: Directory to save results
            query_batch_size: Number of packages to query from Neptune at once
            save_batch_size: Number of packages per output file
            use_threads: Use ThreadPoolExecutor (better for I/O) vs ProcessPoolExecutor
        """
        
        print("="*80)
        print("NEPTUNE DATA EXTRACTION (Optimized)")
        print("="*80)
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Query Batch Size: {query_batch_size}")
        print(f"Max Workers: {self.max_workers}")
        print(f"Mode: {'Threaded (I/O optimized)' if use_threads else 'Multi-process'}")
        print(f"Output Directory: {output_dir}")
        print("="*80)
        print()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Count total packages
        try:
            total_count = self.count_delivered_packages(start_date, end_date)
        except Exception as e:
            print(f"Could not count total packages: {e}")
            total_count = None
        
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
            
            try:
                package_ids = self.extract_delivered_packages_batch(
                    start_date=start_date,
                    end_date=end_date,
                    skip=skip,
                    limit=query_batch_size
                )
            except Exception as e:
                print(f"Error fetching batch: {e}")
                print("Waiting 10 seconds before retry...")
                time.sleep(10)
                continue
            
            if not package_ids:
                print("No more packages to process")
                break
            
            print(f"Retrieved {len(package_ids)} package IDs")
            
            # Step 2: Process this batch
            print(f"Processing {len(package_ids)} packages...")
            
            if use_threads:
                valid_lifecycles = self.process_package_batch_threaded(package_ids)
            else:
                valid_lifecycles = self.process_package_batch(package_ids)
            
            print(f"✓ Processed {len(valid_lifecycles)} valid packages from this batch")
            
            # Step 3: Save batch results
            self.save_batch_results(valid_lifecycles, batch_num, output_dir)
            
            all_valid_lifecycles.extend(valid_lifecycles)
            
            # Print progress
            print(f"\nProgress Summary:")
            print(f"  - Batches processed: {batch_num}")
            print(f"  - Packages queried: {skip + len(package_ids)}")
            print(f"  - Valid lifecycles: {len(all_valid_lifecycles)}")
            print(f"  - Invalid (timeout): {len(self.invalid_packages['timeout'])}")
            print(f"  - Invalid (error): {len(self.invalid_packages['error'])}")
            
            if total_count:
                progress = ((skip + len(package_ids)) / total_count) * 100
                print(f"  - Overall progress: {progress:.1f}%")
            
            if len(package_ids) < query_batch_size:
                print("\nReached last batch")
                break
            
            skip += query_batch_size
            
            # Small delay between batches to avoid overwhelming Neptune
            time.sleep(0.2)
        
        # Print final statistics
        self._print_final_stats(all_valid_lifecycles, batch_num, output_dir)
        
        return pd.DataFrame(all_valid_lifecycles)
    
    def _print_final_stats(self, all_valid_lifecycles: List[Dict], batch_num: int, output_dir: str):
        """Print final extraction statistics"""
        print(f"\n{'='*80}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total batches processed: {batch_num}")
        print(f"Total valid packages: {len(all_valid_lifecycles)}")
        
        print("\nInvalid Packages Summary:")
        for reason, packages in self.invalid_packages.items():
            print(f"  - {reason}: {len(packages)}")
        
        total_invalid = sum(len(v) for v in self.invalid_packages.values())
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
            json.dump({k: list(v) for k, v in self.invalid_packages.items()}, f, indent=2)
        print(f"  ✓ Saved invalid package IDs")
    
    def close(self):
        """Close all connections"""
        try:
            self.main_client.close()
        except:
            pass
        
        try:
            self.manager.shutdown()
        except:
            pass


# Example usage
if __name__ == "__main__":
    NEPTUNE_ENDPOINT = "swa-shipgraph-neptune-instance-prod-us-east-1-read-replica.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182"
    START_DATE = "2025-12-10T00:00:00Z"
    END_DATE = "2026-01-06T00:00:00Z"
    
    # Use ALL CPUs
    num_cpus = os.cpu_count() - 1
    print(f"System has {num_cpus} CPUs")
    
    extractor = NeptuneDataExtractor(
        endpoint=NEPTUNE_ENDPOINT,
        max_workers=num_cpus  # Use all CPUs
    )
    
    try:
        df = extractor.extract_lifecycles(
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir='data/graph-data/12100106/',
            query_batch_size=5000,  # Larger batches since we have fewer queries per package
            save_batch_size=1000,
            use_threads=True  # Use threads - better for I/O bound Neptune queries
        )
        
        print(f"\nExtracted {len(df)} package lifecycles")
        
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user")
    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        extractor.close()
        print("\nClosed all connections")
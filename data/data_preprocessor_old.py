import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler

class PackageLifecyclePreprocessor:
    """Preprocess package lifecycle data for graph transformer"""
    
    def __init__(self, config):
        self.config = config
        
        self.event_type_encoder = LabelEncoder()
        self.sort_center_encoder = LabelEncoder()
        self.carrier_encoder = LabelEncoder()
        self.leg_type_encoder = LabelEncoder()
        
        self.time_scaler = StandardScaler()
        self.package_feature_scaler = StandardScaler()
        
        self.event_types = config.data.event_types
        self.fitted = False
    
    def fit(self, df: pd.DataFrame):
        """Fit encoders and scalers on training data"""
        
        all_sort_centers = set()
        all_carriers = set()
        all_leg_types = set()
        
        for _, row in df.iterrows():
            events = row['events']
            
            for event in events:
                if 'sort_center' in event and event['sort_center']:
                    all_sort_centers.add(str(event['sort_center']))  # Convert to string here
                    
                if 'carrier_id' in event and event['carrier_id']:
                    all_carriers.add(str(event['carrier_id']))  # Convert to string here
                    
                if 'leg_type' in event and event['leg_type']:
                    all_leg_types.add(str(event['leg_type']))  # Convert to string here
        
        # Add unknown tokens
        all_sort_centers.add('UNKNOWN')
        all_carriers.add('UNKNOWN')
        all_leg_types.add('UNKNOWN')
        
        # Fit encoders
        self.event_type_encoder.fit(self.event_types)
        self.sort_center_encoder.fit(sorted(list(all_sort_centers)))
        self.carrier_encoder.fit(sorted(list(all_carriers)))
        self.leg_type_encoder.fit(sorted(list(all_leg_types)))
        
        # Fit time scaler
        time_deltas = []
        for _, row in df.iterrows():
            events = row['events']
            for i in range(1, len(events)):
                try:
                    prev_time = datetime.fromisoformat(str(events[i-1]['event_time']).replace('Z', '+00:00'))
                    curr_time = datetime.fromisoformat(str(events[i]['event_time']).replace('Z', '+00:00'))
                    delta = (curr_time - prev_time).total_seconds() / 3600
                    time_deltas.append([delta])
                except Exception as e:
                    continue
        
        if time_deltas:
            self.time_scaler.fit(np.array(time_deltas))
        
        # Fit package feature scaler
        package_features = df[['weight', 'length', 'width', 'height']].fillna(0).values
        self.package_feature_scaler.fit(package_features)
        
        self.fitted = True
        return self
    
    def process_lifecycle(self, package_data: Dict, return_labels: bool = True) -> Dict:
        """Process a single package lifecycle into graph features"""
        
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before processing")
        
        events = package_data['events']
        num_events = len(events)
        
        node_features = []
        event_times = []
        
        for i, event in enumerate(events):
            # Event type (one-hot)
            event_type = str(event['event_type'])  # Convert to string
            event_type_idx = self.event_type_encoder.transform([event_type])[0]
            event_type_onehot = np.zeros(len(self.event_types))
            event_type_onehot[event_type_idx] = 1
            
            # Sort center - Convert to string here
            sort_center = event.get('sort_center', 'UNKNOWN')
            sort_center = str(sort_center) if sort_center else 'UNKNOWN'  # Convert to string
            if sort_center not in self.sort_center_encoder.classes_:
                sort_center = 'UNKNOWN'
            sort_center_idx = self.sort_center_encoder.transform([sort_center])[0]
            
            # Carrier - Convert to string here
            carrier = event.get('carrier_id', 'UNKNOWN')
            carrier = str(carrier) if carrier else 'UNKNOWN'  # Convert to string
            if carrier not in self.carrier_encoder.classes_:
                carrier = 'UNKNOWN'
            carrier_idx = self.carrier_encoder.transform([carrier])[0]
            
            # Leg type - Convert to string here
            leg_type = event.get('leg_type', 'UNKNOWN')
            leg_type = str(leg_type) if leg_type else 'UNKNOWN'  # Convert to string
            if leg_type not in self.leg_type_encoder.classes_:
                leg_type = 'UNKNOWN'
            leg_type_idx = self.leg_type_encoder.transform([leg_type])[0]
            
            # Event time
            event_time = datetime.fromisoformat(str(event['event_time']).replace('Z', '+00:00'))
            event_times.append(event_time)
            
            if i == 0:
                time_since_start = 0
                time_since_prev = 0
            else:
                time_since_start = (event_time - event_times[0]).total_seconds() / 3600
                time_since_prev = (event_time - event_times[i-1]).total_seconds() / 3600
            
            # Positional encoding
            position = i / num_events
            
            # Dwelling time (for EXIT events)
            dwelling_time = event.get('dwelling_seconds', 0) / 3600 if event.get('dwelling_seconds') else 0
            
            # Combine features
            features = np.concatenate([
                event_type_onehot,
                [sort_center_idx, carrier_idx, leg_type_idx],
                [time_since_start, time_since_prev, position, dwelling_time]
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
        
        for i in range(num_events - 1):
            edge_index.append([i, i+1])
            
            # Time delta as edge feature
            time_delta = (event_times[i+1] - event_times[i]).total_seconds() / 3600
            
            # Same sort center flag
            same_sc = int(events[i].get('sort_center') == events[i+1].get('sort_center'))
            
            # Same carrier flag
            same_carrier = int(events[i].get('carrier_id') == events[i+1].get('carrier_id'))
            
            edge_features.append([time_delta, same_sc, same_carrier])
        
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_features = np.array(edge_features, dtype=np.float32)
        
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
            
            labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
            
            if self.config.data.normalize_time:
                labels = self.time_scaler.transform(labels)
            
            result['labels'] = labels
            # ✅ FIX: Create mask with SAME length as nodes
            label_mask = np.zeros(num_events, dtype=bool)
            label_mask[:-1] = True  # All True except last node
            result['label_mask'] = label_mask  # Shape: (num_events,) ← MUST match node count!
        
        return result
    
    def inverse_transform_time(self, scaled_time: np.ndarray) -> np.ndarray:
        """Convert scaled time back to hours"""
        if self.config.data.normalize_time:
            return self.time_scaler.inverse_transform(scaled_time)
        return scaled_time
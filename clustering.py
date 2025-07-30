import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from config import TRACK_CLUSTERING_PARAMS, DEFAULT_CLUSTERING_PARAMS, FEATURE_COLUMNS


def cluster_track_data(data: pd.DataFrame, track_name: str, n_samples_per_driver: int = 800) -> pd.DataFrame:
    """Apply DBSCAN clustering to track-specific data."""
    # Sample data evenly across drivers
    sampled_data = []
    for driver in data['Driver'].unique():
        driver_data = data[data['Driver'] == driver]
        if len(driver_data) > n_samples_per_driver:
            driver_sample = driver_data.sample(n_samples_per_driver, random_state=42)
        else:
            driver_sample = driver_data
        sampled_data.append(driver_sample)
    
    clustered_data = pd.concat(sampled_data, ignore_index=True)
    
    # Prepare features for clustering
    features_matrix = clustered_data[FEATURE_COLUMNS].values
    
    # Handle infinite and NaN values
    features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=999999, neginf=-999999)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_matrix)
    
    # Get track-specific parameters
    params = TRACK_CLUSTERING_PARAMS.get(track_name, DEFAULT_CLUSTERING_PARAMS)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    clusters = dbscan.fit_predict(features_scaled)
    
    clustered_data['Cluster'] = clusters
    
    # Print clustering results
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"{track_name} Clustering: {n_clusters} patterns, {n_noise} noise points ({n_noise/len(clusters)*100:.1f}%)")
    
    return clustered_data


def analyze_driving_style_similarity(clustered_data: pd.DataFrame, track_name: str = "Track") -> tuple:
    """Analyze driving style similarities between drivers."""
    driver_profiles = {}
    
    for driver in clustered_data['Driver'].unique():
        driver_data = clustered_data[clustered_data['Driver'] == driver]
        
        if len(driver_data) == 0:
            continue
            
        # Create comprehensive driving style profile with safe calculations
        profile = {
            'avg_speed': driver_data['Speed'].mean(),
            'speed_variability': driver_data['Speed'].std(),
            'max_speed': driver_data['Speed'].max(),
            'throttle_aggression': driver_data['Throttle'].mean(),
            'throttle_smoothness': 1 / (driver_data['ThrottleRate'].mean() + 0.001),
            'brake_frequency': driver_data['nBrake'].mean(),
            'brake_intensity': driver_data['BrakeIntensity'].mean(),
            'gear_efficiency': driver_data['GearEfficiency'].mean(),
            'acceleration_pattern': driver_data['Acceleration'].mean(),
            'acceleration_variability': driver_data['Acceleration'].std(),
        }
        
        # Safe cornering and straight line calculations
        low_speed_data = driver_data[driver_data['Speed'] < driver_data['Speed'].quantile(0.3)]
        high_speed_data = driver_data[driver_data['Speed'] > driver_data['Speed'].quantile(0.7)]
        
        profile['cornering_style'] = low_speed_data['Throttle'].mean() if len(low_speed_data) > 0 else driver_data['Throttle'].mean()
        profile['straight_line_style'] = high_speed_data['Throttle'].mean() if len(high_speed_data) > 0 else driver_data['Throttle'].mean()
        
        # Replace NaN values with 0
        for key, value in profile.items():
            if pd.isna(value):
                profile[key] = 0.0
        
        driver_profiles[driver] = profile
    
    if len(driver_profiles) < 2:
        print("Warning: Not enough drivers for similarity analysis")
        return np.array([]), np.array([]), list(driver_profiles.keys()), driver_profiles
    
    # Convert to matrix for similarity analysis
    drivers = list(driver_profiles.keys())
    features = list(next(iter(driver_profiles.values())).keys())
    profile_matrix = np.array([[driver_profiles[driver][feature] for feature in features]
                              for driver in drivers])
    
    # Handle infinite values
    profile_matrix = np.nan_to_num(profile_matrix, nan=0.0, posinf=999999, neginf=-999999)
    
    # Standardize features
    scaler = StandardScaler()
    profile_matrix_scaled = scaler.fit_transform(profile_matrix)
    
    # Calculate similarity matrices
    similarity_matrix = cosine_similarity(profile_matrix_scaled)
    distance_matrix = euclidean_distances(profile_matrix_scaled)
    
    return similarity_matrix, distance_matrix, drivers, driver_profiles

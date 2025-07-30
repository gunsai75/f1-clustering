
# Team and driver mappings
TEAMS = {
    'Williams': ['ALB', 'SAI'],
    'Mercedes': ['ANT', 'RUS'],
    'Ferrari': ['HAM', 'LEC'],
    'RedBull': ['VER', 'TSU'],
    'McLaren': ['PIA', 'NOR']
}

# Enhanced team colors with better visibility
TEAM_COLORS = {
    'Williams': '#37BEDD',    # Bright cyan blue
    'Mercedes': '#00D2BE',    # Teal green
    'Ferrari': '#DC143C',     # Red
    'RedBull': '#1E41FF',     # Bright blue
    'McLaren': '#FF8000',     # Orange
    'Unknown': '#666666'      # Dark gray for unknown teams
}

# Track-specific DBSCAN parameters
TRACK_CLUSTERING_PARAMS = {
    'Australia': {'eps': 0.4, 'min_samples': 40},
    'Bahrain': {'eps': 0.35, 'min_samples': 50},
    'China': {'eps': 0.45, 'min_samples': 45},
    'Japan': {'eps': 0.4, 'min_samples': 35}
}

# Default clustering parameters
DEFAULT_CLUSTERING_PARAMS = {'eps': 0.4, 'min_samples': 45}

# Feature columns for analysis
FEATURE_COLUMNS = [
    'RPM', 'Speed', 'nGear', 'Throttle', 'nBrake',
    'ThrottleRate', 'BrakeIntensity', 'GearEfficiency',
    'SpeedVariability', 'Acceleration'
]

# Radar chart metrics
RADAR_METRICS = [
    'throttle_aggression', 'throttle_smoothness', 'brake_intensity',
    'speed_variability', 'acceleration_pattern'
]

# Data paths
DATA_PATHS = {
    'telemetry': 'telemetry-data',  # Fixed to match your system
    'qualifying': 'Qualifying',
    'race': 'Race'
}

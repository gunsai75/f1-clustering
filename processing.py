import pandas as pd
import os
from config import TEAMS, DATA_PATHS
import numpy as np

def get_driver_team(driver_code: str) -> str:
    """Get team name for a driver."""
    for team, drivers in TEAMS.items():
        if driver_code in drivers:
            return team
    print(f"Warning: Driver {driver_code} not found in team mappings")
    return 'Unknown'


def debug_directory_structure():
    """Debug function to show the actual directory structure."""
    print("=== DIRECTORY STRUCTURE DEBUG ===")
    base_path = DATA_PATHS['telemetry']
    
    if not os.path.exists(base_path):
        print(f"Base telemetry path '{base_path}' does not exist!")
        return
    
    print(f"Base path: {base_path}")
    
    quali_path = os.path.join(base_path, DATA_PATHS['qualifying'])
    if not os.path.exists(quali_path):
        print(f"Qualifying path '{quali_path}' does not exist!")
        return
    
    print(f"Qualifying path: {quali_path}")
    print("Available tracks:")
    
    for item in os.listdir(quali_path):
        track_path = os.path.join(quali_path, item)
        if os.path.isdir(track_path):
            print(f"  📁 {item}/")
            
            # List files in track directory
            files = []
            dirs = []
            for subitem in os.listdir(track_path):
                subpath = os.path.join(track_path, subitem)
                if os.path.isfile(subpath):
                    files.append(subitem)
                elif os.path.isdir(subpath):
                    dirs.append(subitem)
            
            if files:
                print(f"    Files: {files}")
            if dirs:
                print(f"    Subdirs: {dirs}")
                # Check subdirectories for files too
                for subdir in dirs:
                    subdir_path = os.path.join(track_path, subdir)
                    subfiles = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
                    if subfiles:
                        print(f"      {subdir}/: {subfiles}")
    print("=== END DEBUG ===\n")


def is_second_driver(driver_code: str, team: str) -> bool:
    """Check if driver is the second driver for their team."""
    team_drivers = TEAMS.get(team, [])
    return len(team_drivers) > 1 and driver_code == team_drivers[1]


def process_driver_telemetry(filepath: str, driver_code: str) -> pd.DataFrame:
    """Process individual driver telemetry file."""
    try:
        df = pd.read_csv(filepath)
        
        # Handle different CSV formats - check if Brake column exists
        if 'Brake' in df.columns:
            df['nBrake'] = df["Brake"].astype(int)
            df_clean = df.drop(labels=["Brake"], axis=1, errors='ignore')
        elif 'nBrake' not in df.columns:
            # If no brake data, create dummy column
            df['nBrake'] = 0
            df_clean = df.copy()
        else:
            df_clean = df.copy()
        
        # Drop optional columns that might not exist
        columns_to_drop = ["Date", "Time", "SessionTime", "Source"]
        for col in columns_to_drop:
            if col in df_clean.columns:
                df_clean = df_clean.drop(labels=[col], axis=1)
        
        # Ensure required columns exist
        required_cols = ['RPM', 'Speed', 'nGear', 'Throttle']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} for {driver_code}")
            return None
        
        # Create telemetry features
        features = pd.DataFrame()
        features['RPM'] = df_clean['RPM']
        features['Speed'] = df_clean['Speed']
        features['nGear'] = df_clean['nGear']
        features['Throttle'] = df_clean['Throttle']
        features['nBrake'] = df_clean['nBrake']
        
        # Advanced features for driving style with error handling
        features['ThrottleRate'] = df_clean['Throttle'].diff().abs().fillna(0)
        features['BrakeIntensity'] = df_clean['nBrake'] * df_clean['Speed']
        features['GearEfficiency'] = df_clean['RPM'] / (df_clean['nGear'] + 1)
        features['SpeedVariability'] = df_clean['Speed'].rolling(10, min_periods=1).std().fillna(0)
        features['Acceleration'] = df_clean['Speed'].diff().fillna(0)
        
        # Remove infinite values and NaN
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Add metadata
        features['Driver'] = driver_code
        features['Team'] = get_driver_team(driver_code)
        
        return features
    except Exception as e:
        print(f"Error processing {driver_code}: {e}")
        return None


def load_track_data(race_location: str) -> tuple:
    """Load telemetry data for all drivers from a specific track."""
    all_data = []
    driver_stats = {}
    
    try:
        # Construct path based on your directory structure: telemetry-data\Qualifying\Australia\
        track_path = os.path.join(DATA_PATHS['telemetry'], DATA_PATHS['qualifying'], race_location)
        
        print(f"Looking for data in: {track_path}")
        
        if not os.path.exists(track_path):
            print(f"Path {track_path} does not exist!")
            print("Available directories:")
            base_path = os.path.join(DATA_PATHS['telemetry'], DATA_PATHS['qualifying'])
            if os.path.exists(base_path):
                for item in os.listdir(base_path):
                    if os.path.isdir(os.path.join(base_path, item)):
                        print(f"  - {item}")
            return None, None
        
        # Get all CSV files in the track directory
        csv_files = []
        for item in os.listdir(track_path):
            item_path = os.path.join(track_path, item)
            if os.path.isfile(item_path) and item.endswith('.csv'):
                csv_files.append(item)
            elif os.path.isdir(item_path):
                # If there are subdirectories, look for CSV files in them too
                for subitem in os.listdir(item_path):
                    if subitem.endswith('.csv'):
                        csv_files.append(os.path.join(item, subitem))
        
        if not csv_files:
            print(f"No CSV files found in {track_path}")
            print("Available files:")
            for item in os.listdir(track_path):
                print(f"  - {item}")
            return None, None
        
        print(f"Loading {race_location} telemetry data...")
        print(f"Found {len(csv_files)} driver files: {csv_files}")
        
        for filename in csv_files:
            # Extract driver code from filename
            # Handle both direct files and files in subdirectories
            base_filename = os.path.basename(filename)
            
            # Try different patterns to extract driver code
            if base_filename.endswith('.csv'):
                driver_code = base_filename.replace('.csv', '')
                
                # If filename is just a 3-letter code, use it directly
                if len(driver_code) == 3 and driver_code.isupper():
                    pass  # driver_code is already correct
                else:
                    # Try to extract 3-letter code from longer filename
                    parts = driver_code.split('-')
                    potential_codes = [part for part in parts if len(part) == 3 and part.isupper()]
                    if potential_codes:
                        driver_code = potential_codes[0]
                    else:
                        # Fallback: use the whole filename without extension
                        driver_code = base_filename.replace('.csv', '')
            
            filepath = os.path.join(track_path, filename)
            print(f"Processing {driver_code} from {filepath}")
            
            driver_data = process_driver_telemetry(filepath, driver_code)
            
            if driver_data is not None and len(driver_data) > 0:
                all_data.append(driver_data)
                driver_stats[driver_code] = {
                    'samples': len(driver_data),
                    'team': get_driver_team(driver_code),
                    'avg_speed': driver_data['Speed'].mean(),
                    'max_speed': driver_data['Speed'].max(),
                    'avg_throttle': driver_data['Throttle'].mean(),
                    'brake_usage': driver_data['nBrake'].mean()
                }
                print(f"  ✓ Loaded {len(driver_data)} samples for {driver_code}")
            else:
                print(f"  ✗ Failed to process data for {driver_code}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully loaded {len(all_data)} drivers with {len(combined_data)} total samples")
            return combined_data, driver_stats
        else:
            print(f"No valid data loaded for {race_location}")
            return None, None
            
    except Exception as e:
        print(f"Error loading {race_location} data: {e}")
        import traceback
        traceback.print_exc()
        return None, None
from typing import Optional
import os
import numpy as np
import pandas as pd
from processing import load_track_data
from clustering import cluster_track_data, analyze_driving_style_similarity
from visualisation import create_driving_style_visualizations
import matplotlib.pyplot as plt


def analyze_track_driving_styles(clustered_data, track_name: str, output_dir: Optional[str] = None):
    """Complete driving style analysis for a track."""
    print(f"\n{'='*60}")
    print(f"DRIVING STYLE ANALYSIS - {track_name.upper()}")
    print(f"{'='*60}")
    
    # Perform similarity analysis
    similarity_matrix, distance_matrix, drivers, driver_profiles = analyze_driving_style_similarity(
        clustered_data, track_name
    )
    
    # Create visualization
    fig, pca_result = create_driving_style_visualizations(
        similarity_matrix, distance_matrix, drivers, driver_profiles, clustered_data, track_name
    )
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{track_name}_analysis.png"), dpi=300, bbox_inches='tight')
    
    # Print insights
    print_analysis_insights(similarity_matrix, drivers, driver_profiles)
    
    return fig, driver_profiles


def analyze_single_track(track_name: str, output_dir: Optional[str] = None, show_plots: bool = True):
    """Analyze a single track."""
    print(f"\n{'='*40}")
    print(f"ANALYZING {track_name.upper()}")
    print(f"{'='*40}")
    
    # Add debug information
    from processing import debug_directory_structure
    debug_directory_structure()
    
    # Load data
    track_data, driver_stats = load_track_data(track_name)
    
    if track_data is not None and len(track_data) > 100:
        print(f"Data loaded successfully: {len(track_data)} samples")
        
        # Apply clustering
        clustered_data = cluster_track_data(track_data, track_name)
        
        # Perform driving style analysis
        fig, driver_profiles = analyze_track_driving_styles(
            clustered_data, track_name, output_dir
        )
        
        if show_plots:
            plt.show()
            
        return fig, driver_profiles
    else:
        print(f"Insufficient data for {track_name} - skipping")
        return None, None


def analyze_all_tracks_driving_styles(output_dir: Optional[str] = None, show_plots: bool = True):
    """Enhanced analysis for all tracks."""
    available_tracks = ['Australia', 'Bahrain', 'China', 'Japan']
    
    results = {}
    for track in available_tracks:
        result = analyze_single_track(track, output_dir, show_plots)
        if result[0] is not None:
            results[track] = result
    
    print(f"\n{'='*60}")
    print("DRIVING STYLE ANALYSIS COMPLETE!")
    print("The PC1/PC2 plots show the spatial relationships you wanted.")
    print(f"{'='*60}")
    
    return results


def print_analysis_insights(similarity_matrix, drivers, driver_profiles):
    """Print analysis insights."""
    print(f"\nDRIVING STYLE INSIGHTS:")
    print(f"Drivers analyzed: {len(drivers)}")
    
    if similarity_matrix.size == 0 or len(drivers) < 2:
        print("Insufficient data for detailed insights")
        return
    
    # Find most similar pair
    max_similarity = 0
    most_similar_pair = None
    for i in range(len(drivers)):
        for j in range(i+1, len(drivers)):
            similarity = similarity_matrix[i][j]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (drivers[i], drivers[j])
    
    if most_similar_pair:
        print(f"Most similar driving styles: {most_similar_pair[0]} & {most_similar_pair[1]} ({max_similarity:.3f})")
    
    # Identify driving style archetypes
    if driver_profiles:
        try:
            aggressive_drivers = sorted(drivers, key=lambda d: driver_profiles[d].get('throttle_aggression', 0), reverse=True)[:2]
            smooth_drivers = sorted(drivers, key=lambda d: driver_profiles[d].get('throttle_smoothness', 0), reverse=True)[:2]
            print(f"Most aggressive drivers: {', '.join(aggressive_drivers)}")
            print(f"Smoothest drivers: {', '.join(smooth_drivers)}")
        except Exception as e:
            print(f"Error generating insights: {e}")


if __name__ == "__main__":
    # Run analysis for all tracks
    results = analyze_all_tracks_driving_styles(output_dir="analysis_results", show_plots=True)
    
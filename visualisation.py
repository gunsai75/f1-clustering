import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from config import TEAMS, TEAM_COLORS
from processing import get_driver_team, is_second_driver

def create_driving_style_visualizations(
    similarity_matrix,
    distance_matrix,
    drivers,
    driver_profiles,
    clustered_data,
    track_name
):
    """Create comprehensive driving style visualizations with error handling."""
    
    if len(drivers) < 2:
        print("Not enough drivers for visualization")
        return plt.figure(), np.array([])
    
    # Adjusted figure size and grid layout for equal space and less cramped appearance
    fig = plt.figure(figsize=(20, 16))  # Slightly larger figure for better spacing
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], 
                         hspace=0.3, wspace=0.3)  # Equal width for heatmap and PCA, adjusted spacing
    fig.suptitle(f'{track_name} - Driving Style Analysis', fontsize=18, fontweight='bold')

    # 1. Similarity Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    if similarity_matrix.size > 0:
        sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', center=0,
                    xticklabels=drivers, yticklabels=drivers, ax=ax1, fmt='.2f',
                    annot_kws={'size': 10})
        ax1.set_title('Driver Similarity Matrix\n(Higher values = more similar driving)', fontsize=12)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
    else:
        ax1.text(0.5, 0.5, 'Insufficient data for similarity analysis', 
                ha='center', va='center', transform=ax1.transAxes)

    # 2. Principal Component Analysis
    ax2 = fig.add_subplot(gs[0, 1])  # Now occupies one column for equal space
    pca_result = np.array([])
    
    if len(driver_profiles) >= 2:
        try:
            features = list(next(iter(driver_profiles.values())).keys())
            profile_matrix = np.array([[driver_profiles[driver][feature] for feature in features] 
                                  for driver in drivers])
            
            # Handle infinite and NaN values
            profile_matrix = np.nan_to_num(profile_matrix, nan=0.0, posinf=999999, neginf=-999999)
            
            # Standardize features
            scaler = StandardScaler()
            profile_matrix_scaled = scaler.fit_transform(profile_matrix)
            
            # Apply PCA
            pca = PCA(n_components=min(2, len(features), len(drivers)))
            pca_result = pca.fit_transform(profile_matrix_scaled)

            # Plot each driver
            for i, driver in enumerate(drivers):
                team = get_driver_team(driver)
                color = TEAM_COLORS.get(team, '#666666')
                ax2.scatter(pca_result[i, 0], pca_result[i, 1] if pca_result.shape[1] > 1 else 0,
                            c=color, s=500, alpha=0.8, edgecolors='black', linewidth=2, label=driver)
                ax2.annotate(driver, (pca_result[i, 0], pca_result[i, 1] if pca_result.shape[1] > 1 else 0),
                             xytext=(0, 0), textcoords='offset points',
                             fontweight='bold', fontsize=14, ha='center', va='center')

            # Add reference lines
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            # Labels and title
            variance_ratio = pca.explained_variance_ratio_
            ax2.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} variance)', 
                           fontweight='bold', fontsize=12)
            ax2.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} variance)' if len(variance_ratio) > 1 else 'PC2', 
                           fontweight='bold', fontsize=12)
            ax2.set_title('Driving Style Principal Components\n(Similar drivers cluster together)', 
                          fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3)

            # Add subtle legend on the axes
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                          markeredgecolor='black', markersize=8, 
                          label='Drivers (by team color)'),
                plt.Line2D([0], [0], color='none', 
                          label='PC1: Main driving style variation'),
                plt.Line2D([0], [0], color='none', 
                          label='PC2: Secondary style variation'),
                plt.Line2D([0], [0], color='none', 
                          label='Closer points = similar styles')
            ]
            ax2.legend(handles=legend_elements, loc='lower left', 
                      fontsize=8, framealpha=0.7, edgecolor='gray', 
                      title='Legend', title_fontsize=9)
        except Exception as e:
            ax2.text(0.5, 0.5, f'PCA Error: {str(e)}', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for PCA', ha='center', va='center', transform=ax2.transAxes)

    # 3. Radar Chart
    ax3 = fig.add_subplot(gs[1, :], projection='polar')  # Spans full width in second row
    radar_metrics = ['throttle_aggression', 'throttle_smoothness', 'brake_intensity',
                     'speed_variability', 'acceleration_pattern']

    if len(driver_profiles) > 0:
        try:
            # Prepare radar data
            radar_data = []
            for driver in drivers:
                driver_radar = []
                for metric in radar_metrics:
                    if metric in driver_profiles[driver]:
                        value = driver_profiles[driver][metric]
                        all_values = [driver_profiles[d][metric] for d in drivers 
                                     if metric in driver_profiles[d] and not pd.isna(driver_profiles[d][metric])]
                        if len(all_values) > 1:
                            min_val, max_val = min(all_values), max(all_values)
                            normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                        else:
                            normalized = 0.5
                    else:
                        normalized = 0.5
                    
                    # Ensure normalized value is valid
                    if pd.isna(normalized) or np.isinf(normalized):
                        normalized = 0.5
                    
                    driver_radar.append(normalized)
                radar_data.append(driver_radar)

            # Set up angles for radar chart
            angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            # Plot each driver
            for i, driver in enumerate(drivers):
                values = radar_data[i] + radar_data[i][:1]  # Complete the circle
                team = get_driver_team(driver)
                color = TEAM_COLORS.get(team, '#666666')
                line_style = '--' if is_second_driver(driver, team) else '-'
                ax3.plot(angles, values, line_style, linewidth=2, label=driver, 
                        color=color, alpha=0.8)
                ax3.fill(angles, values, alpha=0.15, color=color)

            # Customize radar chart
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels([metric.replace('_', ' ').title() for metric in radar_metrics])
            ax3.set_ylim(0, 1)
            ax3.set_title('Driving Style Profiles\n(Larger area = more extreme style)', 
                          y=1.08, fontweight='bold', fontsize=14)
            ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)  # Further adjusted legend position
        except Exception as e:
            ax3.text(0.5, 0.5, f'Radar Chart Error: {str(e)}', ha='center', va='center', transform=ax3.transAxes)
    
    plt.tight_layout()
    return fig, pca_result
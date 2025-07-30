# Formula 1 Driving Style Analysis - Qualifying Laps

This project analyzes Formula 1 driver telemetry data to identify and visualize driving style patterns using clustering and similarity analysis. It processes telemetry data from qualifying and race sessions, applies DBSCAN clustering, and generates visualizations to compare driver behaviors across different tracks.

## Directory Structure
- **data_collection.py**: Collects fastest lap telemetry data for specified drivers and races using the `fastf1` library and saves it as CSV files.
- **clustering.py**: Implements DBSCAN clustering and driving style similarity analysis using scikit-learn.
- **main.py**: Orchestrates the analysis pipeline, coordinating data loading, clustering, and visualization.
- **visualisation.py**: Generates visualizations including similarity heatmaps, PCA plots, and radar charts.
- **processing.py**: Handles data loading and preprocessing, including feature engineering for telemetry data.
- **config.py**: Contains configuration settings such as team mappings, clustering parameters, and data paths.
- **new-telemetry/**: Directory where telemetry data is stored (Qualifying and Race subdirectories).
- **analysis_results/**: Directory where visualization outputs are saved.

## Prerequisites
- Python 3.8+
- Required Python packages:
  ```bash
  pip install fastf1 pandas numpy scikit-learn matplotlib seaborn
  ```
- Ensure `texlive-full` and `texlive-fonts-extra` are installed if generating LaTeX-based documentation.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Note: Create a `requirements.txt` file with the above packages if not already present.

3. **Prepare Data**:
   - Ensure the `new-telemetry` directory exists or update `DATA_PATHS` in `config.py` to match your data storage.
   - Run `data_collection.py` to fetch telemetry data:
     ```bash
     python data_collection.py
     ```
   - Data will be saved in `new-telemetry/Qualifying` and `new-telemetry/Race` directories.

4. **Directory Setup**:
   - Create a `cache` directory for `fastf1` caching if not already present.
   - Ensure write permissions for `new-telemetry` and `analysis_results` directories.

## Usage
1. **Run Analysis**:
   - Execute the main script to analyze all tracks:
     ```bash
     python main.py
     ```
   - This will process data for Australia, Bahrain, China, and Japan tracks, generating visualizations in `analysis_results`.

2. **Analyze a Single Track**:
   - Modify `main.py` to call `analyze_single_track(track_name)` for a specific track, e.g.:
     ```python
     analyze_single_track("Australia", output_dir="analysis_results", show_plots=True)
     ```

3. **Output**:
   - Visualizations (heatmaps, PCA plots, radar charts) are saved as PNG files in `analysis_results`.
   - Console output includes clustering results and driving style insights (e.g., most similar drivers, aggressive/smooth drivers).

## Configuration
- **config.py**:
  - `TEAMS`: Maps drivers to their teams.
  - `TEAM_COLORS`: Defines colors for visualizations.
  - `TRACK_CLUSTERING_PARAMS`: DBSCAN parameters for each track.
  - `DEFAULT_CLUSTERING_PARAMS`: Fallback DBSCAN parameters.
  - `FEATURE_COLUMNS`: Telemetry features used for clustering.
  - `DATA_PATHS`: Directory paths for telemetry data.

## Notes
- **Data Availability**: Ensure telemetry data for 2025 races is available via `fastf1`. Update `year` in `data_collection.py` if analyzing different seasons.
- **Error Handling**: The code includes robust error handling for missing data or invalid files. Check console output for warnings.
- **Visualization**: PCA plots require at least two drivers for meaningful results. Radar charts normalize metrics for comparison.
- **File Paths**: Update `DATA_PATHS` in `config.py` if your directory structure differs.

## Troubleshooting
- **Data Not Found**: Verify `new-telemetry` contains CSV files in the format `<track>-quali-<driver>.csv` or `<track>-race-<driver>.csv`.
- **Path Issues**: Run `debug_directory_structure()` in `processing.py` to inspect the directory structure.
- **Missing Columns**: Ensure telemetry CSVs include required columns (`RPM`, `Speed`, `nGear`, `Throttle`).
- **Visualization Errors**: Check for sufficient data (at least 100 samples per track) and valid driver profiles.

## Future Improvements
- Add support for additional tracks and seasons.
- Implement more advanced clustering algorithms (e.g., HDBSCAN).
- Enhance visualizations with interactive plots using Plotly.
- Add statistical significance tests for driver similarities.

## License
This project is licensed under the MIT License.
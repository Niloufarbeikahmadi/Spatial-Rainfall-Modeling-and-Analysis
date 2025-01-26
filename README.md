# Spatial-Rainfall-Modeling-and-Analysis
# Spatial Rainfall Analysis

## Project Overview
This project focuses on spatial interpolation of rainfall data using advanced geostatistical methods. The analysis includes occurrence/Magnitude variogram modeling, kriging interpolation, and visualization of spatial rainfall patterns.

## Features
- Preprocessing and aggregating rainfall data.
- Variogram modeling with Spherical, Exponential, Gaussian, wave-effect and Matérn models.
- Kriging interpolation for creating the probability maps of rainfall occurrence and estimation of rainfall field.
- Classification of rainfall patterns based on frequency and magnitude.
- Visualization of variograms and spatial rainfall grids.

## Repository Structure
SpatialRainfallAnalysis/ ├── data_processing.py # Functions for data loading and preprocessing ├── variogram_models.py # Variogram model definitions and fitting ├── kriging_methods.py # Kriging interpolation methods ├── visualizations.py # Visualization and plotting functions ├── main.py # Main script for running the analysis └── README.md # Project description and instructions


## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SpatialRainfallAnalysis.git
   
2. Install the required Python libraries:  
   Run the following command in your terminal:  
   ```bash
   pip install -r requirements.txt

3. Run the main script:
Execute the main script using this command:
```bash
python main.py



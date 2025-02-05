# Rainfall Interpolation and Variogram Analysis

This repository contains a modularized Python project that demonstrates a two-phase approach for interpolating rainfall point records into high-resolution grid maps. The project is designed to work with sparse gauge networks and covers:

1. **Rainfall Occurrence Interpolation**  
   - Classifies daily rainfall occurrence into three classes based on the frequency of dry events (e.g., F0-25, F25-75, F75-100).
   - Computes experimental variograms for each class.
   - Selects the best-fit variogram model (using criteria such as AIC) and performs ordinary kriging to generate daily probability maps (with a resolution of 2 km).
   - Converts the continuous probability maps into binary maps using class-specific cutoff thresholds.

2. **Rainfall Magnitude Interpolation**  
   - Computes daily rainfall metrics (mean, maximum, standard deviation, and coefficient of variation).
   - Classifies daily rainfall using standard Mediterranean rainfall categories (e.g., Light, Light-Moderate, Heavy, etc.).
   - Groups days based on these classifications and calculates experimental variograms for each group.
   - Selects the best theoretical variogram model (based on AIC) and performs kriging to generate rainfall magnitude maps, using the occurrence binary maps as a mask.

## Features

- **Modular Design**:  
  The code is organized into several modules for clarity and maintainability:
  - `data.py`: Data loading, preprocessing, and statistical analysis.
  - `variogram.py`: Variogram models, model fitting, and experimental variogram computation.
  - `kriging.py`: Functions for performing kriging on both occurrence and magnitude data.
  - `visualization.py`: Functions for plotting variograms, daily probability maps, and overview figures.
  - `main.py`: Main script that ties together all the components.

- **Extensibility**:  
  The repository is structured to allow easy modification or extension. You can add new variogram models, change the interpolation method, or adjust the classification criteria with minimal changes to the overall code.

- **Visualization Tools**:  
  The project includes several visualization routines to help interpret the variogram models, the interpolated probability maps, and the final rainfall maps.

## Repository Structure

```plaintext
rainfall_interpolation/
├── data.py                 # Data loading and preprocessing routines.
├── variogram.py            # Variogram models and fitting routines.
├── kriging.py              # Kriging functions for occurrence and magnitude interpolation.
├── visualization.py        # Plotting and visualization routines.
├── main.py                 # Main script that orchestrates the workflow.
├── README.md               # This file.
└── requirements.txt        # Python package requirements.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rainfall_interpolation.git
cd rainfall_interpolation
```
   
2. Install Dependencies

It is recommended to use a virtual environment. Then install the required packages:  
   Run the following command in your terminal:  
   ```bash
   python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, install packages manually:
```bash
pip install numpy pandas matplotlib geopandas pykrige scipy tqdm openpyxl
```

3. Run the main script:
Execute the main script using this command:
```bash
python main.py
```
The script will process the data, generate the variograms, perform kriging for both occurrence and magnitude, and display several plots. Processed results (e.g., kriging outputs) are also saved as pickle files.


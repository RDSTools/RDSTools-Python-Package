# RDS Tools

A Python package for Respondent-Driven Sampling (RDS) analysis and bootstrap resampling with parallel processing capabilities.

## Table of Contents

1. [Installation](#installation)
2. [Example Dataset](#example-dataset)
3. [Data Processing](#data-processing)
4. [Estimation](#estimation)
   - [Means](#means)
   - [Tables](#tables)
   - [Regression](#regression)
5. [Sampling Variance](#sampling-variance)
6. [Visualization](#visualization)
   - [Recruitment Networks](#recruitment-networks)
   - [Geographic Mapping](#geographic-mapping)
7. [Performance Enhancement](#performance-enhancement)
8. [Requirements](#requirements)

## Installation
```bash
pip install RDSTools
```

For development (from source):
```bash
git clone https://github.com/RDSTools/RDSTools-Python-Package.git
cd RDSTools-Python-Package/RDSTools
pip install -e .
```

## Example Dataset

RDSTools includes a toy dataset for testing and learning. You can load it in three ways:

### Method 1: Using load_toy_data() (Recommended)

```python
from RDSTools import load_toy_data, RDSdata

# Load the example dataset
toy_data = load_toy_data()
print(f"Loaded {len(toy_data)} observations")

# Process it with RDSdata
rds_data = RDSdata(
    data=toy_data,
    unique_id="ID",
    redeemed_coupon="CouponR",
    issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
    degree="Degree"
)
```

### Method 2: Using the RDSToolsToyData variable

```python
from RDSTools import RDSToolsToyData, RDSdata

# The dataset is automatically loaded
rds_data = RDSdata(
    data=RDSToolsToyData,
    unique_id="ID",
    redeemed_coupon="CouponR",
    issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
    degree="Degree"
)
```

### Method 3: Getting the file path

```python
from RDSTools import get_toy_data_path
import pandas as pd

# Get the path and load manually
path = get_toy_data_path()
toy_data = pd.read_csv(path)
```

## Data Processing

The `RDSdata()` function processes respondent-driven sampling data by reconstructing recruitment chains, calculating wave numbers, identifying seeds, and imputing missing degree values. It tracks how participants recruited one another through coupon redemption. **Use RDSdata before applying any estimation or plotting functions from the RDSTools package.**

### Usage

```python
RDSdata(data, unique_id, redeemed_coupon, issued_coupons, degree, zero_degree="hotdeck", NA_degree="hotdeck")
```

### Arguments

- **data**: pandas.DataFrame. Should contain an ID variable for sample case, corresponding redeemed coupon code, and issued coupon code.

- **unique_id**: str. The column name of the column with respondent IDs.

- **redeemed_coupon**: str. The column name of the column with coupon codes redeemed by respondents when participating in the study.

- **issued_coupons**: list of str. The column name of the column with coupon codes issued to respondents (i.e., coupons given to respondents to recruit their peers). If multiple coupons are issued, list all coupon code variables.

- **degree**: str. The column name of the column with degree (i.e., network size) reported by respondents.

- **zero_degree**: str, optional. Used to set the method for handling zero values in the 'degree' variable. Three available methods are: mean imputation, median imputation, and hotdeck imputation. Default: 'hotdeck'.

- **NA_degree**: str, optional. Used to set the method for handling missing values in the 'degree' variable. Three available methods are: mean imputation, median imputation, and hotdeck imputation. Default: 'hotdeck'.

### Example

```python
from RDSTools import load_toy_data, RDSdata

# Using the built-in toy dataset
data = load_toy_data()

rds_data = RDSdata(
    data=data,
    unique_id="ID",
    redeemed_coupon="CouponR",
    issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
    degree="Degree"
)

# With custom imputation methods
rds_data = RDSdata(
    data=data,
    unique_id="ID",
    redeemed_coupon="CouponR",
    issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
    degree="Degree",
    zero_degree="median",
    NA_degree="mean"
)

print(f"Seeds: {rds_data['SEED'].sum()}")
print(f"Max wave: {rds_data['WAVE'].max()}")
```

## Estimation

### Means

Estimating mean with respondent driven sampling sample data. This function calculates weighted or unweighted means for either a continuous or a categorical variable. Standard errors are calculated using naive or resampling approaches from 'RDSboot'.

```python
from RDSTools import RDSmean

# Basic mean calculation
result = RDSmean(
    x='age',
    data=rds_data,
    weight='WEIGHT',
    var_est='chain1',
    resample_n=1000
)

# With optional returns
result, bootstrap_means = RDSmean(
    x='age',
    data=rds_data,
    var_est='chain1',
    resample_n=1000,
    return_bootstrap_means=True
)

# With both optional returns
result, bootstrap_means, node_counts = RDSmean(
    x='age',
    data=rds_data,
    var_est='chain1', 
    resample_n=1000,
    return_bootstrap_means=True,
    return_node_counts=True
)
```

### Tables

Estimating one and two-way tables with respondent driven sampling sample data. One-way tables are constructed by specifying a categorical variable for x argument only. Two-way tables are constructed by specifying two categorical variables for x and y arguments. Standard errors of proportions are calculated using naive or resampling approaches from 'RDSboot'.

```python
from RDSTools import RDStable

# One-way table
result = RDStable(
    x="Sex",
    data=rds_data,
    var_est='chain1',
    resample_n=1000
)

# Two-way table
result = RDStable(
    x="Sex",
    y="Race", 
    data=rds_data,
    var_est='chain1',
    resample_n=1000,
    margins=1  # row proportions
)

# With optional returns
result, bootstrap_tables = RDStable(
    x="Sex",
    y="Race",
    data=rds_data,
    var_est='chain1',
    resample_n=1000,
    return_bootstrap_tables=True
)
```

### Regression

Linear and Logistic Regression Modeling with Respondent Driven Sampling (RDS) Sample Data. This function mimics the lm function in R stats package with capabilities to handle RDS data in model estimation. Standard errors of regression coefficients are calculated using naive or resampling approaches from 'RDSboot'. The formula syntax follows R-style/patsy conventions.

```python
from RDSTools import RDSlm

# Linear regression (continuous dependent variable)
result = RDSlm(
    data=rds_data,
    formula="Age ~ Sex + Race",
    weight='WEIGHT',
    var_est='chain1',
    resample_n=1000
)

# Use C() to explicitly mark categorical variables
# This is especially important for numeric codes (e.g., 0/1, 1/2/3)
result = RDSlm(
    data=rds_data,
    formula="Income ~ Age + C(Sex) + C(Race)",
    var_est='chain1',
    resample_n=1000
)

# Logistic regression (binary dependent variable)
result = RDSlm(
    data=rds_data,
    formula="Employed ~ Age + Education",
    var_est='chain1',
    resample_n=1000
)

# With optional returns
result, bootstrap_estimates = RDSlm(
    data=rds_data,
    formula="Age ~ Sex + Race",
    var_est='chain1',
    resample_n=1000,
    return_bootstrap_estimates=True
)
```

**Note on Categorical Variables:** Use `C()` around variable names to treat them as categorical. This is important when:
- Variables are numeric codes (e.g., Sex coded as 0/1)
- You want to ensure proper dummy variable creation
- Variables might be interpreted as continuous otherwise

## Sampling Variance

Bootstrap Resampling for Respondent Driven Sampling (RDS). Although resampling is incorporated within the estimation functions, users who wish to perform resampling separately can use `RDSboot()` or `RDSBootOptimizedParallel()`. After preprocessing with RDSdata, ensure the presence of at least four variables: `ID`, `S_ID`, `SEED`, and `R_ID`. Note that the sampling of respondents (seeds and recruits) is conducted with replacement, and the resulting data frame will contain duplicates.

```python
from RDSTools import RDSboot

# Bootstrap resampling
boot_results = RDSboot(
    data=rds_data,
    respondent_id_col='ID',
    seed_id_col='S_ID', 
    seed_col='SEED',
    recruiter_id_col='R_ID',
    type='tree_uni1',
    resample_n=1000
)

# Parallel bootstrap for better performance
from RDSTools import RDSBootOptimizedParallel

boot_results = RDSBootOptimizedParallel(
    data=rds_data,
    respondent_id_col='ID',
    seed_id_col='S_ID',
    seed_col='SEED',
    recruiter_id_col='R_ID',
    type='tree_uni1',
    resample_n=1000,
    n_cores=4
)
```

### Bootstrap Methods

In all bootstrap methods, versions 1 and 2 differ as version 1 sets the number of seeds in a given resample to be consistent with the number of seeds in the original sample (s), while version 2 sets the sample size of a given resample (n_r) to be at least equal to or greater than the original sample (n_s).

#### Bootstrap Chain

- **chain1**: Selects (s) seeds using SRSWR from all seeds in the original sample and then all nodes in the chains created by each of the resampled seeds are retained.

- **chain2**: 1 seed is sampled using SRSWR from all seeds in the original sample, and all nodes from the chain created by this seed are retained. It then compares (n_r) against (n_s), and, if (n_r < n_s), continues the resampling process by drawing 1 seed and its chains one by one until (n_r ≥ n_s).

#### Resample Tree Unidirectional

- **tree_uni1**: (s) seeds are selected using Simple Random Sampling with Replacement (SRSWR) from all seeds. For each selected seed, this method (A) checks its recruit counts, (B) selects SRSWR of the recruits counts from all recruits identified in (A), and (C) for each sampled recruit, this method repeats Steps A and B. (D) Steps A, B, and C continue until reaching the last wave of each chain.

- **tree_uni2**: Instead of selecting (s) seeds, it selects one seed, performs Steps B and C for the selected seed. It compares the size of the resample (n_r) and the original sample (n_s), and, if (n_r < n_s), it continues the resampling process by drawing 1 seed, performs Steps B and C and checks (n_r) against (n_s). If (n_r < n_s), the process continues until the sample size of a given resample (n_r) is at least equal to the original sample size (n_s), i.e., (n_r ≥ n_s).

#### Bootstrap Tree Bidirectional

- **tree_bi1**: Selects (s) nodes from the recruitment chains using SRSWR. For each selected node, it (A) checks its connected nodes (i.e., both recruiters and recruits) and their count, (B) from all connected nodes identified in (A), performs SRSWR of the same node count, and (C) for each selected node, performs steps A and B, but does not resample already resampled nodes. (D) Steps A, B, and C are repeated until the end of the chain.

- **tree_bi2**: Instead of (s) nodes, it selects 1 node using SRSWR from anywhere in all recruitment chains and repeats steps (B),(C), and (D) until (n_r ≥ n_s).

## Visualization

### Recruitment Networks

Visualize recruitment relationships through network graphs with various layout options and customizable styling.

```python
from RDSTools import RDSnetgraph, get_available_seeds, get_available_waves

# Get available seeds and waves
seeds = get_available_seeds(rds_data)
waves = get_available_waves(rds_data)

# Basic network graph
G = RDSnetgraph(
    data=rds_data,
    seed_ids=seeds[:2],
    waves=waves[:4],
    layout='Spring'
)

# Color nodes by demographic variable
G = RDSnetgraph(
    data=rds_data,
    seed_ids=seeds[:2],
    waves=waves[:3],
    layout='Spring',
    variable='Sex',
    title='Recruitment Network by Sex',
    save_path='network.png'
)

# Different layout options
G = RDSnetgraph(
    data=rds_data,
    seed_ids=['1'],
    waves=[0, 1, 2, 3, 4],
    layout='Tree',  # Options: 'Spring', 'Tree', 'Circular', 'Kamada-Kawai'
    figsize=(12, 10)
)
```


### Geographic Mapping

When longitude and latitude data are available, users can create interactive maps showing participant distributions and recruitment patterns across geographic areas.

```python
from RDSTools import RDSmap, get_available_seeds, get_available_waves, print_map_info

# Check available data for mapping
print_map_info(rds_data, lat_column='Latitude', lon_column='Longitude')

# Get available seeds and waves
seeds = get_available_seeds(rds_data)
waves = get_available_waves(rds_data)
print(f"Available seeds: {seeds}")
print(f"Available waves: {waves}")

# Simplest map - uses all available waves by default
m = RDSmap(
    data=rds_data,
    lat='Latitude',
    long='Longitude',
    seed_ids=['1', '2'],
    output_file='my_rds_map.html'
)

# Basic map with specific waves
m = RDSmap(
    data=rds_data,
    lat='Latitude',
    long='Longitude',
    seed_ids=['1', '2'],
    waves=[0, 1, 2, 3],
    output_file='my_rds_map.html'
)

# Map with custom styling
m = RDSmap(
    data=rds_data,
    lat='Latitude',
    long='Longitude',
    seed_ids=['1', '2', '3'],
    waves=[0, 1, 2, 3, 4],
    seed_color='red',
    seed_radius=7,
    recruit_color='blue',
    recruit_radius=7,
    line_color='black',
    line_weight=2,
    zoom_start=5,
    output_file='geographic_map.html',
    open_browser=True
)

# Using helper functions for seed and wave selection
m = RDSmap(
    data=rds_data,
    lat='Latitude',
    long='Longitude',
    seed_ids=seeds[:3],
    waves=waves[:4],
    line_dashArray='5,6',  # Dashed lines
    output_file='custom_map.html'
)
```

**Key Parameters:**
- `lat` - Column name for latitude coordinates
- `long` - Column name for longitude coordinates
- `seed_ids` - List of seed IDs to display
- `waves` - List of wave numbers to display (optional, defaults to all available waves)
- `seed_color` - Color of seed markers (default: "red")
- `seed_radius` - Size of seed markers (default: 7)
- `recruit_color` - Color of recruit markers (default: "blue")
- `recruit_radius` - Size of recruit markers (default: 7)
- `line_color` - Color of recruitment lines (default: "black")
- `line_weight` - Thickness of recruitment lines (default: 2)
- `line_dashArray` - Optional dash pattern for lines (e.g., '5,6')
- `zoom_start` - Initial map zoom level (default: 5)
- `output_file` - Name of HTML file to save (default: 'participant_map.html')
- `open_browser` - Whether to open map in browser automatically (default: False)

## Performance Enhancement

The package includes parallel processing for bootstrap methods. Unidirectional and bidirectional bootstrap sampling methods benefit the most from parallel processing.

```python
# Use parallel processing for faster bootstrap
result = RDSmean(
    x='income',
    data=rds_data,
    var_est='tree_uni1',
    resample_n=2000,
    n_cores=8  # Use 8 cores for parallel processing
)
```

### Performance Comparison

With 252 observations:

| Cores | Bootstrap Samples | Standard Time | Parallel Time | Speedup |
|-------|-------------------|---------------|---------------|---------|
| 1     | 1000             | 120s          | 120s          | 1.0x    |
| 4     | 1000             | 120s          | 18s           | 6.7x    |
| 8     | 1000             | 120s          | 12s           | 10.0x   |

## Complete Example Workflow

```python
from RDSTools import (
    load_toy_data, RDSdata, RDSboot, RDSmean, RDStable, RDSlm,
    RDSmap, RDSnetgraph, get_available_seeds, get_available_waves, print_map_info
)

# 1. Load and process data
# Option A: Use the included toy dataset
toy_data = load_toy_data()
rds_data = RDSdata(
    data=toy_data,
    unique_id="ID",
    redeemed_coupon="CouponR",
    issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
    degree="Degree"
)

# Option B: Load your own data
# import pandas as pd
# data = pd.read_csv("survey_data.csv")
# rds_data = RDSdata(
#     data=data,
#     unique_id="ID",
#     redeemed_coupon="CouponR",
#     issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
#     degree="Degree"
# )

# 2. Calculate weighted means
age_mean = RDSmean(
    x='Age',
    data=rds_data,
    weight='WEIGHT',
    var_est='tree_uni1',
    resample_n=1000,
    n_cores=4
)
print(age_mean)

# 3. Create frequency tables
sex_table = RDStable(
    x='Sex',
    data=rds_data,
    weight='WEIGHT',
    var_est='tree_uni1',
    resample_n=1000
)
print(sex_table)

# 4. Run regression analysis
model = RDSlm(
    data=rds_data,
    formula='Income ~ Age + C(Sex) + C(Race)',
    weight='WEIGHT',
    var_est='tree_uni1',
    resample_n=1000,
    n_cores=4
)
print(model)

# 5. Visualize recruitment network
seeds = get_available_seeds(rds_data)
waves = get_available_waves(rds_data)

G = RDSnetgraph(
    data=rds_data,
    seed_ids=seeds[:2],
    waves=waves[:4],
    layout='Spring',
    variable='Sex',
    title='Recruitment Network by Sex',
    save_path='network.png'
)

# 6. Create geographic map (uses all waves by default)
print_map_info(rds_data, lat_column='Latitude', lon_column='Longitude')

m = RDSmap(
    data=rds_data,
    lat='Latitude',
    long='Longitude',
    seed_ids=seeds[:2],  # Uses all available waves automatically
    output_file='recruitment_map.html',
    open_browser=True
)
```

## Requirements

- Python ≥ 3.7
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0
- statsmodels ≥ 0.12.0
- matplotlib ≥ 3.3.0
- networkx ≥ 2.5
- igraph ≥ 0.9.0 (python-igraph)
- folium ≥ 0.12.0 (for geographic mapping)
- scipy ≥ 1.7.0
- patsy ≥ 0.5.0

**Optional:**
- pygraphviz (for Tree layout in network graphs)

## API Reference

### Core Functions

- **`RDSdata()`** - Process RDS survey data
- **`RDSboot()`** - Bootstrap resampling for variance estimation
- **`RDSmean()`** - Calculate means with RDS adjustments
- **`RDStable()`** - Generate frequency tables
- **`RDSlm()`** - Linear and logistic regression models

### Visualization Functions

- **`RDSnetgraph()`** - Create recruitment network visualizations
- **`RDSmap()`** - Generate interactive geographic maps
- **`get_available_seeds()`** - Get list of seed IDs in data
- **`get_available_waves()`** - Get list of wave numbers in data
- **`print_map_info()`** - Display mapping information summary

### Data Utilities

- **`load_toy_data()`** - Load the included example dataset
- **`get_toy_data_path()`** - Get the file path to the example dataset
- **`RDSToolsToyData`** - Pre-loaded example dataset variable

### Advanced Functions

- **`RDSBootOptimizedParallel()`** - Parallelized bootstrap (used internally)

### Bootstrap Methods

Available variance estimation methods for `var_est` parameter:

- `chain1` - Bootstrap chain maintaining seed count
- `chain2` - Bootstrap chain with varying seed count
- `tree_uni1` - Unidirectional tree resampling maintaining seed count
- `tree_uni2` - Unidirectional tree resampling with varying seed count
- `tree_bi1` - Bidirectional tree resampling from n starting nodes
- `tree_bi2` - Bidirectional tree resampling with sample size matching

## Documentation

For comprehensive documentation and examples:
- [Full Documentation](https://rdstools-python-package.readthedocs.io/en/latest/)
- [Examples](https://rdstools-python-package.readthedocs.io/en/latest/examples.html)

## Citation

If you use RDS Tools in your research, please cite:

```
[Your citation here]
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues and Support

If you encounter any problems or have suggestions for improvements, please open an issue on GitHub.

## Changelog

### Version 0.1.0
- Initial release with core RDS analysis functions
- Bootstrap variance estimation with 6 resampling methods
- Parallel processing support
- Network visualization capabilities with customizable aesthetics
- Geographic mapping features with interactive controls
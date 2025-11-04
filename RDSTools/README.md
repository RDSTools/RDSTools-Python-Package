# RDS Tools

A Python package for Respondent-Driven Sampling (RDS) analysis and bootstrap resampling with parallel processing capabilities.

## Table of Contents

1. [Installation](#installation)
2. [Data Processing](#data-processing)
3. [Estimation](#estimation)
   - [Means](#means)
   - [Tables](#tables)
   - [Regression](#regression)
4. [Sampling Variance](#sampling-variance)
5. [Visualization](#visualization)
   - [Recruitment Networks](#recruitment-networks)
   - [Geographic Mapping](#geographic-mapping)
6. [Performance Enhancement](#performance-enhancement)
7. [Requirements](#requirements)

## Installation

```bash
cd RDSTools
pip install .
```

For development:
```bash
pip install -e .
```

## Data Processing

The `RDS_data()` function processes data collected through Respondent-Driven Sampling (RDS). This function extracts the unique ID, redeemed coupon numbers, and issued coupon numbers from the original dataset. By processing this information, users can obtain the key data typically required for RDS-related research.

### Usage

```python
RDS_data(data, unique_id, redeemed_coupon, issued_coupons, degree)
```

### Arguments

- **data**: A pandas DataFrame containing ID numbers for nodes in the social network and corresponding redeemed/issued coupon numbers.

- **unique_id**: The column name representing ID numbers for nodes in the social network.

- **redeemed_coupon**: The column name representing coupon numbers redeemed by respondents when participating in the survey.

- **issued_coupons**: List of column names representing coupon numbers issued to respondents.

- **degree**: The column name representing the degree (network size) of respondents.

- **zero_degree**: Method for imputing zero values in degree variable ('mean', 'median', 'hotdeck', 'drop'). Default: 'hotdeck'.

- **NA_degree**: Method for imputing missing values in degree variable ('mean', 'median', 'hotdeck', 'drop'). Default: 'hotdeck'.

### Example

```python
import pandas as pd
from RDSTools import RDS_data

# Load your data
data = pd.read_csv("survey_data.csv")

# Process RDS structure
rds_data = RDS_data(
    data=data,
    unique_id="ID",
    redeemed_coupon="CouponR",
    issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
    degree="Degree",
    zero_degree="median",
    NA_degree="hotdeck"
)

print(f"Seeds: {rds_data['SEED'].sum()}")
print(f"Max wave: {rds_data['WAVE'].max()}")
```

## Estimation

### Means

Calculate means and standard errors for RDS data with optional weighting and different variance estimation methods.

```python
from RDSTools import RDSMean

# Basic mean calculation
result = RDSMean(
    x='age',
    data=rds_data,
    weight='WEIGHT',
    var_est='resample_tree_uni1',
    resample_n=1000
)

# With optional returns
result, bootstrap_means = RDSMean(
    x='age',
    data=rds_data,
    var_est='resample_tree_uni1',
    resample_n=1000,
    return_bootstrap_means=True
)

# With both optional returns
result, bootstrap_means, node_counts = RDSMean(
    x='age',
    data=rds_data,
    var_est='resample_tree_uni1', 
    resample_n=1000,
    return_bootstrap_means=True,
    return_node_counts=True
)
```

### Tables

Generate frequency tables and proportions for categorical variables with RDS-adjusted standard errors.

```python
from RDSTools import RDSTable

# One-way table
result = RDSTable(
    formula="~Sex",
    data=rds_data,
    var_est='resample_tree_uni1',
    resample_n=1000
)

# Two-way table
result = RDSTable(
    formula="~Sex+Race", 
    data=rds_data,
    var_est='resample_tree_uni1',
    resample_n=1000,
    margins=1  # row totals
)

# With optional returns
result, bootstrap_tables = RDSTable(
    formula="~Sex+Race",
    data=rds_data,
    var_est='resample_tree_uni1',
    resample_n=1000,
    return_bootstrap_tables=True
)
```

### Regression

Fit linear and logistic regression models with RDS-adjusted standard errors.

```python
from RDSTools import RDSRegression

# Linear regression (continuous dependent variable)
result = RDSRegression(
    data=rds_data,
    formula="Age ~ Sex + Race",
    weight='WEIGHT',
    var_est='resample_tree_uni1',
    resample_n=1000
)

# Logistic regression (binary dependent variable)
result = RDSRegression(
    data=rds_data,
    formula="Employed ~ Age + Education",
    var_est='resample_tree_uni1',
    resample_n=1000
)

# With optional returns
result, bootstrap_estimates = RDSRegression(
    data=rds_data,
    formula="Age ~ Sex + Race",
    var_est='resample_tree_uni1',
    resample_n=1000,
    return_bootstrap_estimates=True
)
```

## Sampling Variance

Although resampling is incorporated within the estimation functions, users who wish to perform resampling separately can use `RDSBoot()`. After preprocessing, ensure the presence of at least four variables: `ID`, `S_ID`, `SEED`, and `R_ID`. Note that the sampling of respondents (seeds and recruits) is conducted with replacement, and the resulting data frame will contain duplicates.

```python
from RDSTools import RDSBoot

# Bootstrap resampling
boot_results = RDSBoot(
    data=rds_data,
    respondent_id_col='ID',
    seed_id_col='S_ID', 
    seed_col='SEED',
    recruiter_id_col='R_ID',
    type='resample_tree_uni1',
    resample_n=1000
)

```

### Bootstrap Chain

In bootstrap chain functions, the first step is to select seeds with replacement with the subsequent selection of seeds' full recruitment chains.

- **resample_chain1**: The number of selected seeds equals the number of seeds in the data frame. Since the seeds are selected with replacement, the resulting data frame will contain exactly the same number of seeds, but a different number of recruits.

- **resample_chain2**: Selects only 1 seed at each iteration. The resulting number of seeds will vary, but the number of recruits will be equal or larger to the original number of recruits.

### Resample Tree Unidirectional

In the resample tree, the function performs SRSWR from the seeds and their recruitment chains. As before, seeds are selected with replacement. For each selected seed, the function identifies its recruits and then samples with replacement from these recruits. For each sampled recruit, this process repeats until the end of each individual recruitment chain.

- **resample_tree_uni1**: Since all seeds are selected with replacement, the resulting number of seeds will equal the number of seeds from the original data, but the number of recruits will vary.

- **resample_tree_uni2**: Samples only 1 seed at a time and then performs sampling with replacement from each wave of seed's recruits. The resulting data frame will have at least the original number of observations, but a varying number of seeds.

### Bootstrap Tree Bidirectional

Unlike the unidirectional case, bidirectional resampling starts from a random position in a chain, checks its connected nodes, and then samples with replacement from these nodes. For each sampled node, the process repeats, but does not go backwards; that is, already visited nodes are excluded from subsequent sampling.

- **resample_tree_bi1**: The function starts from n nodes, depending on the number of seeds.

- **resample_tree_bi2**: The function samples one node at a time and then evaluates whether the resulting sample is at least equal to the size of the original data. If not, the function continues resampling until the desired number of respondents is achieved.

### Example: Bootstrap Chain

```python
# Chain bootstrap 1 - maintains number of seeds
res_chain1 = RDSBoot(
    data=rds_data,
    respondent_id_col='ID',
    seed_id_col='S_ID',
    seed_col='SEED', 
    recruiter_id_col='R_ID',
    type='resample_chain1',
    resample_n=1
)

# Check results - merge with original data
sample_1 = res_chain1[res_chain1['RESAMPLE.N'] == 1]
merged = pd.merge(sample_1, rds_data, left_on='RESPONDENT_ID', right_on='ID')
print(f"Original seeds: {rds_data['SEED'].sum()}")
print(f"Bootstrap seeds: {merged['SEED'].sum()}")
```

## Visualization

The package supports visualization of respondents' networks and the geographic distribution of recruitment waves starting from seeds. Users can generate network plots to examine recruitment chains overall and by demographic characteristics, as well as geographic maps that display participant locations and the spread of recruitment over time or across regions.

### Recruitment Networks

The `create_network_graph()` function creates network visualizations showing recruitment relationships with support for different layouts and node coloring by demographic variables.

```python
from RDSTools.network_graph import create_network_graph

# Basic network graph
G = create_network_graph(
    data=rds_data,
    seed_ids=['1', '2'],
    waves=[0, 1, 2, 3],
    layout='Spring'
)

# Tree layout showing hierarchical structure
G = create_network_graph(
    data=rds_data,
    seed_ids=['1'],
    waves=[0, 1, 2, 3, 4],
    layout='Tree',
    save_path='recruitment_tree.png'
)

# Color nodes by demographic variable
G = create_network_graph(
    data=rds_data,
    seed_ids=['1', '2', '3'],
    waves=[0, 1, 2],
    layout='Kamada-Kawai',
    group_by='Gender',
    node_size=20,
    figsize=(16, 14)
)
```

### Geographic Mapping

When longitude and latitude data are available, users can create interactive maps showing participant distributions and recruitment patterns across geographic areas.

```python
from RDSTools.rds_map import create_participant_map, print_map_info

# Check available data for mapping
print_map_info(rds_data, lat_column='Latitude', lon_column='Longitude')

# Basic map
m = create_participant_map(
    data=rds_data,
    seed_ids=['1', '2'],
    waves=[0, 1, 2, 3],
    output_file='my_rds_map.html'
)

# Map with custom coordinates and auto-open browser
m = create_participant_map(
    data=rds_data,
    seed_ids=['1', '2', '3'],
    waves=[0, 1, 2, 3, 4],
    lat_column='lat',
    lon_column='long',
    output_file='geographic_map.html',
    zoom_start=7,
    open_browser=True
)
```

## Performance Enhancement

The package includes parallel processing for bootstrap methods. Unidirectional and bidirectional bootstrap sampling methods are the methods that benefit the most from parallel processing.

```python
# Use parallel processing for faster bootstrap
result = RDSMean(
    x='income',
    data=rds_data,
    var_est='resample_tree_uni1',
    resample_n=2000,
    n_cores=8  # Use 8 cores for parallel processing
)
```

### Performance Comparison

With 252 observations

| Cores | Bootstrap Samples | Standard Time | Parallel Time | Speedup |
|-------|-------------------|---------------|---------------|---------|
| 1     | 1000             | 120s          | 120s          | 1.0x    |
| 4     | 1000             | 120s          | 18s           | 6.7x    |
| 8     | 1000             | 120s          | 12s           | 10.0x   |

## Requirements

- Python ≥ 3.7
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0
- statsmodels ≥ 0.12.0

## Documentation

For comprehensive documentation, examples, and API reference:
- [Full Documentation](https://your-docs-url.readthedocs.io/)
- [API Reference](https://your-docs-url.readthedocs.io/en/latest/api.html)
- [Examples](https://your-docs-url.readthedocs.io/en/latest/examples.html)

## License

MIT License - see LICENSE file for details.

## Contributing

This package is currently in development. Please report issues or suggestions for improving performance and functionality.
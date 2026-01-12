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

The `RDSdata()` function processes data collected through Respondent-Driven Sampling (RDS). This function extracts the unique ID, redeemed coupon numbers, and issued coupon numbers from the original dataset. By processing this information, users can obtain the key data typically required for RDS-related research.

### Usage

```python
RDSdata(data, unique_id, redeemed_coupon, issued_coupons, degree)
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
from RDSTools import RDSdata

# Load your data
data = pd.read_csv("survey_data.csv")

# Process RDS structure
rds_data = RDSdata(
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

Generate frequency tables and proportions for categorical variables with RDS-adjusted standard errors.

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

Fit linear and logistic regression models with RDS-adjusted standard errors. The formula syntax follows R-style/patsy conventions.

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

Although resampling is incorporated within the estimation functions, users who wish to perform resampling separately can use `RDSboot()`. After preprocessing, ensure the presence of at least four variables: `ID`, `S_ID`, `SEED`, and `R_ID`. Note that the sampling of respondents (seeds and recruits) is conducted with replacement, and the resulting data frame will contain duplicates.

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
```

### Bootstrap Methods

All bootstrap methods select seeds with replacement and then sample from recruitment chains. The six available methods are:

#### Bootstrap Chain

In bootstrap chain functions, the first step is to select seeds with replacement with the subsequent selection of seeds' full recruitment chains.

- **chain1**: The number of selected seeds equals the number of seeds in the data frame. Since the seeds are selected with replacement, the resulting data frame will contain exactly the same number of seeds, but a different number of recruits.

- **chain2**: Selects only 1 seed at each iteration. The resulting number of seeds will vary, but the number of recruits will be equal or larger to the original number of recruits.

#### Resample Tree Unidirectional

In the resample tree, the function performs SRSWR from the seeds and their recruitment chains. As before, seeds are selected with replacement. For each selected seed, the function identifies its recruits and then samples with replacement from these recruits. For each sampled recruit, this process repeats until the end of each individual recruitment chain.

- **tree_uni1**: Since all seeds are selected with replacement, the resulting number of seeds will equal the number of seeds from the original data, but the number of recruits will vary.

- **tree_uni2**: Samples only 1 seed at a time and then performs sampling with replacement from each wave of seed's recruits. The resulting data frame will have at least the original number of observations, but a varying number of seeds.

#### Bootstrap Tree Bidirectional

Unlike the unidirectional case, bidirectional resampling starts from a random position in a chain, checks its connected nodes, and then samples with replacement from these nodes. For each sampled node, the process repeats, but does not go backwards; that is, already visited nodes are excluded from subsequent sampling.

- **tree_bi1**: The function starts from n nodes, depending on the number of seeds.

- **tree_bi2**: The function samples one node at a time and then evaluates whether the resulting sample is at least equal to the size of the original data. If not, the function continues resampling until the desired number of respondents is achieved.

### Example: Bootstrap Chain

```python
# Chain bootstrap 1 - maintains number of seeds
res_chain1 = RDSboot(
    data=rds_data,
    respondent_id_col='ID',
    seed_id_col='S_ID',
    seed_col='SEED', 
    recruiter_id_col='R_ID',
    type='chain1',
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

The `RDSnetgraph()` function creates network visualizations showing recruitment relationships with support for different layouts and node coloring by demographic variables.

```python
from RDSTools import RDSnetgraph, get_available_seeds, get_available_waves

# Get available seeds and waves
available_seeds = get_available_seeds(rds_data)
available_waves = get_available_waves(rds_data)

# Basic network graph
G = RDSnetgraph(
    data=rds_data,
    seed_ids=['1', '2'],
    waves=[0, 1, 2, 3],
    layout='Spring'
)

# Tree layout showing hierarchical structure
G = RDSnetgraph(
    data=rds_data,
    seed_ids=['1'],
    waves=[0, 1, 2, 3, 4],
    layout='Tree',
    save_path='recruitment_tree.png'
)

# Color nodes by demographic variable
G = RDSnetgraph(
    data=rds_data,
    seed_ids=['1', '2', '3'],
    waves=[0, 1, 2],
    layout='Kamada-Kawai',
    variable='Sex',
    title='Recruitment by Sex',
    vertex_size_seed=10,
    vertex_size=6,
    figsize=(16, 14)
)

# Customize seed and non-seed colors (when not grouping by variable)
G = RDSnetgraph(
    data=rds_data,
    seed_ids=available_seeds[:2],
    waves=list(range(0, 4)),
    seed_color='purple',
    nonseed_color='orange',
    edge_width=2.0
)
```

**Available Layouts:**
- `Spring` - Force-directed layout (default, uses igraph Fruchterman-Reingold)
- `Circular` - Nodes arranged in a circle
- `Kamada-Kawai` - Force-directed with optimal distances
- `Grid` - Regular grid arrangement
- `Star` - Star-shaped layout
- `Random` - Random positioning
- `Tree` - Hierarchical tree layout (uses NetworkX with pygraphviz)

**Key Parameters:**
- `seed_ids` - List of seed IDs to include in network
- `waves` - List of wave numbers to include
- `variable` - Optional demographic variable for node coloring (overrides seed_color/nonseed_color)
- `category_colors` - Optional list of custom colors for each category (must match number of categories in sorted order)
- `title` - Optional plot title
- `vertex_size_seed` - Size of seed vertices (default: 45)
- `vertex_size` - Size of non-seed vertices (default: 30)
- `seed_color` - Color for seed nodes when not grouping (default: "#E41A1C" red)
- `nonseed_color` - Color for non-seed nodes when not grouping (default: "#377EB8" blue)
- `edge_width` - Thickness of edges (default: 1.5)
- `layout` - Graph layout algorithm (default: "Spring")

**Color Customization:**
When using `variable` to color nodes by demographic categories:
- **Default palette**: A 20-color palette is used automatically (colors 1-8 from Set1, 9-16 from Dark2, 17-20 from Pastel1)
- **Custom colors**: Provide `category_colors` parameter with colors matching the number of categories (in sorted alphabetical/numerical order)
- **Many categories**: Variables with 10+ categories show a warning; 20+ categories will recycle colors

Example with custom colors:
```python
# Assuming 'Race' has 3 categories: ['1', '2', '3'] (sorted)
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
G = RDSnetgraph(
    data=rds_data,
    seed_ids=['1', '2'],
    waves=[0, 1, 2],
    variable='Race',
    category_colors=custom_colors
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
Examples
========

Complete Workflow Example
-------------------------

Here's a complete example showing how to analyze RDS data from start to finish::

    import pandas as pd
    from RDSTools import (
        RDSdata, RDSmean, RDStable, RDSlm,
        RDSnetgraph, RDSmap, get_available_seeds, print_map_info
    )

    # 1. Load and examine your data
    data = pd.read_csv("rds_survey.csv")
    print(data.columns)
    print(f"Total participants: {len(data)}")

    # 2. Process the RDS structure
    rds_data = RDSdata(
        data=data,
        unique_id="ID",
        redeemed_coupon="RecruitCoupon",
        issued_coupons=["Coupon_1", "Coupon_2", "Coupon_3"],
        degree="NetworkSize",
        zero_degree="median",
        NA_degree="hotdeck"
    )

    # Check the processed data
    print(f"Seeds: {rds_data['SEED'].sum()}")
    print(f"Max wave: {rds_data['WAVE'].max()}")

    # 3. Calculate means with parallel processing
    mean_age = RDSmean(
        x='Age',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )
    print(mean_age)

    # 4. Generate frequency tables
    sex_table = RDStable(
        formula='~Sex',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000
    )
    print(sex_table)

    # Two-way table
    cross_table = RDStable(
        formula='~Sex+Race',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000,
        margins=1  # row proportions
    )
    print(cross_table)

    # 5. Fit regression models
    income_model = RDSlm(
        data=rds_data,
        formula='Income ~ Age + C(Sex) + C(Race)',
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=2000,
        n_cores=6
    )
    print(income_model)

Descriptive Statistics Examples
-------------------------------

Unweighted mean with naive variance::

    result = RDSmean(
        x='Age',
        data=rds_data,
        var_est=None  # naive method
    )

Weighted mean with bootstrap variance::

    result = RDSmean(
        x='Age',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_chain1',
        resample_n=1000
    )

Return bootstrap means for custom analysis::

    result, bootstrap_means, node_counts = RDSmean(
        x='Age',
        data=rds_data,
        var_est='resample_tree_uni1',
        resample_n=1000,
        return_bootstrap_means=True,
        return_node_counts=True
    )

    # Analyze bootstrap distribution
    import numpy as np
    print(f"Bootstrap mean: {np.mean(bootstrap_means)}")
    print(f"Bootstrap SE: {np.std(bootstrap_means)}")

Table Examples
--------------

One-way table with different margin options::

    # Simple one-way table
    table = RDStable(
        formula='~Sex',
        data=rds_data
    )

    # Weighted one-way table with bootstrap
    table = RDStable(
        formula='~Race',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=500
    )

Two-way tables with different proportions::

    # Cell proportions (default)
    table_cell = RDStable(
        formula='~Sex+Race',
        data=rds_data,
        margins=3
    )

    # Row proportions
    table_row = RDStable(
        formula='~Sex+Race',
        data=rds_data,
        margins=1
    )

    # Column proportions
    table_col = RDStable(
        formula='~Sex+Race',
        data=rds_data,
        margins=2
    )

Regression Examples
-------------------

Simple linear regression::

    model = RDSlm(
        data=rds_data,
        formula='Income ~ Age'
    )

Multiple linear regression with categorical predictors::

    model = RDSlm(
        data=rds_data,
        formula='Income ~ Age + C(Sex) + C(Education) + C(Race)',
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )

Logistic regression::

    # Binary outcome (0/1)
    model = RDSlm(
        data=rds_data,
        formula='Employed ~ Age + C(Sex) + C(Education)',
        var_est='resample_chain1',
        resample_n=500
    )

Return bootstrap estimates::

    model, boot_estimates, node_counts = RDSlm(
        data=rds_data,
        formula='Income ~ Age + C(Sex)',
        var_est='resample_tree_uni1',
        resample_n=1000,
        return_bootstrap_estimates=True,
        return_node_counts=True
    )

Network Visualization Examples
------------------------------

Basic network graph with different layouts::

    # Spring layout (default)
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        layout='Spring'
    )

    # Tree layout (hierarchical)
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1'],
        waves=[0, 1, 2, 3, 4],
        layout='Tree',
        save_path='recruitment_tree.png'
    )

    # Circular layout
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2],
        layout='Circular',
        figsize=(12, 12)
    )

Color nodes by demographic variables::

    # Color by Sex
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2', '3'],
        waves=[0, 1, 2],
        layout='Kamada-Kawai',
        group_by='Sex',
        node_size=50,
        figsize=(16, 14)
    )

    # Color by Race
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1'],
        waves=[0, 1, 2, 3],
        layout='Spring',
        group_by='Race',
        node_size=40
    )

Geographic Mapping Examples
---------------------------

Check available mapping data::

    # Print comprehensive map information
    print_map_info(rds_data, lat_column='Latitude', lon_column='Longitude')

    # Get available seeds and waves
    seeds = get_available_seeds(rds_data)
    waves = get_available_waves(rds_data)
    print(f"Seeds: {seeds}")
    print(f"Waves: {waves}")

Basic map::

    m = RDSmap(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        output_file='my_rds_map.html'
    )

Map with custom coordinates and settings::

    m = RDSmap(
        data=rds_data,
        seed_ids=['1', '2', '3'],
        waves=[0, 1, 2, 3, 4],
        lat_column='lat',
        lon_column='long',
        output_file='geographic_map.html',
        zoom_start=10,
        open_browser=True
    )

Bootstrap Examples
------------------

Standalone bootstrap resampling::

    from RDSTools import RDSboot

    # Standard bootstrap
    boot_results = RDSboot(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_uni1',
        resample_n=1000
    )

    # Check first resample
    sample_1 = boot_results[boot_results['RESAMPLE.N'] == 1]
    merged = pd.merge(sample_1, rds_data,
                     left_on='RESPONDENT_ID', right_on='ID')
    print(f"Bootstrap sample size: {len(merged)}")

Parallel bootstrap for large datasets::

    from RDSTools import RDSBootOptimizedParallel

    boot_results = RDSBootOptimizedParallel(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_uni1',
        resample_n=10000,
        n_cores=8
    )

Performance Comparison
----------------------

The parallel bootstrap provides significant speedups:

.. list-table:: Performance Comparison (252 observations)
   :header-rows: 1

   * - Cores
     - Bootstrap Samples
     - Standard Time
     - Parallel Time
     - Speedup
   * - 1
     - 1000
     - 120s
     - 120s
     - 1.0x
   * - 4
     - 1000
     - 120s
     - 18s
     - 6.7x
   * - 8
     - 1000
     - 120s
     - 12s
     - 10.0x

Complete Analysis Pipeline
---------------------------

Here's a complete pipeline from data loading to final results::

    import pandas as pd
    from RDSTools import (
        RDSdata, RDSmean, RDStable, RDSlm,
        RDSnetgraph, RDSmap, get_available_seeds
    )

    # Load data
    data = pd.read_csv("survey.csv")

    # Process RDS structure
    rds_data = RDSdata(
        data=data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree"
    )

    # Descriptive statistics
    age_mean = RDSmean(
        x='Age',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )

    # Frequency tables
    sex_table = RDStable(
        formula='~Sex',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000
    )

    race_sex_table = RDStable(
        formula='~Sex+Race',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000,
        margins=1
    )

    # Regression analysis
    model = RDSlm(
        data=rds_data,
        formula='Income ~ Age + C(Sex) + C(Race) + C(Education)',
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=2000,
        n_cores=4
    )

    # Visualizations
    seeds = get_available_seeds(rds_data)

    # Network graph
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=seeds[:2],
        waves=[0, 1, 2, 3],
        layout='Spring',
        group_by='Sex',
        save_path='network.png'
    )

    # Geographic map
    m = RDSmap(
        data=rds_data,
        seed_ids=seeds[:2],
        waves=[0, 1, 2, 3],
        output_file='map.html',
        open_browser=True
    )

    # Print results
    print("Age Mean:")
    print(age_mean)
    print("\nSex Distribution:")
    print(sex_table)
    print("\nRegression Model:")
    print(model)
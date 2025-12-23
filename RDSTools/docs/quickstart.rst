Quick Start
===========

This guide will get you started with RDS Tools in just a few minutes.

Basic Usage
-----------

Import the necessary modules::

    from RDSTools import RDSdata, RDSmean, RDStable, RDSlm, RDSnetgraph, RDSmap

Load your data::

    import pandas as pd
    data = pd.read_csv("your_survey_data.csv")

Process RDS Data
----------------

Process your raw survey data to create the RDS network structure::

    rds_processed = RDSdata(
        data=data,
        unique_id="participant_id",
        redeemed_coupon="coupon_used",
        issued_coupons=["coupon1", "coupon2", "coupon3"],
        degree="network_size"
    )

Calculate Means
---------------

Calculate means with bootstrap resampling::

    mean_results = RDSmean(
        x='age',
        data=rds_processed,
        var_est='resample_tree_uni1',
        resample_n=1000
    )

For faster processing, use parallel bootstrap::

    mean_results = RDSmean(
        x='age',
        data=rds_processed,
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4  # Use 4 cores for parallel processing
    )

Create Tables
-------------

Generate frequency tables for categorical variables::

    sex_table = RDStable(
        formula='~Sex',
        data=rds_processed,
        var_est='resample_tree_uni1',
        resample_n=1000
    )

    # Two-way table
    cross_table = RDStable(
        formula='~Sex+Race',
        data=rds_processed,
        var_est='resample_tree_uni1',
        resample_n=1000,
        margins=1  # row proportions
    )

Run Regression Models
---------------------

Fit linear and logistic regression models::

    # Linear regression
    model = RDSlm(
        data=rds_processed,
        formula='Income ~ Age + C(Sex) + C(Race)',
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )

    # Logistic regression (binary outcome)
    logit_model = RDSlm(
        data=rds_processed,
        formula='Employed ~ Age + C(Education)',
        var_est='resample_tree_uni1',
        resample_n=1000
    )

Network Visualization
---------------------

Create network graphs to visualize recruitment relationships::

    from RDSTools import RDSnetgraph

    # Basic network graph
    G = RDSnetgraph(
        data=rds_processed,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        layout='Spring'
    )

    # Color by demographic variable
    G = RDSnetgraph(
        data=rds_processed,
        seed_ids=['1', '2'],
        waves=[0, 1, 2],
        layout='Spring',
        group_by='Sex'
    )

Geographic Mapping
------------------

Create interactive maps showing participant locations::

    from RDSTools import RDSmap, get_available_seeds, print_map_info

    # Check available data
    print_map_info(rds_processed)
    seeds = get_available_seeds(rds_processed)

    # Create interactive map
    map_obj = RDSmap(
        data=rds_processed,
        seed_ids=seeds[:2],
        waves=[0, 1, 2, 3],
        output_file='participant_map.html',
        open_browser=True
    )


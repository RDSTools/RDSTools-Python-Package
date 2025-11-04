Examples
========

Complete Workflow Example
-------------------------

Here's a complete example showing how to analyze RDS data from start to finish::

    import pandas as pd
    from RDSTools import RDS_data, RDSMean, RDSTable, RDSRegression
    from RDSTools.network_graph import create_network_graph
    from RDSTools.rds_map import create_participant_map, print_map_info

    # 1. Load and examine your data
    data = pd.read_csv("rds_survey.csv")
    print(data.columns)

    # 2. Process the RDS structure
    rds_data = RDS_data(
        data=data,
        unique_id="ID",
        redeemed_coupon="RecruitCoupon",
        issued_coupons=["Coupon_1", "Coupon_2", "Coupon_3"],
        degree="NetworkSize",
        zero_degree="median",
        NA_degree="hotdeck"
    )

    # 3. Calculate means with parallel processing
    mean_age = RDSMean(
        x='age',
        data=rds_data,
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )

    # 4. Generate frequency tables
    education_table = RDSTable(
        x='education_level',
        data=rds_data,
        var_est='resample_tree_uni1',
        resample_n=1000
    )

    # 5. Fit regression models
    income_model = RDSRegression(
        formula='log_income ~ age + education_years + gender',
        data=rds_data,
        family='gaussian',
        var_est='resample_tree_uni1',
        resample_n=2000,
        n_cores=6
    )

Network Visualization Examples
------------------------------

Basic network graph::

    G = create_network_graph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        layout='Spring'
    )

Tree layout showing hierarchical structure::

    G = create_network_graph(
        data=rds_data,
        seed_ids=['1'],
        waves=[0, 1, 2, 3, 4],
        layout='Tree',
        save_path='recruitment_tree.png'
    )

Color nodes by demographic variable::

    G = create_network_graph(
        data=rds_data,
        seed_ids=['1', '2', '3'],
        waves=[0, 1, 2],
        layout='Kamada-Kawai',
        group_by='Gender',
        node_size=20,
        figsize=(16, 14)
    )

Geographic Mapping Examples
---------------------------

Basic map::

    m = create_participant_map(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        output_file='my_rds_map.html'
    )

Map with custom coordinates::

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

Performance Comparison
---------------------

The parallel bootstrap provides significant speedups:

.. list-table:: Performance Comparison
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
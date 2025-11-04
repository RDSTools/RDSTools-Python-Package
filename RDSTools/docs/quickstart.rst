Quick Start
===========

This guide will get you started with RDS Tools in just a few minutes.

Basic Usage
-----------

Import the necessary modules::

    from RDSTools import RDS_data, RDSMean, RDSTable, RDSRegression
    from RDSTools.network_graph import create_network_graph

Load your data::

    data = pd.read_csv("your_survey_data.csv")

Process RDS Data
----------------

Process your raw survey data to create the RDS network structure::

    rds_processed = RDS_data(
        data=data,
        unique_id="participant_id",
        redeemed_coupon="coupon_used",
        issued_coupons=["coupon1", "coupon2", "coupon3"],
        degree="network_size"
    )

Calculate Means
---------------

Calculate means with bootstrap resampling::

    mean_results = RDSMean(
        x='age',
        data=rds_processed,
        var_est='resample_tree_uni1',
        resample_n=1000
    )

For faster processing, use parallel bootstrap::

    mean_results = RDSMean(
        x='age',
        data=rds_processed,
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4  # Use 4 cores for parallel processing
    )

Network Visualization
---------------------

Create network graphs to visualize recruitment relationships::

    from rds_tools.network_graph import create_network_graph

    # Basic network graph
    G = create_network_graph(
        data=rds_processed,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        layout='Spring'
    )

Geographic Mapping
------------------

Create interactive maps showing participant locations::

    from rds_tools.rds_map import create_participant_map

    # Create interactive map
    map_obj = create_participant_map(
        data=rds_processed,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        output_file='participant_map.html'
    )
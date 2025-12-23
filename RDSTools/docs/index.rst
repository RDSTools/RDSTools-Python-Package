RDS Tools Documentation
=======================

RDS Tools is a Python package for Respondent-Driven Sampling (RDS) analysis and bootstrap resampling with parallel processing capabilities.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   data_processing
   estimation
   sampling_variance
   visualization
   performance
   examples

Installation
------------

.. code-block:: bash

    cd RDSTools
    pip install .

For development:

.. code-block:: bash

    pip install -e .

Quick Start
-----------

.. code-block:: python

    import pandas as pd
    from RDSTools import RDSdata, RDSmean, RDStable, RDSlm

    # Process RDS data
    data = pd.read_csv("survey_data.csv")
    rds_data = RDSdata(
        data=data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree"
    )

    # Calculate means with bootstrap variance
    result = RDSmean(
        x='Age',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4  # parallel processing
    )
    print(result)

    # Create frequency tables
    table = RDStable(
        formula='~Sex',
        data=rds_data,
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000
    )
    print(table)

    # Fit regression models
    model = RDSlm(
        data=rds_data,
        formula='Income ~ Age + C(Sex) + C(Race)',
        weight='WEIGHT',
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )
    print(model)

Key Features
------------

**Data Processing**
    - ``RDSdata()`` - Process RDS survey data and create network structure
    - Automatic wave detection and seed identification
    - Flexible degree imputation methods (mean, median, hotdeck, drop)

**Estimation**
    - ``RDSmean()`` - Calculate means with RDS-adjusted standard errors
    - ``RDStable()`` - Generate one-way and two-way frequency tables
    - ``RDSlm()`` - Linear and logistic regression models
    - Weighted and unweighted analyses
    - Naive and bootstrap variance estimation

**Bootstrap Variance**
    - ``RDSboot()`` - Six bootstrap resampling methods
    - ``RDSBootOptimizedParallel()`` - Parallel bootstrap for performance
    - Chain, tree unidirectional, and tree bidirectional methods

**Visualization**
    - ``RDSnetgraph()`` - Network visualizations with multiple layouts
    - ``RDSmap()`` - Interactive geographic maps
    - Helper functions: ``get_available_seeds()``, ``get_available_waves()``, ``print_map_info()``

**Performance**
    - Parallel processing support with ``n_cores`` parameter
    - Up to 10x speedup with 8 cores
    - Optimized algorithms for large datasets

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
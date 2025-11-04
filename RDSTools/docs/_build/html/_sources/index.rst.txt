RDS Tools Documentation
======================

RDS Tools is a Python package for Respondent-Driven Sampling (RDS) analysis and bootstrap resampling.

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

    cd rds-tools
    pip install .

For development:

.. code-block:: bash

    pip install -e .

Quick Start
-----------

.. code-block:: python

    import pandas as pd
    from RDSTools import RDS_data, RDSMean, RDSTable, RDSRegression

    # Process RDS data
    data = pd.read_csv("survey_data.csv")
    rds_data = RDS_data(
        data=data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree"
    )

    # Calculate means
    result = RDSMean(
        x='age',
        data=rds_data,
        var_est='resample_tree_uni1',
        resample_n=1000
    )

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
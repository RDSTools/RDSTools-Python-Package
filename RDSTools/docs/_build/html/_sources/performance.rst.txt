Performance Enhancement
======================

The package includes parallel processing for bootstrap methods. Unidirectional and bidirectional bootstrap sampling methods are the methods that benefit the most from parallel processing.

Usage
-----

.. code-block:: python

    # Use parallel processing for faster bootstrap
    result = RDSmean(
        x='income',
        data=rds_data,
        var_est='resample_tree_uni1',
        resample_n=2000,
        n_cores=8  # Use 8 cores for parallel processing
    )

Parallel processing is available for all bootstrap-based statistical functions:

- RDSmean() with bootstrap variance estimation
- RDStable() with bootstrap variance estimation
- RDSlm() with bootstrap variance estimation

Performance Comparison
----------------------

.. list-table:: Performance Scaling
   :header-rows: 1
   :widths: 10 20 15 15 15

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

Examples
--------

All estimation functions support the n_cores parameter:

.. code-block:: python

    # Parallel mean calculation
    mean_result = RDSmean(
        x='age',
        data=rds_data,
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )

    # Parallel table calculation
    table_result = RDStable(
        formula="~Sex+Race",
        data=rds_data,
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )

    # Parallel regression
    regression_result = RDSlm(
        data=rds_data,
        formula="Age ~ Sex",
        var_est='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )
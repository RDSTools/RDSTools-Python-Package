Estimation
==========

RDS Tools package has 3 estimation functions: (1) RDSmean, (2) RDStable, (3) RDSlm. Users can control the use of weights, selection of variance estimation, as well as the number of resamples if one of the variance estimation resampling approaches is used. Note, before using the estimation functions, please make sure that you preprocess data with RDSdata function.

RDSmean - Descriptive Statistics
=================================

For a continuous variable the basic information from RDSmean object contains point estimates, standard errors and information about the analysis. This function calculates means and standard errors for RDS data with optional weighting and different variance estimation methods.

Usage
-----

.. code-block:: python

    RDSmean(x, data, weight=None, var_est=None, resample_n=None, n_cores=None, return_bootstrap_means=False, return_node_counts=False)

Arguments
---------

**x**
    str. A variable of interest - name of the column in the data to calculate mean for.

**data**
    pandas.DataFrame. The output DataFrame from RDSdata function containing preprocessed RDS data.

**weight**
    str, optional. Name of the weight variable. User specified weights to calculate weighted mean and standard errors. When set to None the function calculates unweighted mean and standard errors. Default is None.

**var_est**
    str, optional. One of the six bootstrap types or the delta (naive) method. By default the function calculates naive standard errors. Variance estimation options include 'naive' or bootstrap methods like 'chain1', 'chain2', 'tree_uni1', 'tree_uni2', 'tree_bi1', 'tree_bi2'. Default is None (naive).

**resample_n**
    int, optional. Specifies the number of resample iterations. Note that this argument is None when var_est = 'naive'. Required for bootstrap methods, default 300.

**n_cores**
    int, optional. Number of CPU cores to use for parallel bootstrap processing. If specified, uses optimized parallel bootstrap. If None, uses standard sequential bootstrap. Default is None.

**return_bootstrap_means**
    bool, optional. If True, return bootstrap mean estimates along with main results (only for bootstrap methods). Default is False.

**return_node_counts**
    bool, optional. If True, return node counts per iteration along with main results (only for bootstrap methods). Default is False.

Returns
-------

**RDSResult or tuple**
    When return_bootstrap_means and return_node_counts are both False (default):
        RDSResult object (DataFrame subclass) containing weighted or unweighted mean with standard errors, additional information about the analysis: (1) var_est method, (2) weighted or not, (3) n_Data, (4) n_Analysis, (5) n_Iteration if var_est is not naive, descriptive summary of resamples if var_est is not naive, resample estimates.

    When return_bootstrap_means is True and return_node_counts is False:
        tuple of (RDSResult, bootstrap_means_list).

    When return_bootstrap_means is False and return_node_counts is True:
        tuple of (RDSResult, node_counts_list).

    When both return_bootstrap_means and return_node_counts are True:
        tuple of (RDSResult, bootstrap_means_list, node_counts_list).

Examples
--------

.. code-block:: python

    from RDSTools import RDSmean

    # Basic mean with naive variance estimation
    result = RDSmean(x='Age', data=rds_data, var_est='naive')

    # Weighted analysis with inverse weights
    result = RDSmean(x='Age', data=rds_data, weight='WEIGHT')

    # Bootstrap method with resampling
    result = RDSmean(
        x='Age',
        data=rds_data,
        weight='WEIGHT',
        var_est='chain1',
        resample_n=1000
    )

    # Parallel processing with 4 cores
    result = RDSmean(
        x='Age',
        data=rds_data,
        var_est='tree_uni1',
        resample_n=1000,
        n_cores=4
    )

    # Return bootstrap means and node counts
    result, bootstrap_means, node_counts = RDSmean(
        x='Age',
        data=rds_data,
        var_est='tree_uni1',
        resample_n=1000,
        return_bootstrap_means=True,
        return_node_counts=True
    )

RDStable - Contingency Tables
==============================

The package allows users to estimate one-way and two-way tables, using either naive or resampling approaches to estimate the standard errors of proportions. Creates contingency tables from RDS data with optional weighting and different variance estimation methods.

Usage
-----

.. code-block:: python

    RDStable(x, y=None, data=None, weight=None, var_est=None, resample_n=None, margins=3, n_cores=None, return_bootstrap_tables=False, return_node_counts=False)

Arguments
---------

**x**
    str. Column name; For a 1-way table, specify one categorical variable. By default the function returns a 1-way table.

**y**
    str, optional. Column name; Optional, for 2-way tables specify the second categorical variable of interest. Default is None.

**data**
    pandas.DataFrame. The output from RDSdata function containing preprocessed RDS data.

**weight**
    str, optional. Name of the weight variable. A user specified weights to calculate weighted proportions and standard errors. When set to None the function calculates unweighted proportions and standard errors. Default is None.

**var_est**
    str, optional. One of the six bootstrap types or the delta (naive) method. By default the function calculates naive standard errors. Variance estimation options include 'naive' or bootstrap methods like 'chain1', 'chain2', 'tree_uni1', 'tree_uni2', 'tree_bi1', 'tree_bi2'. Default is None (naive).

**resample_n**
    int, optional. Specifies the number of resample iterations. Note that this argument is None when var_est = 'naive'. Required for bootstrap methods, default 300.

**margins**
    int, optional. For two-way tables: 1=row proportions, 2=column proportions, 3=cell proportions. Default is 3 (cell proportions).

**n_cores**
    int, optional. Number of CPU cores to use for parallel bootstrap processing. If specified, uses optimized parallel bootstrap. If None, uses standard sequential bootstrap. Default is None.

**return_bootstrap_tables**
    bool, optional. If True, return bootstrap table estimates along with main results (only for bootstrap methods). Default is False.

**return_node_counts**
    bool, optional. If True, return node counts per iteration along with main results (only for bootstrap methods). Default is False.

Returns
-------

**RDSTableResult or tuple**
    When return_bootstrap_tables and return_node_counts are both False (default):
        RDSTableResult object containing weighted or unweighted proportions and their standard errors, additional information about the analysis: (1) var_est method, (2) weighted or not, (3) n_Data, (4) n_Analysis, (5) n_Iteration if var_est is not naive, descriptive summary of resamples if var_est is not naive, resample estimates.

    When return_bootstrap_tables is True and return_node_counts is False:
        tuple of (RDSTableResult, bootstrap_tables_list).

    When return_bootstrap_tables is False and return_node_counts is True:
        tuple of (RDSTableResult, node_counts_list).

    When both return_bootstrap_tables and return_node_counts are True:
        tuple of (RDSTableResult, bootstrap_tables_list, node_counts_list).

Examples
--------

.. code-block:: python

    from RDSTools import RDStable

    # One-way table
    result = RDStable(x="Sex", data=rds_data)

    # Two-way table with bootstrap variance estimation
    result = RDStable(
        x="Sex",
        y="Race",
        data=rds_data,
        var_est='chain1',
        resample_n=100
    )

    # Two-way table with row proportions and parallel processing
    result = RDStable(
        x="Sex",
        y="Race",
        data=rds_data,
        var_est='tree_uni1',
        resample_n=1000,
        margins=1,  # row proportions
        n_cores=4
    )

    # Return bootstrap tables and node counts
    result, bootstrap_tables, node_counts = RDStable(
        x="Sex",
        data=rds_data,
        var_est='tree_uni1',
        resample_n=1000,
        return_bootstrap_tables=True,
        return_node_counts=True
    )

RDSlm - Linear and Logistic Regression
=======================================

Users can specify linear and logistic regression models using the RDSlm function. The function performs a linear regression when the dependent variable is numeric and a logistic regression with binomial link function when the dependent variable is either character or factor.

Usage
-----

.. code-block:: python

    RDSlm(data, formula, weight=None, var_est=None, resample_n=None, n_cores=None, return_bootstrap_estimates=False, return_node_counts=False)

Arguments
---------

**data**
    pandas.DataFrame. The output from RDSdata function containing preprocessed RDS data.

**formula**
    str. Description of the model with dependent and independent variables. Note that the functions performs a linear regression when the dependent variable is numeric and a logistic regression with binomial link function when the dependent variable is either character or factor. (e.g., "y ~ x1 + x2").

**weight**
    str, optional. Name of the weight variable. A user specified weights to calculate weighted point estimates and standard errors. When set to None the function calculates unweighted point estimates and standard errors. Default is None.

**var_est**
    str, optional. One of the six bootstrap types or the delta (naive) method. By default the function calculates naive standard errors. Variance estimation options include 'naive' or bootstrap methods like 'chain1', 'chain2', 'tree_uni1', 'tree_uni2', 'tree_bi1', 'tree_bi2'. Default is None (naive).

**resample_n**
    int, optional. Specifies the number of resample iterations. Note that this argument is None when var_est = 'naive'. Required for bootstrap methods, default 300.

**n_cores**
    int, optional. Number of CPU cores to use for parallel bootstrap processing. If specified, uses optimized parallel bootstrap. If None, uses standard sequential bootstrap. Default is None.

**return_bootstrap_estimates**
    bool, optional. If True, return bootstrap coefficient estimates along with main results (only for bootstrap methods). Default is False.

**return_node_counts**
    bool, optional. If True, return node counts per iteration along with main results (only for bootstrap methods). Default is False.

Returns
-------

**RDSRegressionResult or tuple**
    When return_bootstrap_estimates and return_node_counts are both False (default):
        RDSRegressionResult object containing point estimates, se, t-values and p-values, linear and logistic regression specific model fit statistics, additional information about the analysis: (1) var_est method, (2) weighted or not, (3) n_Data, (4) n_Analysis, (5) n_Iteration if var_est is not naive, descriptive summary of resamples if var_est is not naive, resample estimates.

    When return_bootstrap_estimates is True and return_node_counts is False:
        tuple of (RDSRegressionResult, bootstrap_estimates_list).

    When return_bootstrap_estimates is False and return_node_counts is True:
        tuple of (RDSRegressionResult, node_counts_list).

    When both return_bootstrap_estimates and return_node_counts are True:
        tuple of (RDSRegressionResult, bootstrap_estimates_list, node_counts_list).

Examples
--------

.. code-block:: python

    from RDSTools import RDSlm

    # Linear regression (continuous dependent variable)
    result = RDSlm(
        data=rds_data,
        formula="Age ~ C(Sex)",
        weight='WEIGHT',
        var_est='chain1',
        resample_n=1000
    )

    # Logistic regression (categorical dependent variable)
    # Make sure to convert to binary (0,1) if doesn't work.
    result = RDSlm(
        data=rds_data,
        formula="Employed ~ Age + C(Sex)",
        weight='WEIGHT',
        var_est='chain1',
        resample_n=100
    )

    # Parallel regression with multiple predictors
    result = RDSlm(
        data=rds_data,
        formula="Income ~ Age + C(Education) + C(Race)",
        var_est='tree_uni1',
        resample_n=1000,
        n_cores=4
    )

    # Return bootstrap estimates and node counts
    result, bootstrap_estimates, node_counts = RDSlm(
        data=rds_data,
        formula="Age ~ C(Sex)",
        var_est='tree_uni1',
        resample_n=1000,
        return_bootstrap_estimates=True,
        return_node_counts=True
    )


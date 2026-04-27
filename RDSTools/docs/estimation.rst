Estimation
==========

RDS Tools package has 3 estimation functions: (1) RDSmean, (2) RDStable, (3) RDSlm. Users can control the use of weights, selection of variance estimation, as well as the number of resamples if one of the variance estimation resampling approaches is used. Note, before using the estimation functions, please make sure that you preprocess data with RDSdata function.

RDSmean - Descriptive Statistics
=================================

Estimating mean with respondent driven sampling sample data. This function calculates weighted or unweighted means for either a continuous or a categorical variable. For continuous variables, it returns a single mean and standard error. For categorical variables, it returns one proportion and standard error per level (consistent with applying ``svymean`` to a factor in R). Standard errors are calculated using naive (delta-method) or resampling approaches from 'RDSboot'.

Usage
-----

.. code-block:: python

    RDSmean(x, data, weight=None, var_est=None, resample_n=None, n_cores=None, na_rm=True, return_bootstrap_means=False, return_node_counts=False)

Arguments
---------

**x**
    str. A variable of interest. Continuous variables (numeric dtypes) return a single mean and SE. Categorical variables (object, string, bool, or pandas Categorical dtypes) return one proportion and SE per level. Note that integer-coded categoricals such as ``Race=1,2,3`` are treated as numeric by default; convert them with ``data[col] = data[col].astype('category')`` before calling ``RDSmean`` to get per-category output.

**data**
    pandas.DataFrame. The output DataFrame from RDSdata.

**weight**
    str, optional. Name of the weight variable. User specified weight variable for a weighted analysis. When set to None, the function performs an unweighted analysis. Default is None.

**var_est**
    str, optional. One of the six bootstrap types or the delta (naive) method. By default the function calculates naive standard errors. Variance estimation options include 'naive' or bootstrap methods like 'chain1', 'chain2', 'tree_uni1', 'tree_uni2', 'tree_bi1', 'tree_bi2'. Default is None (naive).

**resample_n**
    int, optional. Specifies the number of resample iterations. Note that this argument is None when var_est = 'naive'. Required for bootstrap methods, default 300.

**n_cores**
    int, optional. Number of CPU cores to use for parallel bootstrap processing. If specified, uses optimized parallel bootstrap. If None, uses standard sequential bootstrap. Default is None.

**na_rm**
    bool, optional. If True (default), observations with missing values in ``x`` (or in the weight column, when supplied) are removed before estimation. If False, missing values are retained and the estimator returns ``NaN`` whenever NAs are present, mirroring R's ``svymean(..., na.rm = FALSE)`` behaviour. Default is True.

**return_bootstrap_means**
    bool, optional. If True, return the per-iteration estimates along with the main results (only for bootstrap methods). For continuous variables this is a list of scalar means; for categorical variables it is a list of proportion arrays aligned with the level order. Default is False.

**return_node_counts**
    bool, optional. If True, return sample size per iteration along with main results (only for bootstrap methods). Default is False.

Returns
-------

**RDSResult or tuple**
    An RDSResult object containing the following elements:

    results
        DataFrame; A tidy results table. For continuous variables, columns are ``Mean`` and ``SE`` with a single row. For categorical variables, columns are ``Category``, ``Mean``, and ``SE`` with one row per level. For categorical variables, the reported "Mean" for each level is the estimated proportion of observations in that level.

    additional_info
        Information about the estimation:
        (1) SE method: variance estimation method
        (2) Weight: indicator of whether weighted analysis was used
        (3) n_Data: total number of observations in the input data
        (4) n_Analysis: number of observations used in the analysis (after NA removal when ``na_rm=True``)
        (5) n_Iteration: number of resampling iterations (if SE method is not 'naive')

    resample_summary
        Descriptive summary of resamples if var_est is not 'naive': mean, SD, min, quartiles, and max of resample sizes

    resample_estimates
        Per-iteration estimates if var_est is not 'naive'. For continuous variables, a list of scalar means (one per iteration). For categorical variables, a list of proportion arrays (one per iteration, aligned with the level order).

    When return_bootstrap_means=False and return_node_counts=False (default):
        Returns RDSResult object only

    When return_bootstrap_means=True and return_node_counts=False:
        Returns (RDSResult, bootstrap_estimates_list)

    When return_bootstrap_means=False and return_node_counts=True:
        Returns (RDSResult, node_counts_list)

    When return_bootstrap_means=True and return_node_counts=True:
        Returns (RDSResult, bootstrap_estimates_list, node_counts_list)

Notes
-----
The RDSResult object is a pandas DataFrame subclass that:
    - Retains all DataFrame functionality for analysis
    - Has custom print formatting for clean display
    - Exposes the tidy results table via ``result.results`` and the underlying bootstrap estimates and node counts as attributes

For categorical variables, the reported "Mean" for each level is the estimated proportion of observations in that level. Each level has its own standard error

Integer-coded categorical variables (such as ``Race=1,2,3``) are treated as numeric by default and will produce a single mean rather than per-category proportions. To obtain per-category output, convert the column with ``data[col] = data[col].astype('category')`` before calling ``RDSmean``.

Examples
--------

.. code-block:: python

    from RDSTools import RDSmean

    # Basic mean with naive variance estimation (continuous variable)
    result = RDSmean(x='Age', data=rds_data, var_est='naive')

    # Weighted analysis with inverse weights
    result = RDSmean(x='Age', data=rds_data, weight='WEIGHT')

    # Categorical variable: convert to category dtype first
    rds_data['Race'] = rds_data['Race'].astype('category')
    result = RDSmean(x='Race', data=rds_data, weight='WEIGHT')
    # Output is a tidy table with one row per Race level

    # Retain NAs and propagate to NaN (instead of dropping)
    result = RDSmean(x='Age', data=rds_data, na_rm=False)

    # Bootstrap method with resampling
    result = RDSmean(
        x='Age',
        data=rds_data,
        weight='WEIGHT',
        var_est='chain1',
        resample_n=1000
    )

    # Categorical bootstrap: per-category proportions and bootstrap SEs
    rds_data['Race'] = rds_data['Race'].astype('category')
    result = RDSmean(
        x='Race',
        data=rds_data,
        weight='WEIGHT',
        var_est='chain1',
        resample_n=300
    )

    # Parallel processing with 4 cores
    result = RDSmean(
        x='Age',
        data=rds_data,
        var_est='tree_uni1',
        resample_n=1000,
        n_cores=4
    )

    # Return bootstrap estimates and node counts
    result, bootstrap_estimates, node_counts = RDSmean(
        x='Age',
        data=rds_data,
        var_est='tree_uni1',
        resample_n=1000,
        return_bootstrap_means=True,
        return_node_counts=True
    )

RDStable - Contingency Tables
==============================

Estimating one and two-way tables with respondent driven sampling sample data. One-way tables are constructed by specifying a categorical variable for x argument only. Two-way tables are constructed by specifying two categorical variables for x and y arguments. Standard errors of proportions are calculated using naive or resampling approaches from 'RDSboot'.

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
    pandas.DataFrame. The output DataFrame from RDSdata.

**weight**
    str, optional. Name of the weight variable. User specified weight variable for a weighted analysis. When set to NULL, the function performs an unweighted analysis. Default is None.

**var_est**
    str, optional. One of the six bootstrap types or the delta (naive) method. By default the function calculates naive standard errors. Variance estimation options include 'naive' or bootstrap methods like 'chain1', 'chain2', 'tree_uni1', 'tree_uni2', 'tree_bi1', 'tree_bi2'. Default is None (naive).

**resample_n**
    int, optional. Specifies the number of resample iterations. Note that this argument is None when var_est = 'naive'. Required for bootstrap methods, default 300.

**margins**
    int, optional. For two-way tables: 1=row proportions, 2=column proportions, 3=cell proportions (default). Default is 3.

**n_cores**
    int, optional. Number of CPU cores to use for parallel bootstrap processing. If specified, uses optimized parallel bootstrap. If None, uses standard sequential bootstrap. Default is None.

**return_bootstrap_tables**
    bool, optional. If True, return bootstrap table estimates along with main results (only for bootstrap methods). Default is False.

**return_node_counts**
    bool, optional. If True, return sample size per iteration along with main results (only for bootstrap methods). Default is False.

Returns
-------

**RDSTableResult or tuple**
    An RDSTableResult object containing the following elements:

    formula
        Formula; Variable(s) used for the estimation

    results
        DataFrame or tables; Weighted or unweighted proportions (prop_table) and their standard errors (se_table)

    additional_info
        Information about the estimation:
        (1) SE method: variance estimation method
        (2) Weight: indicator of whether weighted analysis was used
        (3) n_Data: total number of observations in the input data
        (4) n_Iteration: number of resampling iterations (if SE method is not 'naive')

    resample_summary
        Descriptive summary of resamples if var_est is not 'naive': mean, SD, min, quartiles, and max of resample sizes

    resample_estimates
        Proportions calculated for each resampling iteration if var_est is not 'naive'

    When return_bootstrap_tables=False and return_node_counts=False (default):
        Returns RDSTableResult object only

    When return_bootstrap_tables=True and return_node_counts=False:
        Returns (RDSTableResult, bootstrap_tables_list)

    When return_bootstrap_tables=False and return_node_counts=True:
        Returns (RDSTableResult, node_counts_list)

    When return_bootstrap_tables=True and return_node_counts=True:
        Returns (RDSTableResult, bootstrap_tables_list, node_counts_list)

Notes
-----
The RDSTableResult object is a custom class that:
    - Provides formatted display of contingency tables
    - Includes cell counts, proportions, and standard errors
    - Supports different margin calculations (row, column, cell)
    - Provides access to bootstrap tables and node counts

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

Linear and Logistic Regression Modeling with Respondent Driven Sampling (RDS) Sample Data. This function mimics the lm function in R stats package with capabilities to handle RDS data in model estimation. Standard errors of regression coefficients are calculated using naive or resampling approaches from 'RDSboot'.

Usage
-----

.. code-block:: python

    RDSlm(data, formula, weight=None, var_est=None, resample_n=None, n_cores=None, return_bootstrap_estimates=False, return_node_counts=False)

Arguments
---------

**data**
    pandas.DataFrame. The output DataFrame from RDSdata.

**formula**
    str. Description of the model with dependent and independent variables. (e.g., "y ~ x1 + x2"). Note that the function performs linear regression when the dependent variable is numeric and logistic regression with binomial link function when the dependent variable is either character or factor.

**weight**
    str, optional. Name of the weight variable. User specified weight variable for a weighted analysis. When set to NULL, the function performs an unweighted analysis. Default is None.

**var_est**
    str, optional. One of the six bootstrap types or the delta (naive) method. By default, the function calculates naive standard errors. Variance estimation options include 'naive' or bootstrap methods like 'chain1', 'chain2', 'tree_uni1', 'tree_uni2', 'tree_bi1', 'tree_bi2'. Default is None (naive).

**resample_n**
    int, optional. Specifies the number of resample iterations. Note that this argument is None when var_est = 'naive'. Required for bootstrap methods, default 300.

**n_cores**
    int, optional. Number of CPU cores to use for parallel bootstrap processing. If specified, uses optimized parallel bootstrap. If None, uses standard sequential bootstrap. Default is None.

**return_bootstrap_estimates**
    bool, optional. If True, return bootstrap coefficient estimates along with main results (only for bootstrap methods). Default is False.

**return_node_counts**
    bool, optional. If True, return sample size per iteration along with main results (only for bootstrap methods). Default is False.

Returns
-------

**RDSRegressionResult or tuple**
    An RDSRegressionResult object containing the following elements:

    formula
        Formula; Variable(s) used for the estimation

    coefficients
        DataFrame; Point estimates, standard errors, t-values (or z-values for logistic), and p-values for each coefficient

    model_fit
        Model fit statistics; For linear regression: R-squared, Adjusted R-squared, F-statistic, and residual standard error. For logistic regression: null deviance, residual deviance, and AIC

    additional_info
        Information about the estimation:
        (1) SE method: variance estimation method
        (2) Weight: indicator of whether weighted analysis was used
        (3) n_Data: total number of observations in the input data
        (4) n_Iteration: number of resampling iterations (if SE method is not 'naive')

    resample_summary
        Descriptive summary of resamples if var_est is not 'naive': mean, SD, min, quartiles, and max of resample sizes

    resample_estimates
        Coefficient estimates for each resampling iteration if var_est is not 'naive'

    When return_bootstrap_estimates=False and return_node_counts=False (default):
        Returns RDSRegressionResult object only

    When return_bootstrap_estimates=True and return_node_counts=False:
        Returns (RDSRegressionResult, bootstrap_estimates_list)

    When return_bootstrap_estimates=False and return_node_counts=True:
        Returns (RDSRegressionResult, node_counts_list)

    When return_bootstrap_estimates=True and return_node_counts=True:
        Returns (RDSRegressionResult, bootstrap_estimates_list, node_counts_list)

Notes
-----
In all bootstrap methods, versions 1 and 2 differ as version 1 sets the number of seeds in a given resample to be consistent with the number of seeds in the original sample (:math:`s`), while version 2 sets the sample size of a given resample (:math:`n_r`) to be at least equal to or greater than the original sample (:math:`n_s`).

'chain1' selects :math:`s` seeds using SRSWR from all seeds in the original sample and then all nodes in the chains created by each of the resampled seeds are retained. With 'chain2', 1 seed is sampled using SRSWR from all seeds in the original sample, and all nodes from the chain created by this seed are retained. It then compares :math:`n_r` against :math:`n_s`, and, if :math:`n_r < n_s`, continues the resampling process by drawing 1 seed and its chains one by one until :math:`n_r \geq n_s`.

In the 'tree_uni1' method, :math:`s` seeds are selected using Simple Random Sampling with Replacement (SRSWR) from all seeds. For each selected seed, this method (A) checks its recruit counts, (B) selects SRSWR of the recruits counts from all recruits identified in (A), and (C) for each sampled recruit, this method repeats Steps A and B. (D) Steps A, B, and C continue until reaching the last wave of each chain. In 'tree_uni2', instead of selecting :math:`s` seeds, it selects one seed, performs Steps B and C for the selected seed. It compares the size of the resample (:math:`n_r`) and the original sample (:math:`n_s`), and, if :math:`n_r < n_s`, it continues the resampling process by drawing 1 seed, performs Steps B and C and checks :math:`n_r` against :math:`n_s`. If :math:`n_r < n_s`, the process continues until the sample size of a given resample (:math:`n_r`) is at least equal to the original sample size (:math:`n_s`), i.e., :math:`n_r \geq n_s`.

'tree_bi1' selects :math:`s` nodes from the recruitment chains using SRSWR. For each selected node, it (A) checks its connected nodes (i.e., both recruiters and recruits) and their count, (B) from all connected nodes identified in (A), performs SRSWR of the same node count, and (C) for each selected node, performs steps A and B, but does not resample already resampled nodes. (D) Steps A, B, and C are repeated until the end of the chain. In 'tree_bi2', instead of :math:`s` nodes, it selects 1 node using SRSWR from anywhere in all recruitment chains and repeats steps (B),(C), and (D) until :math:`n_r \geq n_s`.

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


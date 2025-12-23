Sampling Variance
=================

Variance estimation with bootstrap chain and tree methods. Although resampling is incorporated within the estimation functions, users who wish to perform resampling separately can use RDSboot or RDSBootOptimizedParallel. After preprocessing with RDSdata, ensure the presence of at least four variables: respondent ID, seed ID, seed indicator, and recruiter ID. Note that the sampling of respondents (seeds and recruits) is conducted with replacement, and the resulting data frame will contain duplicates.

RDSboot - Standard Bootstrap
=============================

Standard bootstrap resampling for Respondent-Driven Sampling (RDS) data.

Usage
-----

.. code-block:: python

    RDSboot(data, respondent_id_col, seed_id_col, seed_col, recruiter_id_col, type, resample_n)

Arguments
---------

**data**
    pandas.DataFrame. The input DataFrame containing RDS data

**respondent_id_col**
    str. Name of the column containing respondent IDs - A variable indicating respondent ID

**seed_id_col**
    str. Name of the column containing seed IDs - A variable indicating seed ID

**seed_col**
    str. Name of the column containing seed indicators - A variable indicating whether a particular respondent is seed or not

**recruiter_id_col**
    str. Name of the column containing recruiter IDs - A variable indicating recruiter ID

**type**
    str. One of the six types of bootstrap methods: (1) resample_chain1, (2) resample_chain2, (3) resample_tree_uni1, (4) resample_tree_uni2, (5) resample_tree_bi1, (6) resample_tree_bi2.

**resample_n**
    int. A specified number of resamples

Returns
-------

**pandas.DataFrame**
    Returns a data frame consisting of the following elements:

    * **RESPONDENT_ID**: A variable indicating respondent ID
    * **RESAMPLE.N**: An indicator variable for each resample iteration

Example
-------

.. code-block:: python

    from RDSTools import RDSboot

    # Bootstrap resampling
    boot_results = RDSboot(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_uni1',
        resample_n=1000
    )

RDSBootOptimizedParallel - Parallel Bootstrap
==============================================

Combined optimized + parallel bootstrap resampling for RDS data. Provides significant performance improvements through dictionary-based lookups (1.2-1.6x speedup) and multi-core parallelization (8.0x speedup) for total potential speedup of 9.6x faster.

Usage
-----

.. code-block:: python

    RDSBootOptimizedParallel(data, respondent_id_col, seed_id_col, seed_col, recruiter_id_col, type, resample_n, n_cores=2)

Arguments
---------

**data**
    pandas.DataFrame. The input DataFrame containing RDS data

**respondent_id_col**
    str. Name of the column containing respondent IDs - A variable indicating respondent ID

**seed_id_col**
    str. Name of the column containing seed IDs - A variable indicating seed ID

**seed_col**
    str. Name of the column containing seed indicators - A variable indicating whether a particular respondent is seed or not

**recruiter_id_col**
    str. Name of the column containing recruiter IDs - A variable indicating recruiter ID

**type**
    str. One of the six types of bootstrap methods: (1) resample_chain1, (2) resample_chain2, (3) resample_tree_uni1, (4) resample_tree_uni2, (5) resample_tree_bi1, (6) resample_tree_bi2.

**resample_n**
    int. A specified number of resamples

**n_cores**
    int, optional. Number of cores to use for parallel processing. If None, uses all available cores. Default is 2.

Returns
-------

**pandas.DataFrame**
    Returns a data frame consisting of the following elements:

    * **RESPONDENT_ID**: A variable indicating respondent ID
    * **RESAMPLE.N**: An indicator variable for each resample iteration

Example
-------

.. code-block:: python

    from RDSTools import RDSBootOptimizedParallel

    # Parallel bootstrap resampling with 4 cores
    boot_results = RDSBootOptimizedParallel(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_uni1',
        resample_n=1000,
        n_cores=4
    )

Bootstrap Methods
=================

Six bootstrap methods are available: resample_chain1, resample_chain2, resample_tree_uni1, resample_tree_uni2, resample_tree_bi1, and resample_tree_bi2.

In all resampling functions, versions 1 and 2 differ as 1 focuses on the number of seeds in a given resample to be consistent with the original sample, while 2 keeps the overall sample size of a given resample to be at least equal to the original sample.

Bootstrap Chain Methods
-----------------------

**resample_chain1**
    (n) seeds are selected using Simple Random Sampling with Replacement (SRSWR), with all nodes in the chains created by resampled seeds retained. The number of selected seeds equals the number of seeds in the original data frame. Since the seeds are selected with replacement, the resulting data frame will contain exactly the same number of seeds as the original, but a different number of recruits.

.. code-block:: python

    # Chain bootstrap 1
    res_chain1 = RDSboot(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_chain1',
        resample_n=1000
    )

**resample_chain2**
    1 seed is sampled using SRSWR, with all nodes retained in the chain. The process continues until the sample size of a given resample (n_r) is at least equal to the original sample size (n_s). Selects only 1 seed at each iteration. The resulting number of seeds will vary, but the number of recruits will be equal or larger to the original number of recruits.

.. code-block:: python

    # Chain bootstrap 2
    res_chain2 = RDSboot(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_chain2',
        resample_n=1000
    )

Resample Tree Unidirectional Methods
------------------------------------

**resample_tree_uni1**
    (n) seeds are selected using SRSWR. For each selected seed, the function (A) checks its recruit counts, (B) performs SRSWR on the recruits counts from all recruits identified in (A), and (C) for each sampled recruit, repeats steps A and B. Steps A, B, and C are performed until the last wave of the chain. Since all seeds are selected with replacement, the resulting number of seeds will equal the number of seeds from the original data, but the number of recruits will vary.

.. code-block:: python

    # Tree unidirectional 1
    res_uni1 = RDSboot(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_uni1',
        resample_n=1000
    )

**resample_tree_uni2**
    Instead of selecting (n) seeds, the function selects one seed at a time and then performs steps A, B, and C for each wave of respondents. Samples only 1 seed at a time and then performs sampling with replacement from each wave of the seed's recruits. The resulting data frame will have at least the original sample size, but a varying number of seeds.

.. code-block:: python

    # Tree unidirectional 2
    res_uni2 = RDSboot(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_uni2',
        resample_n=1000
    )

Bootstrap Tree Bidirectional Methods
------------------------------------

**resample_tree_bi1**
    Selects (n) nodes from the recruitment chains using SRSWR. For each selected node, it (A) checks its connected nodes, (B) performs SRSWR on all connected nodes identified in (A), and (C) for each selected node, performs steps A and B, but does not resample already resampled nodes. (D) Steps A, B, and C are repeated until the end of the chain. The function starts from multiple nodes, depending on the number of seeds.

.. code-block:: python

    # Tree bidirectional 1
    results_bi1 = RDSboot(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_bi1',
        resample_n=1000
    )

**resample_tree_bi2**
    1 node is selected using SRSWR from the recruitment chain and steps A, B, C, and D are performed as in resample_tree_bi1. The function samples one node at a time and then evaluates whether the resulting sample is at least equal to the size of the original data. If not, the function continues resampling until the desired number of respondents is achieved.

.. code-block:: python

    # Tree bidirectional 2
    results_bi2 = RDSboot(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_bi2',
        resample_n=1000
    )

Working with Results
====================

The bootstrap results can be merged with the original data to examine resampled characteristics:

.. code-block:: python

    # Get first bootstrap sample
    sample_1 = boot_results[boot_results['RESAMPLE.N'] == 1]

    # Merge with original data
    merged = pd.merge(sample_1, rds_data,
                     left_on='RESPONDENT_ID', right_on='ID')

    # Check characteristics
    print(f"Original sample size: {len(rds_data)}")
    print(f"Bootstrap sample size: {len(merged)}")
    print(f"Original seeds: {rds_data['SEED'].sum()}")
    print(f"Bootstrap seeds: {merged['SEED'].sum()}")

Performance Considerations
==========================

For large datasets or high numbers of resamples, consider using the parallel version:

.. code-block:: python

    # For large-scale bootstrap operations
    boot_results = RDSBootOptimizedParallel(
        data=rds_data,
        respondent_id_col='ID',
        seed_id_col='S_ID',
        seed_col='SEED',
        recruiter_id_col='R_ID',
        type='resample_tree_uni1',
        resample_n=10000,  # Large number of resamples
        n_cores=8  # Use 8 cores for parallel processing
    )
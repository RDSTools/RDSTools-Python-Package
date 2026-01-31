Sampling Variance
=================

Variance estimation with bootstrap chain and tree methods. Although resampling is incorporated within the estimation functions, users who wish to perform resampling separately can use RDSboot or RDSBootOptimizedParallel. After preprocessing with RDSdata, ensure the presence of at least four variables: respondent ID, seed ID, seed indicator, and recruiter ID. Note that the sampling of respondents (seeds and recruits) is conducted with replacement, and the resulting data frame will contain duplicates.

There are six bootstrap methods available: 'chain1', 'chain2', 'tree_uni1', 'tree_uni2', 'tree_bi1', 'tree_bi2'

In all bootstrap methods, versions 1 and 2 differ as version 1 sets the number of seeds in a given resample to be consistent with the number of seeds in the original sample (:math:`s`), while version 2 sets the sample size of a given resample (:math:`n_r`) to be at least equal to or greater than the original sample (:math:`n_s`).

'chain1' selects :math:`s` seeds using SRSWR from all seeds in the original sample and then all nodes in the chains created by each of the resampled seeds are retained. With 'chain2', 1 seed is sampled using SRSWR from all seeds in the original sample, and all nodes from the chain created by this seed are retained. It then compares :math:`n_r` against :math:`n_s`, and, if :math:`n_r < n_s`, continues the resampling process by drawing 1 seed and its chains one by one until :math:`n_r \geq n_s`.

In the 'tree_uni1' method, :math:`s` seeds are selected using Simple Random Sampling with Replacement (SRSWR) from all seeds. For each selected seed, this method (A) checks its recruit counts, (B) selects SRSWR of the recruits counts from all recruits identified in (A), and (C) for each sampled recruit, this method repeats Steps A and B. (D) Steps A, B, and C continue until reaching the last wave of each chain. In 'tree_uni2', instead of selecting :math:`s` seeds, it selects one seed, performs Steps B and C for the selected seed. It compares the size of the resample (:math:`n_r`) and the original sample (:math:`n_s`), and, if :math:`n_r < n_s`, it continues the resampling process by drawing 1 seed, performs Steps B and C and checks :math:`n_r` against :math:`n_s`. If :math:`n_r < n_s`, the process continues until the sample size of a given resample (:math:`n_r`) is at least equal to the original sample size (:math:`n_s`), i.e., :math:`n_r \geq n_s`.

'tree_bi1' selects :math:`s` nodes from the recruitment chains using SRSWR. For each selected node, it (A) checks its connected nodes (i.e., both recruiters and recruits) and their count, (B) from all connected nodes identified in (A), performs SRSWR of the same node count, and (C) for each selected node, performs steps A and B, but does not resample already resampled nodes. (D) Steps A, B, and C are repeated until the end of the chain. In 'tree_bi2', instead of :math:`s` nodes, it selects 1 node using SRSWR from anywhere in all recruitment chains and repeats steps (B),(C), and (D) until :math:`n_r \geq n_s`.


RDSboot - Standard Bootstrap
=============================

Bootstrap Resampling for Respondent Driven Sampling (RDS). This function performs resampling RDS sample data by bootstrapping edges in recruitment trees or bootstrapping recruitment chains as a whole.


Usage
-----

.. code-block:: python

    RDSboot(data, respondent_id_col, seed_id_col, seed_col, recruiter_id_col, type, resample_n)

Arguments
---------

**data**
    pd.DataFrame. The input DataFrame containing RDS data.

**respondent_id_col**
    str. Name of the column containing respondent IDs - A variable indicating respondent ID.

**seed_id_col**
    str. Name of the column containing seed IDs - A variable indicating seed ID.

**seed_col**
    str. Name of the column containing seed indicators - A variable indicating whether a particular respondent is seed or not.

**recruiter_id_col**
    str. Name of the column containing recruiter IDs - A variable indicating recruiter ID.

**type**
    str. One of the six types of bootstrap methods: (1) 'chain1', (2) 'chain2', (3) 'tree_uni1', (4) 'tree_uni2', (5) 'tree_bi1', (6) 'tree_bi2'.

**resample_n**
    int. Specifies the number of resamples.

Returns
-------

**pd.DataFrame**
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
        type='tree_uni1',
        resample_n=1000
    )

RDSBootOptimizedParallel - Parallel Bootstrap
==============================================

Parallelized Bootstrap Resampling for Respondent Driven Sampling (RDS). This function performs resampling RDS sample data by bootstrapping edges in recruitment trees or bootstrapping recruitment chains as a whole with parallel processing.

Combines:
1. Dictionary-based lookups for 1.2-1.6x speedup
2. Multi-core parallelization

Usage
-----

.. code-block:: python

    RDSBootOptimizedParallel(data, respondent_id_col, seed_id_col, seed_col, recruiter_id_col, type, resample_n, n_cores=2)

Arguments
---------

**data**
    pd.DataFrame. The input DataFrame containing RDS data.

**respondent_id_col**
    str. Name of the column containing respondent IDs - A variable indicating respondent ID.

**seed_id_col**
    str. Name of the column containing seed IDs - A variable indicating seed ID.

**seed_col**
    str. Name of the column containing seed indicators - A variable indicating whether a particular respondent is seed or not.

**recruiter_id_col**
    str. Name of the column containing recruiter IDs - A variable indicating recruiter ID.

**type**
    str. One of the six types of bootstrap methods: (1) 'chain1', (2) 'chain2', (3) 'tree_uni1', (4) 'tree_uni2', (5) 'tree_bi1', (6) 'tree_bi2'.

**resample_n**
    int. Specifies the number of resamples.

**n_cores**
    int, optional. Number of cores to use for parallel processing. If None, uses all available cores. Default is 2.

Returns
-------

**pd.DataFrame**
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
        type='tree_uni1',
        resample_n=1000,
        n_cores=4
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
        type='tree_uni1',
        resample_n=10000,  # Large number of resamples
        n_cores=8  # Use 8 cores for parallel processing
    )


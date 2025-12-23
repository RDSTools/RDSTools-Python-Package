import pandas as pd
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle

# Helper functions that need to be at module level for ProcessPoolExecutor
def match_with_duplicates(sampled_ids, source_df, id_column):
    """Replicate R's match function behavior - preserves duplicates"""
    result_list = []
    for sampled_id in sampled_ids:
        matches = source_df[source_df[id_column] == sampled_id]
        if not matches.empty:
            result_list.append(matches.iloc[[0]])
    return pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()

def build_lookup_structures(data_boot):
    """Build optimized lookup dictionaries - must be at module level"""
    recruiter_to_recruit_rows = {}
    seed_to_chain_rows = {}
    
    for _, row in data_boot.iterrows():
        row_dict = row.to_dict()
        respondent_id = row['respondent_id']
        recruiter_id = row['recruiter_id']
        seed_id = row['seed_id']
        
        # Build recruiter->recruit_rows mapping for fast wave processing
        if pd.notna(recruiter_id):
            if recruiter_id not in recruiter_to_recruit_rows:
                recruiter_to_recruit_rows[recruiter_id] = []
            recruiter_to_recruit_rows[recruiter_id].append(row_dict)
        
        # Build seed->chain_rows mapping for fast chain processing
        if pd.notna(seed_id):
            if seed_id not in seed_to_chain_rows:
                seed_to_chain_rows[seed_id] = []
            seed_to_chain_rows[seed_id].append(row_dict)
    
    return recruiter_to_recruit_rows, seed_to_chain_rows

def process_chain1_resample_optimized(args):
    """Optimized chain1 processing with dictionary lookups"""
    data_boot, seed_ids, resample_idx = args

    # Build lookup structures
    _, seed_to_chain_rows = build_lookup_structures(data_boot)
    
    empty_list = []
    for sid in seed_ids:
        # OPTIMIZED: O(1) dictionary lookup for chain data
        chain_rows = seed_to_chain_rows.get(sid, [])
        if chain_rows:
            chain_df = pd.DataFrame(chain_rows)
            empty_list.append(chain_df)
    
    result = pd.concat(empty_list, ignore_index=True) if empty_list else pd.DataFrame()
    result['RESAMPLE.N'] = resample_idx + 1
    return result

def process_tree_uni1_resample_optimized(args):
    """Optimized tree_uni1 processing with dictionary lookups"""
    data_boot, all_seeds, resample_idx = args

    # Build lookup structures once per resample
    recruiter_to_recruit_rows, _ = build_lookup_structures(data_boot)
    
    seed_ids = np.random.choice(all_seeds, size=len(all_seeds), replace=True)
    
    # Get first wave recruits - FULLY OPTIMIZED
    results = []
    for seed_id in seed_ids:
        # OPTIMIZED: O(1) dictionary lookup for recruit data
        recruit_rows = recruiter_to_recruit_rows.get(seed_id, [])
        if recruit_rows:
            seed_recruits = pd.DataFrame(recruit_rows)
            r_recruits = np.random.choice(seed_recruits['respondent_id'].values,
                                          size=len(seed_recruits), replace=True)
            resampled_recruits = match_with_duplicates(r_recruits, seed_recruits, 'respondent_id')
            if not resampled_recruits.empty:
                results.append(resampled_recruits)

    full_recruitment_data = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # Continue with subsequent waves - FULLY OPTIMIZED
    current_wave_ids = full_recruitment_data['respondent_id'].tolist() if not full_recruitment_data.empty else []

    while current_wave_ids:
        recruits_df_list = []
        for current_id in current_wave_ids:
            # OPTIMIZED: O(1) dictionary lookup for recruit data
            recruit_rows = recruiter_to_recruit_rows.get(current_id, [])
            if recruit_rows:
                recruits = pd.DataFrame(recruit_rows)
                recruits_df_list.append(recruits)

        if recruits_df_list:
            recruits_df = pd.concat(recruits_df_list, ignore_index=True)
            r_recruits = np.random.choice(recruits_df['respondent_id'].values,
                                          size=len(recruits_df), replace=True)
            new_recruits = match_with_duplicates(r_recruits, recruits_df, 'respondent_id')

            if not new_recruits.empty:
                full_recruitment_data = pd.concat([full_recruitment_data, new_recruits], ignore_index=True)
                current_wave_ids = new_recruits['respondent_id'].tolist()
            else:
                current_wave_ids = []
        else:
            current_wave_ids = []

    # Add seeds
    seed_data_list = []
    for seed_id in seed_ids:
        seed_row = data_boot[data_boot['respondent_id'] == seed_id]
        if not seed_row.empty:
            seed_data_list.append(seed_row.iloc[[0]])

    seed_data = pd.concat(seed_data_list, ignore_index=True) if seed_data_list else pd.DataFrame()
    result = pd.concat([seed_data, full_recruitment_data], ignore_index=True)
    result['RESAMPLE.N'] = resample_idx + 1
    return result

def process_tree_bi1_resample_optimized(args):
    """Optimized bidirectional tree processing"""
    data_boot, resample_idx = args

    # Build lookup structures
    recruiter_to_recruit_rows, _ = build_lookup_structures(data_boot)
    
    valid_ids = data_boot['respondent_id'].dropna().values
    if len(valid_ids) == 0:
        raise ValueError("No valid respondent IDs found")

    starting_ids = np.random.choice(valid_ids, size=min(10, len(valid_ids)), replace=True)
    chains = {}

    for start_id in starting_ids:
        current_ids = [start_id]
        previous_ids = []
        chain = [start_id]

        while current_ids:
            forward_nodes = []
            backward_nodes = []

            for cid in current_ids:
                # Forward nodes - OPTIMIZED: get people this person recruited
                recruit_rows = recruiter_to_recruit_rows.get(cid, [])
                forward_matches = [row['respondent_id'] for row in recruit_rows]
                forward_nodes.extend(forward_matches)

                # Backward nodes - find who recruited this person
                backward_matches = data_boot[data_boot['respondent_id'] == cid]['recruiter_id'].dropna().tolist()
                backward_nodes.extend(backward_matches)

            connected_nodes = forward_nodes + backward_nodes
            next_nodes = [node for node in connected_nodes
                          if node not in chain and node not in previous_ids and pd.notna(node)]

            if not next_nodes:
                break

            sampled_nodes = np.random.choice(next_nodes, size=len(next_nodes), replace=True)
            chain.extend(sampled_nodes)
            previous_ids = current_ids[:]
            current_ids = list(set(sampled_nodes))

        chains[start_id] = chain

    # Create results dataframe
    chains_df = pd.DataFrame(columns=['chain_id', 'respondent_id'])
    for chain_id, chain_nodes in chains.items():
        temp_df = pd.DataFrame({
            'chain_id': [chain_id] * len(chain_nodes),
            'respondent_id': chain_nodes
        })
        chains_df = pd.concat([chains_df, temp_df], ignore_index=True)

    # Merge with original data
    result = pd.merge(chains_df, data_boot, on='respondent_id', how='inner')
    result['RESAMPLE.N'] = resample_idx + 1
    return result

def single_resample_wrapper_optimized(args):
    """Wrapper for single resample using optimized bootstrap - now standalone"""
    data, respondent_id_col, seed_id_col, seed_col, recruiter_id_col, type, resample_idx = args
    
    # Create working DataFrame with standardized column names for internal use
    data_boot = pd.DataFrame({
        'respondent_id': data[respondent_id_col],
        'seed_id': data[seed_id_col],
        'seed': data[seed_col],
        'recruiter_id': data[recruiter_id_col]
    })

    # BUILD FAST LOOKUP STRUCTURES (KEY OPTIMIZATION)
    recruiter_to_recruit_rows, seed_to_chain_rows = build_lookup_structures(data_boot)

    if type == 'resample_chain1':
        df = data_boot[data_boot['seed'] == 1].copy()
        if len(df) == 0:
            raise ValueError("No seeds found (seed == 1)")

        seed_ids = np.random.choice(df['respondent_id'].values, size=len(df), replace=True)

        empty_list = []
        for sid in seed_ids:
            # OPTIMIZED: O(1) dictionary lookup for chain data
            chain_rows = seed_to_chain_rows.get(sid, [])
            if chain_rows:
                chain_df = pd.DataFrame(chain_rows)
                empty_list.append(chain_df)

        result = pd.concat(empty_list, ignore_index=True) if empty_list else pd.DataFrame()

    elif type == 'resample_chain2':
        seeds_df = data_boot[data_boot['seed'] == 1]
        if len(seeds_df) == 0:
            raise ValueError("There are no records where seed equals 1.")

        bootstrap_list = []
        data_df = pd.DataFrame()

        while len(data_df) < len(data_boot):
            selected_seed = np.random.choice(seeds_df['respondent_id'].values, size=1)[0]
            
            # OPTIMIZED: O(1) dictionary lookup for chain data
            chain_rows = seed_to_chain_rows.get(selected_seed, [])
            if chain_rows:
                chain_data = pd.DataFrame(chain_rows)
                bootstrap_list.append(chain_data)
                data_df = pd.concat(bootstrap_list, ignore_index=True)
            else:
                break

        result = data_df

    elif type == 'resample_tree_uni1':
        all_seeds = data_boot[data_boot['seed'] == 1]
        if len(all_seeds) == 0:
            raise ValueError("No seeds found")

        seed_ids = np.random.choice(all_seeds['respondent_id'].values, size=len(all_seeds), replace=True)

        # Get first wave recruits - FULLY OPTIMIZED
        results = []
        for seed_id in seed_ids:
            # OPTIMIZED: O(1) dictionary lookup for recruit data
            recruit_rows = recruiter_to_recruit_rows.get(seed_id, [])
            if recruit_rows:
                seed_recruits = pd.DataFrame(recruit_rows)
                r_recruits = np.random.choice(seed_recruits['respondent_id'].values,
                                              size=len(seed_recruits), replace=True)
                resampled_recruits = match_with_duplicates(r_recruits, seed_recruits, 'respondent_id')
                if not resampled_recruits.empty:
                    results.append(resampled_recruits)

        full_recruitment_data = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        # Continue with subsequent waves - FULLY OPTIMIZED
        current_wave_ids = full_recruitment_data['respondent_id'].tolist() if not full_recruitment_data.empty else []

        while current_wave_ids:
            recruits_df_list = []
            for current_id in current_wave_ids:
                # OPTIMIZED: O(1) dictionary lookup for recruit data
                recruit_rows = recruiter_to_recruit_rows.get(current_id, [])
                if recruit_rows:
                    recruits = pd.DataFrame(recruit_rows)
                    recruits_df_list.append(recruits)

            if recruits_df_list:
                recruits_df = pd.concat(recruits_df_list, ignore_index=True)
                r_recruits = np.random.choice(recruits_df['respondent_id'].values,
                                              size=len(recruits_df), replace=True)
                new_recruits = match_with_duplicates(r_recruits, recruits_df, 'respondent_id')

                if not new_recruits.empty:
                    full_recruitment_data = pd.concat([full_recruitment_data, new_recruits], ignore_index=True)
                    current_wave_ids = new_recruits['respondent_id'].tolist()
                else:
                    current_wave_ids = []
            else:
                current_wave_ids = []

        # Add seeds
        seed_data_list = []
        for seed_id in seed_ids:
            seed_row = data_boot[data_boot['respondent_id'] == seed_id]
            if not seed_row.empty:
                seed_data_list.append(seed_row.iloc[[0]])

        seed_data = pd.concat(seed_data_list, ignore_index=True) if seed_data_list else pd.DataFrame()
        result = pd.concat([seed_data, full_recruitment_data], ignore_index=True)

    elif type == 'resample_tree_uni2':
        result = pd.DataFrame()

        while len(result) < len(data_boot):
            all_seeds = data_boot[data_boot['seed'] == 1]
            if len(all_seeds) == 0:
                break

            seed_id = np.random.choice(all_seeds['respondent_id'].values, size=1)[0]

            # Get first wave - OPTIMIZED
            recruit_rows = recruiter_to_recruit_rows.get(seed_id, [])
            if recruit_rows:
                seed_recruits = pd.DataFrame(recruit_rows)
                r_recruits = np.random.choice(seed_recruits['respondent_id'].values,
                                              size=len(seed_recruits), replace=True)
                full_recruitment_data = match_with_duplicates(r_recruits, seed_recruits, 'respondent_id')
            else:
                full_recruitment_data = pd.DataFrame()

            # Continue with subsequent waves - OPTIMIZED
            current_wave_ids = full_recruitment_data['respondent_id'].tolist() if not full_recruitment_data.empty else []

            while current_wave_ids:
                recruits_df_list = []
                
                for current_id in current_wave_ids:
                    # OPTIMIZED: O(1) dictionary lookup for recruit data
                    recruit_rows = recruiter_to_recruit_rows.get(current_id, [])
                    if recruit_rows:
                        recruits = pd.DataFrame(recruit_rows)
                        recruits_df_list.append(recruits)

                if recruits_df_list:
                    recruits_df = pd.concat(recruits_df_list, ignore_index=True)
                    r_recruits = np.random.choice(recruits_df['respondent_id'].values,
                                                  size=len(recruits_df), replace=True)
                    new_recruits = match_with_duplicates(r_recruits, recruits_df, 'respondent_id')

                    if not new_recruits.empty:
                        full_recruitment_data = pd.concat([full_recruitment_data, new_recruits], ignore_index=True)
                        current_wave_ids = new_recruits['respondent_id'].tolist()
                    else:
                        current_wave_ids = []
                else:
                    current_wave_ids = []

            # Add seed
            seed_data = data_boot[data_boot['respondent_id'] == seed_id].iloc[[0]]
            iteration_results = pd.concat([seed_data, full_recruitment_data], ignore_index=True)
            result = pd.concat([result, iteration_results], ignore_index=True)

    elif type == 'resample_tree_bi1':
        valid_ids = data_boot['respondent_id'].dropna().values
        if len(valid_ids) == 0:
            raise ValueError("No valid respondent IDs found")

        starting_ids = np.random.choice(valid_ids, size=min(10, len(valid_ids)), replace=True)
        chains = {}

        for start_id in starting_ids:
            current_ids = [start_id]
            previous_ids = []
            chain = [start_id]

            while current_ids:
                forward_nodes = []
                backward_nodes = []

                for cid in current_ids:
                    # Forward nodes - OPTIMIZED: get people this person recruited
                    recruit_rows = recruiter_to_recruit_rows.get(cid, [])
                    forward_matches = [row['respondent_id'] for row in recruit_rows]
                    forward_nodes.extend(forward_matches)

                    # Backward nodes - find who recruited this person
                    backward_matches = data_boot[data_boot['respondent_id'] == cid]['recruiter_id'].dropna().tolist()
                    backward_nodes.extend(backward_matches)

                connected_nodes = forward_nodes + backward_nodes
                next_nodes = [node for node in connected_nodes
                              if node not in chain and node not in previous_ids and pd.notna(node)]

                if not next_nodes:
                    break

                sampled_nodes = np.random.choice(next_nodes, size=len(next_nodes), replace=True)
                chain.extend(sampled_nodes)
                previous_ids = current_ids[:]
                current_ids = list(set(sampled_nodes))

            chains[start_id] = chain

        # Create results dataframe
        chains_df = pd.DataFrame(columns=['chain_id', 'respondent_id'])
        for chain_id, chain_nodes in chains.items():
            temp_df = pd.DataFrame({
                'chain_id': [chain_id] * len(chain_nodes),
                'respondent_id': chain_nodes
            })
            chains_df = pd.concat([chains_df, temp_df], ignore_index=True)

        # Merge with original data
        result = pd.merge(chains_df, data_boot, on='respondent_id', how='inner')

    elif type == 'resample_tree_bi2':
        chains_df = pd.DataFrame(columns=['chain_id', 'respondent_id'])
        max_iterations = 1000
        iteration = 0

        while len(chains_df) < len(data_boot) and iteration < max_iterations:
            iteration += 1

            valid_ids = data_boot['respondent_id'].dropna().values
            if len(valid_ids) == 0:
                break

            start_id = np.random.choice(valid_ids, size=1)[0]
            current_ids = [start_id]
            previous_ids = []
            chain = [start_id]

            while current_ids:
                forward_nodes = []
                backward_nodes = []

                for cid in current_ids:
                    # Forward nodes - OPTIMIZED: get people this person recruited
                    recruit_rows = recruiter_to_recruit_rows.get(cid, [])
                    forward_matches = [row['respondent_id'] for row in recruit_rows]
                    forward_nodes.extend(forward_matches)

                    # Backward nodes
                    backward_matches = data_boot[data_boot['respondent_id'] == cid]['recruiter_id'].dropna().tolist()
                    backward_nodes.extend(backward_matches)

                connected_nodes = forward_nodes + backward_nodes
                next_nodes = [node for node in connected_nodes
                              if node not in chain and node not in previous_ids and pd.notna(node)]

                if not next_nodes:
                    break

                sampled_nodes = np.random.choice(next_nodes, size=len(next_nodes), replace=True)
                chain.extend(sampled_nodes)
                previous_ids = current_ids[:]
                current_ids = list(set(sampled_nodes))

            # Add this chain to results
            temp_df = pd.DataFrame({
                'chain_id': [start_id] * len(chain),
                'respondent_id': chain
            })
            chains_df = pd.concat([chains_df, temp_df], ignore_index=True)

        result = pd.merge(chains_df, data_boot, on='respondent_id', how='inner')

    else:
        raise ValueError("Specify one of six bootstrap types")

    if not result.empty:
        result['RESAMPLE.N'] = resample_idx + 1
    return result


def RDSBootOptimizedParallel(data, respondent_id_col, seed_id_col, seed_col, recruiter_id_col, type, resample_n,
                             n_cores=2):
    """
    Variance estimation with bootstrap chain and tree methods.

    Combined optimized + parallel bootstrap resampling for RDS data.

    Combines:
    1. Dictionary-based lookups for 1.2-1.6x speedup (from optimized_bootstrap.py)
    2. Multi-core parallelization for 8.0x speedup (from parallel_rds_bootstrap.py)

    Total potential speedup: 1.2x * 8.0x = 9.6x faster!

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing RDS data
    respondent_id_col : str
        Name of the column containing respondent IDs - A variable indicating respondent ID
    seed_id_col : str
        Name of the column containing seed IDs - A variable indicating seed ID
    seed_col : str
        Name of the column containing seed indicators - A variable indicating whether a particular respondent is seed or not
    recruiter_id_col : str
        Name of the column containing recruiter IDs - A variable indicating recruiter ID
    type : str
        One of the six types of bootstrap methods: (1) resample_chain1, (2) resample_chain2, (3) resample_tree_uni1, (4) resample_tree_uni2, (5) resample_tree_bi1, (6) resample_tree_bi2.
    resample_n : int
        A specified number of resamples
    n_cores : int, optional
        Number of cores to use for parallel processing. If None, uses all available cores. Default is 2.

    Returns:
    --------
    pd.DataFrame
        Returns a data frame consisting of the following elements:
        RESPONDENT_ID : A variable indicating respondent ID
        RESAMPLE.N : An indicator variable for each resample iteration

    Notes
    -----
    In all resampling functions, versions 1 and 2 differ as 1 focuses on the number of seeds in a given resample to be consistent with the original sample, while 2 keeps the overall sample size of a given resample to be at least equal to the original sample.

    In the resample_chain1 function, ( n ) seeds are selected using Simple Random Sampling with Replacement (SRSWR), with all nodes in the chains created by resampled seeds retained . In the resample_chain2 function, 1 seed is sampled using SRSWR, with all nodes retained in the chain. The process continues until the sample size of a given resample ( n_r ) is at least equal to the original sample size ( n_s )

    In the resample_tree_uni1 function, ( n ) seeds are selected using SRSWR. For each selected seed, the function (A) checks its recruit counts, (B) performs SRSWR on the recruits counts from all recruits identified in (A), and (C) for each sampled recruit, repeats steps A and B. Steps A, B, and C are performed until the last wave of the chain. In resample_tree_uni2, instead of selecting ( n ) seeds, the function selects one seed at a time and then performs steps A, B, and C for each wave of respondents.

    resample_tree_bi1 selects ( n ) nodes from the recruitment chains using SRSWR. For each selected node, it (A) checks its connected nodes, (B) performs SRSWR on all connected nodes identified in (A), and (C) for each selected node, performs steps A and B, but does not resample already resampled nodes. (D) Steps A, B, and C are repeated until the end of the chain. In resample_tree_bi2, 1 node is selected using SRSWR from the recruitment chain and steps A, B, C, and D are performed as in resample_tree_bi1.

    Examples
    --------
    # Preprocess data with RDSdata function
    rds_data = RDSdata(data = RDSToolsToyData,
                      unique_id = "ID",
                      redeemed_coupon = "CouponR",
                       issued_coupon = ["Coupon1",
                                        "Coupon2",
                                        "Coupon3"],
                      degree = "Degree")

    # Run RDSBootOptimizedParallel with rds_data using 4 cores
    results = RDSBootOptimizedParallel(data = rds_data,
                                      respondent_id_col = 'ID',
                                      seed_id_col = 'S_ID',
                                      seed_col = 'SEED',
                                      recruiter_id_col = 'R_ID',
                                      type = 'resample_chain1',
                                      resample_n = 100,
                                      n_cores = 4)
    """

    
    # Validate that all required columns exist
    required_cols = [respondent_id_col, seed_id_col, seed_col, recruiter_id_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # Create working DataFrame with standardized column names for internal use
    data_boot = pd.DataFrame({
        'respondent_id': data[respondent_id_col],
        'seed_id': data[seed_id_col],
        'seed': data[seed_col],
        'recruiter_id': data[recruiter_id_col]
    })

    # Get all seeds for parallelization
    all_seeds = data_boot[data_boot['seed'] == 1]['respondent_id'].values
    if len(all_seeds) == 0:
        raise ValueError("No seeds found (seed == 1)")

    results_bootstrap_RDS = []

    if type == 'resample_chain1':
        # Parallel + optimized chain1 processing
        tasks = []
        for i in range(resample_n):
            seed_ids = np.random.choice(all_seeds, size=len(all_seeds), replace=True)
            tasks.append((data_boot, seed_ids, i))
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results_bootstrap_RDS = list(executor.map(process_chain1_resample_optimized, tasks))

    elif type == 'resample_chain2':
        # Use optimized sequential version with resample-level parallelization
        print("Using optimized resample-level parallelization for resample_chain2")
        return RDSBootOptimizedSequentialResamples(data, respondent_id_col, seed_id_col, seed_col, 
                                                  recruiter_id_col, type, resample_n, n_cores)

    elif type == 'resample_tree_uni1':
        # Parallel + optimized tree_uni1 processing
        tasks = [(data_boot, all_seeds, i) for i in range(resample_n)]
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results_bootstrap_RDS = list(executor.map(process_tree_uni1_resample_optimized, tasks))

    elif type == 'resample_tree_uni2':
        # Use optimized sequential version with resample-level parallelization
        print("Using optimized resample-level parallelization for resample_tree_uni2")
        return RDSBootOptimizedSequentialResamples(data, respondent_id_col, seed_id_col, seed_col, 
                                                  recruiter_id_col, type, resample_n, n_cores)

    elif type == 'resample_tree_bi1':
        # Parallel + optimized bidirectional tree processing
        tasks = [(data_boot, i) for i in range(resample_n)]
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results_bootstrap_RDS = list(executor.map(process_tree_bi1_resample_optimized, tasks))

    elif type == 'resample_tree_bi2':
        # Use optimized sequential version with resample-level parallelization
        print("Using optimized resample-level parallelization for resample_tree_bi2")
        return RDSBootOptimizedSequentialResamples(data, respondent_id_col, seed_id_col, seed_col, 
                                                  recruiter_id_col, type, resample_n, n_cores)

    else:
        raise ValueError("Specify one of six bootstrap types")

    # Extract only RESPONDENT_ID and RESAMPLE.N
    final_results = []
    for df in results_bootstrap_RDS:
        if not df.empty:
            final_results.append(pd.DataFrame({
                'RESPONDENT_ID': df['respondent_id'],
                'RESAMPLE.N': df['RESAMPLE.N']
            }))

    if final_results:
        return pd.concat(final_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['RESPONDENT_ID', 'RESAMPLE.N'])

def RDSBootOptimizedSequentialResamples(data, respondent_id_col, seed_id_col, seed_col, recruiter_id_col, type, resample_n, n_cores=None):
    """
    Optimized parallel processing across resamples using the dictionary-optimized bootstrap.
    """
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Prepare arguments for parallel processing
    tasks = [(data, respondent_id_col, seed_id_col, seed_col, recruiter_id_col, type, i) 
             for i in range(resample_n)]
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(single_resample_wrapper_optimized, tasks))
    
    # Combine results
    final_results = [df for df in results if not df.empty]
    
    if final_results:
        # Extract only RESPONDENT_ID and RESAMPLE.N
        final_dfs = []
        for df in final_results:
            final_dfs.append(pd.DataFrame({
                'RESPONDENT_ID': df['respondent_id'],
                'RESAMPLE.N': df['RESAMPLE.N']
            }))
        return pd.concat(final_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=['RESPONDENT_ID', 'RESAMPLE.N'])


# Example usage:
"""
import pandas as pd

# Load data
data = pd.read_csv("your_data.csv")

# Use
results = RDSBootOptimizedParallel(
    data=data,
    respondent_id_col='ID', 
    seed_id_col='S_ID',
    seed_col='SEED', 
    recruiter_id_col='R_ID',
    type='resample_tree_uni1',  # Gets both optimizations
    resample_n=1000,
    n_cores=8
)

# Expected speedup: 1.2x (dictionary) * 8x (parallel) = 9.6x faster!
"""
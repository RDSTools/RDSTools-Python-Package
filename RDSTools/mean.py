import pandas as pd
import numpy as np
import math


class RDSResult(pd.DataFrame):
    """Custom DataFrame subclass that displays RDS results in a formatted way"""

    _metadata = ['_mean_val', '_se_val', '_se_method', '_n_original', '_is_weighted',
                 '_bootstrap_means', '_node_counts', '_n_iterations', '_mean_nodes',
                 '_min_nodes', '_q1_nodes', '_median_nodes', '_q3_nodes', '_max_nodes']

    def __init__(self, mean_x, se, method, n_original, is_weighted=False,
                 bootstrap_means=None, node_counts=None, n_iterations=None):

        # Store calculated values as attributes
        object.__setattr__(self, '_mean_val', round(mean_x, 3))
        object.__setattr__(self, '_se_val', round(se, 3))
        object.__setattr__(self, '_se_method', method)
        object.__setattr__(self, '_n_original', n_original)
        object.__setattr__(self, '_is_weighted', is_weighted)
        object.__setattr__(self, '_bootstrap_means', bootstrap_means or [])
        object.__setattr__(self, '_node_counts', node_counts or [])
        object.__setattr__(self, '_n_iterations', n_iterations)

        # Calculate node statistics if bootstrap data exists
        if node_counts:
            object.__setattr__(self, '_mean_nodes', round(np.mean(node_counts), 1))
            object.__setattr__(self, '_min_nodes', int(np.min(node_counts)))
            object.__setattr__(self, '_q1_nodes', int(np.percentile(node_counts, 25)))
            object.__setattr__(self, '_median_nodes', int(np.median(node_counts)))
            object.__setattr__(self, '_q3_nodes', int(np.percentile(node_counts, 75)))
            object.__setattr__(self, '_max_nodes', int(np.max(node_counts)))
        else:
            object.__setattr__(self, '_mean_nodes', None)
            object.__setattr__(self, '_min_nodes', None)
            object.__setattr__(self, '_q1_nodes', None)
            object.__setattr__(self, '_median_nodes', None)
            object.__setattr__(self, '_q3_nodes', None)
            object.__setattr__(self, '_max_nodes', None)

        # Create DataFrame for compatibility (minimal structure)
        mean_label = f"Weighted mean" if is_weighted else f"Mean"

        if n_iterations is not None:
            # Bootstrap case
            data = {
                'Type': [mean_label, 'SE', 'SE_Method', 'n_original', 'n_iterations',
                         'mean_nodes', 'min_nodes', 'q1_nodes', 'median_nodes', 'q3_nodes', 'max_nodes'],
                'Value': [self._mean_val, self._se_val, self._se_method, self._n_original, self._n_iterations,
                          self._mean_nodes, self._min_nodes, self._q1_nodes, self._median_nodes,
                          self._q3_nodes, self._max_nodes]
            }
        else:
            # Non-bootstrap case
            data = {
                'Type': [mean_label, 'SE', 'SE_Method', 'n_original'],
                'Value': [self._mean_val, self._se_val, self._se_method, self._n_original]
            }

        super().__init__(data)

    @property
    def bootstrap_means(self):
        return getattr(self, '_bootstrap_means', [])

    @property
    def node_counts(self):
        return getattr(self, '_node_counts', [])

    def __repr__(self):
        return self._format_output()

    def __str__(self):
        return self._format_output()

    def _format_output(self):
        lines = []

        # Determine weight text
        weight_text = "Weighted" if self._is_weighted else "Not weighted"

        # Basic output
        lines.append(f"Mean                    {self._mean_val}")
        lines.append(f"SE                       {self._se_val}")
        lines.append("")
        lines.append(f"n_Data                  {self._n_original}")
        lines.append(f"Weight                  {weight_text}")
        lines.append(f"SE Method               {self._se_method}")

        # Bootstrap summary if available
        if self._n_iterations is not None and self._node_counts:
            lines.append("")
            lines.append("— Resample Summary —")
            lines.append(f"n_Iteration     {self._n_iterations}")
            lines.append("n_Resample      Mean    SD")

            node_sd = round(np.std(self._node_counts), 2) if self._node_counts else 'NA'
            lines.append(f"                {self._mean_nodes:.1f}   {node_sd}")
            lines.append("                Min     1Q      Med     3Q      Max")
            lines.append(
                f"                 {self._min_nodes}     {self._q1_nodes}     {self._median_nodes}     {self._q3_nodes}     {self._max_nodes}")

        return "\n".join(lines)

    @property
    def _constructor(self):
        """Ensure that pandas operations return RDSResult objects"""
        return RDSResult


def RDSmean(x, data, weight=None, var_est=None, resample_n=None, n_cores=None, return_bootstrap_means=False,
            return_node_counts=False):
    """
    Descriptive statistics in an RDS study

    Calculate means and standard errors for RDS data with optional weighting
    and different variance estimation methods.

    Parameters:
    -----------
    x : str
        A variable of interest
    data : pandas.DataFrame
        The output DataFrame from RDSdata
    weight : str, optional
        Name of the weight variable.
        User specified weights to calculate weighted mean and standard errors.
        When set to None the function calculates unweighted mean and standard errors
    var_est : str, optional
        One of the six bootstrap types or the delta (naive) method.
        By default the function calculates naive standard errors.
        Variance estimation options include 'naive' or bootstrap methods like 'resample_chain1', 'resample_chain2', 'resample_tree_uni1', 'resample_tree_uni2',
        'resample_tree_bi1', 'resample_tree_bi2'
    resample_n : int, optional
        Specifies the number of resample iterations.
        Note that this argument is None when var_est = 'naive'.
        Required for bootstrap methods, default 300
    n_cores : int, optional
        Number of CPU cores to use for parallel bootstrap processing.
        If specified, uses optimized parallel bootstrap. If None, uses
        standard sequential bootstrap.
    return_bootstrap_means : bool, optional
        If True, return bootstrap mean estimates along with main results (only for bootstrap methods)
    return_node_counts : bool, optional
        If True, return node counts per iteration along with main results (only for bootstrap methods)

    Returns:
    --------
    RDSResult : RDSResult object (DataFrame subclass)
        When return_bootstrap_means and return_node_counts are both False (default).
        DataFrame with the following elements; weighted or unweighted mean with standard errors,
        additional information about the analysis: (1) var_est method, (2) weighted or not, (3) n_Data, (4) n_Analysis, (5) n_Iteration if var_est is not naive.
        descriptive summary of resamples if var_est is not naive, resample estimates
        Contains columns 'Type' and 'Value' with summary statistics.

    tuple : (RDSResult, list)
        When return_bootstrap_means is True and return_node_counts is False.
        Returns (formatted_result, bootstrap_means_list).

    tuple : (RDSResult, list)
        When return_bootstrap_means is False and return_node_counts is True.
        Returns (formatted_result, node_counts_list).

    tuple : (RDSResult, list, list)
        When both return_bootstrap_means and return_node_counts are True.
        Returns (formatted_result, bootstrap_means_list, node_counts_list).

    Notes
    -----
    The RDSResult object is a pandas DataFrame subclass that:
    - Retains all DataFrame functionality for analysis
    - Can be used with other pandas operations and statistical functions
    - Contains the same data structure as the original DataFrame implementation

    For bootstrap methods, the DataFrame includes additional rows with bootstrap statistics:
    - n_iterations: number of bootstrap resamples performed
    - mean_nodes: average number of nodes (observations) across all bootstrap samples
    - min_nodes: minimum number of nodes in any bootstrap sample
    - q1_nodes: 25th percentile of nodes across bootstrap samples
    - median_nodes: median number of nodes across bootstrap samples
    - q3_nodes: 75th percentile of nodes across bootstrap samples
    - max_nodes: maximum number of nodes in any bootstrap sample
    """

    # CPU core count check
    if n_cores is not None:
        if not isinstance(n_cores, int) or n_cores < 1:
            raise ValueError("n_cores must be a positive integer")

    # Valid bootstrap methods
    resample_methods = ['resample_chain1', 'resample_chain2',
                        'resample_tree_uni1', 'resample_tree_uni2',
                        'resample_tree_bi1', 'resample_tree_bi2']

    # Default resample_n if bootstrap method is specified but resample_n is not
    if resample_n is None and var_est in resample_methods:
        resample_n = 300

    # Check if bootstrap is specified but not a valid method
    if resample_n is not None and var_est not in resample_methods:
        raise ValueError("resample.n argument should only be applied when var.est is a bootstrap method.")

    # Store original row count
    n_original = len(data)

    # 1. Unweighted with naive variance estimation
    if weight is None and (var_est == 'naive' or var_est is None):
        # Convert to numeric and drop NAs
        x_values = pd.to_numeric(data[x], errors='coerce').dropna()
        n_analysis = len(x_values)

        if n_analysis == 0:
            raise ValueError(f"No valid numeric values found for variable '{x}'")

        mean_x = x_values.mean()

        # Calculate sample variance and standard error
        s_squared = ((x_values - mean_x) ** 2).sum() / (len(x_values) - 1) if len(x_values) > 1 else 0
        se = math.sqrt(s_squared / len(x_values)) if len(x_values) > 0 else float('nan')

        method = "Naive" if var_est is None else var_est
        result = RDSResult(mean_x, se, method, n_original, is_weighted=False)

        if return_bootstrap_means:
            if return_node_counts:
                return result, [], []
            return result, []
        if return_node_counts:
            return result, []
        return result

    # 2. Weighted with naive variance estimation
    elif weight is not None and (var_est == 'naive' or var_est is None):
        # Convert to numeric and drop NAs
        x_values = pd.to_numeric(data[x], errors='coerce')
        weight_values = pd.to_numeric(data[weight], errors='coerce')

        # Remove NA values in either variable
        valid_mask = ~(x_values.isna() | weight_values.isna())
        x_values = x_values[valid_mask]
        weight_values = weight_values[valid_mask]

        n_analysis = len(x_values)

        if n_analysis == 0:
            raise ValueError(f"No valid numeric values found for variable '{x}' with weight '{weight}'")

        if weight_values.sum() == 0:
            raise ValueError(f"Sum of weights is zero")

        x_bar = (x_values * weight_values).sum() / weight_values.sum()
        w_bar = weight_values.mean()

        # Calculate weighted variance
        if n_analysis > 1:
            weight_var_term1 = ((weight_values * x_values - w_bar * x_bar) ** 2).sum()
            weight_var_term2 = 2 * x_bar * ((weight_values - w_bar) * (weight_values * x_values - w_bar * x_bar)).sum()
            weight_var_term3 = x_bar ** 2 * ((weight_values - w_bar) ** 2).sum()

            w_var = n_analysis / ((n_analysis - 1) * (weight_values.sum() ** 2)) * (
                    weight_var_term1 - weight_var_term2 + weight_var_term3
            )
            se = math.sqrt(w_var)
        else:
            se = float('nan')

        method = "Naive" if var_est is None else var_est
        result = RDSResult(x_bar, se, method, n_original, is_weighted=True)

        if return_bootstrap_means:
            if return_node_counts:
                return result, [], []
            return result, []
        if return_node_counts:
            return result, []
        return result

    # 3. Unweighted with bootstrap variance estimation
    elif weight is None and var_est in resample_methods:

        if n_cores is not None:
            # Use optimized parallel bootstrap
            from parallel_bootstrap import RDSBootOptimizedParallel
            boot_out = RDSBootOptimizedParallel(
                data=data,
                respondent_id_col='ID',
                seed_id_col='S_ID',
                seed_col='SEED',
                recruiter_id_col='R_ID',
                type=var_est,
                resample_n=resample_n,
                n_cores=n_cores
            )
        else:
            # Use original bootstrap
            from bootstrap import RDSboot
            boot_out = RDSboot(
                data=data,
                respondent_id_col='ID',
                seed_id_col='S_ID',
                seed_col='SEED',
                recruiter_id_col='R_ID',
                type=var_est,
                resample_n=resample_n
            )

        # Create a copy of data with ID renamed to RESPONDENT_ID for merging
        merge_data = data.copy()
        merge_data = merge_data.rename(columns={'ID': 'RESPONDENT_ID'})

        # Merge bootstrapped data with original data
        merged_data = pd.merge(merge_data, boot_out, on='RESPONDENT_ID')
        n_analysis = len(merged_data)

        if n_analysis == 0:
            raise ValueError(f"No data after merging with bootstrap results")

        # Calculate mean for each bootstrap sample
        bootstrap_means = []
        node_counts = []

        for resample_id in merged_data['RESAMPLE.N'].unique():
            group = merged_data[merged_data['RESAMPLE.N'] == resample_id]
            # Convert to numeric before calculating mean
            numeric_values = pd.to_numeric(group[x], errors='coerce').dropna()
            if len(numeric_values) > 0:
                bootstrap_means.append(numeric_values.mean())
                node_counts.append(len(group))

        # Convert to numeric before calculating mean on original data
        x_values_orig = pd.to_numeric(data[x], errors='coerce').dropna()

        if len(x_values_orig) == 0:
            raise ValueError(f"No valid numeric values found for variable '{x}'")

        mean_x = x_values_orig.mean()
        se_bootstrap = math.sqrt(np.var(bootstrap_means)) if len(bootstrap_means) > 1 else float('nan')

        result = RDSResult(mean_x, se_bootstrap, var_est, n_original,
                           is_weighted=False, bootstrap_means=bootstrap_means,
                           node_counts=node_counts, n_iterations=resample_n)

        if return_bootstrap_means:
            if return_node_counts:
                return result, bootstrap_means, node_counts
            return result, bootstrap_means
        if return_node_counts:
            return result, node_counts
        return result

    # 4. Weighted with bootstrap variance estimation
    else:
        if n_cores is not None:
            # Use optimized parallel bootstrap
            from parallel_bootstrap import (RDSBootOptimizedParallel)
            boot_out = RDSBootOptimizedParallel(
                data=data,
                respondent_id_col='ID',
                seed_id_col='S_ID',
                seed_col='SEED',
                recruiter_id_col='R_ID',
                type=var_est,
                resample_n=resample_n,
                n_cores=n_cores
            )
        else:
            # Use original bootstrap
            from bootstrap import RDSboot
            boot_out = RDSboot(
                data=data,
                respondent_id_col='ID',
                seed_id_col='S_ID',
                seed_col='SEED',
                recruiter_id_col='R_ID',
                type=var_est,
                resample_n=resample_n
            )

        # Create a copy of data with ID renamed to RESPONDENT_ID for merging
        merge_data = data.copy()
        merge_data = merge_data.rename(columns={'ID': 'RESPONDENT_ID'})

        # Merge bootstrapped data with original data
        merged_data = pd.merge(merge_data, boot_out, on='RESPONDENT_ID')
        n_analysis = len(merged_data)

        if n_analysis == 0:
            raise ValueError(f"No data after merging with bootstrap results")

        # Calculate weighted mean for each bootstrap sample
        bootstrap_means = []
        node_counts = []
        for resample_id in merged_data['RESAMPLE.N'].unique():
            group = merged_data[merged_data['RESAMPLE.N'] == resample_id]
            # Convert to numeric before calculating weighted mean
            x_values_group = pd.to_numeric(group[x], errors='coerce')
            weight_values_group = pd.to_numeric(group[weight], errors='coerce')

            # Filter out NaN values
            valid_mask = ~(x_values_group.isna() | weight_values_group.isna())
            x_values_group = x_values_group[valid_mask]
            weight_values_group = weight_values_group[valid_mask]

            if len(weight_values_group) > 0 and weight_values_group.sum() > 0:
                weighted_mean = (x_values_group * weight_values_group).sum() / weight_values_group.sum()
                bootstrap_means.append(weighted_mean)
                node_counts.append(len(group))

        # Calculate weighted mean on original data (ensuring numeric types)
        x_values_orig = pd.to_numeric(data[x], errors='coerce')
        weight_values_orig = pd.to_numeric(data[weight], errors='coerce')

        # Filter out NaN values
        valid_mask = ~(x_values_orig.isna() | weight_values_orig.isna())
        x_values_orig = x_values_orig[valid_mask]
        weight_values_orig = weight_values_orig[valid_mask]

        if len(x_values_orig) == 0:
            raise ValueError(f"No valid numeric values found for variable '{x}' with weight '{weight}'")

        if weight_values_orig.sum() == 0:
            raise ValueError(f"Sum of weights is zero")

        mean_x = (x_values_orig * weight_values_orig).sum() / weight_values_orig.sum()
        se_bootstrap = math.sqrt(np.var(bootstrap_means)) if len(bootstrap_means) > 1 else float('nan')

        result = RDSResult(mean_x, se_bootstrap, var_est, n_original,
                           is_weighted=True, bootstrap_means=bootstrap_means,
                           node_counts=node_counts, n_iterations=resample_n)

        if return_bootstrap_means:
            if return_node_counts:
                return result, bootstrap_means, node_counts
            return result, bootstrap_means
        if return_node_counts:
            return result, node_counts
        return result
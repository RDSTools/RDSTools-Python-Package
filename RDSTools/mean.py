import pandas as pd
import numpy as np
import math


def _is_categorical_series(s: pd.Series) -> bool:
    """Detect whether a column should be treated as categorical.

    Mirrors the R behaviour: character/factor columns are categorical,
    numeric (int/float) columns are continuous. Booleans and pandas
    Categorical/string dtypes are treated as categorical.
    """
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return False
    return True


def _coerce_categorical(s: pd.Series) -> pd.Categorical:
    """Return a pandas Categorical with a stable, sorted level order.

    NaN values are preserved (not assigned a category) so they can be
    handled by na_rm logic downstream.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        # Already a Categorical Series — extract the underlying Categorical
        # and reorder its levels for stable, sorted output.
        existing_levels = sorted(s.cat.categories.tolist(), key=lambda v: str(v))
        return pd.Categorical(s, categories=existing_levels)
    # Build sorted levels from non-null values
    levels = sorted(s.dropna().unique().tolist(), key=lambda v: str(v))
    return pd.Categorical(s, categories=levels)


class RDSResult(pd.DataFrame):
    """Custom DataFrame subclass that displays RDS mean results in a formatted way.

    Holds a tidy ``results`` table (one row per category for categorical
    variables, one row for continuous variables) along with estimation
    metadata such as the variance estimation method, weighting status,
    sample sizes, and (for bootstrap methods) per-iteration estimates.
    """

    _metadata = [
        '_results_df', '_se_method', '_n_original', '_n_analysis',
        '_is_weighted', '_is_categorical', '_levels',
        '_bootstrap_estimates', '_node_counts', '_n_iterations',
        '_mean_nodes', '_min_nodes', '_q1_nodes', '_median_nodes',
        '_q3_nodes', '_max_nodes',
    ]

    def __init__(self, means, ses, method, n_original, n_analysis,
                 is_weighted=False, is_categorical=False, levels=None,
                 bootstrap_estimates=None, node_counts=None, n_iterations=None):

        # Normalise scalars to arrays for uniform handling
        means_arr = np.atleast_1d(np.asarray(means, dtype=float))
        ses_arr = np.atleast_1d(np.asarray(ses, dtype=float))

        # Build the tidy results table
        if is_categorical:
            cats = list(levels) if levels is not None else [f"Level {i}" for i in range(len(means_arr))]
            results_df = pd.DataFrame({
                'Category': cats,
                'Mean': np.round(means_arr, 3),
                'SE': np.round(ses_arr, 3),
            })
        else:
            results_df = pd.DataFrame({
                'Mean': np.round(means_arr, 3),
                'SE': np.round(ses_arr, 3),
            })

        object.__setattr__(self, '_results_df', results_df)
        object.__setattr__(self, '_se_method', method)
        object.__setattr__(self, '_n_original', n_original)
        object.__setattr__(self, '_n_analysis', n_analysis)
        object.__setattr__(self, '_is_weighted', is_weighted)
        object.__setattr__(self, '_is_categorical', is_categorical)
        object.__setattr__(self, '_levels', list(levels) if levels is not None else None)
        object.__setattr__(self, '_bootstrap_estimates', bootstrap_estimates if bootstrap_estimates is not None else [])
        object.__setattr__(self, '_node_counts', node_counts if node_counts is not None else [])
        object.__setattr__(self, '_n_iterations', n_iterations)

        # Resample-size summary statistics
        if node_counts:
            object.__setattr__(self, '_mean_nodes', round(float(np.mean(node_counts)), 1))
            object.__setattr__(self, '_min_nodes', int(np.min(node_counts)))
            object.__setattr__(self, '_q1_nodes', int(np.percentile(node_counts, 25)))
            object.__setattr__(self, '_median_nodes', int(np.median(node_counts)))
            object.__setattr__(self, '_q3_nodes', int(np.percentile(node_counts, 75)))
            object.__setattr__(self, '_max_nodes', int(np.max(node_counts)))
        else:
            for attr in ('_mean_nodes', '_min_nodes', '_q1_nodes',
                         '_median_nodes', '_q3_nodes', '_max_nodes'):
                object.__setattr__(self, attr, None)

        # Initialise the underlying DataFrame with the tidy results table
        super().__init__(results_df)

    # ---- Public read-only accessors -----------------------------------------
    @property
    def results(self):
        """The tidy results table (Category/Mean/SE or Mean/SE)."""
        return self._results_df

    @property
    def bootstrap_estimates(self):
        """Per-resample estimates (list of floats for numeric, list of arrays for categorical)."""
        return getattr(self, '_bootstrap_estimates', [])

    @property
    def node_counts(self):
        """Per-resample sample sizes."""
        return getattr(self, '_node_counts', [])

    # ---- Pretty printing -----------------------------------------------------
    def __repr__(self):
        return self._format_output()

    def __str__(self):
        return self._format_output()

    def _format_output(self):
        lines = []
        weight_text = "Weighted" if self._is_weighted else "Not weighted"

        # Results table
        if self._is_categorical:
            lines.append(self._results_df.to_string(index=False))
        else:
            row = self._results_df.iloc[0]
            lines.append(f"Mean                    {row['Mean']}")
            lines.append(f"SE                      {row['SE']}")

        lines.append("")
        lines.append(f"n_Data                  {self._n_original}")
        lines.append(f"n_Analysis              {self._n_analysis}")
        lines.append(f"Weight                  {weight_text}")
        lines.append(f"SE Method               {self._se_method}")

        # Bootstrap summary
        if self._n_iterations is not None and self._node_counts:
            lines.append("")
            lines.append("— Resample Summary —")
            lines.append(f"n_Iteration     {self._n_iterations}")
            node_sd = round(float(np.std(self._node_counts)), 2) if self._node_counts else 'NA'
            lines.append("                Mean    SD")
            lines.append(f"                {self._mean_nodes:.1f}   {node_sd}")
            lines.append("                Min     1Q      Med     3Q      Max")
            lines.append(
                f"                 {self._min_nodes}     {self._q1_nodes}     "
                f"{self._median_nodes}     {self._q3_nodes}     {self._max_nodes}"
            )

        return "\n".join(lines)

    @property
    def _constructor(self):
        """Ensure that pandas operations return RDSResult objects."""
        return RDSResult


# ---------------------------------------------------------------------------
# Helpers for point estimates and standard errors
# ---------------------------------------------------------------------------
def _numeric_mean_se(x_values, weight_values=None):
    """Return (mean, se) for a continuous variable, optionally weighted.

    Assumes NaNs have already been removed. Returns (nan, nan) if empty.
    """
    n = len(x_values)
    if n == 0:
        return float('nan'), float('nan')

    if weight_values is None:
        mean_x = float(x_values.mean())
        if n > 1:
            s_squared = float(((x_values - mean_x) ** 2).sum() / (n - 1))
            se = math.sqrt(s_squared / n)
        else:
            se = float('nan')
        return mean_x, se

    # Weighted case (mirrors the original linearisation-based SE)
    w_sum = float(weight_values.sum())
    if w_sum == 0:
        raise ValueError("Sum of weights is zero")
    x_bar = float((x_values * weight_values).sum() / w_sum)
    w_bar = float(weight_values.mean())
    if n > 1:
        t1 = float(((weight_values * x_values - w_bar * x_bar) ** 2).sum())
        t2 = 2 * x_bar * float(((weight_values - w_bar) *
                                (weight_values * x_values - w_bar * x_bar)).sum())
        t3 = x_bar ** 2 * float(((weight_values - w_bar) ** 2).sum())
        w_var = n / ((n - 1) * (w_sum ** 2)) * (t1 - t2 + t3)
        se = math.sqrt(max(w_var, 0.0))
    else:
        se = float('nan')
    return x_bar, se


def _categorical_props_se(x_values, levels, weight_values=None):
    """Return (proportions, ses) arrays aligned with ``levels``.

    Implements the standard naive (delta-method) SEs that survey::svymean
    returns for factors:

      unweighted: se_k = sqrt(p_k * (1 - p_k) / n)
      weighted:   se_k = sqrt( sum_i w_i^2 * (1{x_i=k} - p_k)^2 / (sum w_i)^2 )
                  scaled by n/(n-1) for finite-sample correction.
    """
    n = len(x_values)
    k = len(levels)
    if n == 0:
        return np.full(k, np.nan), np.full(k, np.nan)

    props = np.zeros(k, dtype=float)
    ses = np.zeros(k, dtype=float)

    if weight_values is None:
        for i, lvl in enumerate(levels):
            indicator = (x_values == lvl).astype(float)
            p = float(indicator.mean())
            props[i] = p
            ses[i] = math.sqrt(p * (1 - p) / n) if n > 0 else float('nan')
    else:
        w = weight_values.astype(float)
        w_sum = float(w.sum())
        if w_sum == 0:
            raise ValueError("Sum of weights is zero")
        for i, lvl in enumerate(levels):
            indicator = (x_values == lvl).astype(float)
            p = float((indicator * w).sum() / w_sum)
            props[i] = p
            if n > 1:
                resid = indicator - p
                var_est = float((w ** 2 * resid ** 2).sum()) / (w_sum ** 2)
                # Finite-sample scaling consistent with survey::svymean
                var_est *= n / (n - 1)
                ses[i] = math.sqrt(max(var_est, 0.0))
            else:
                ses[i] = float('nan')
    return props, ses


def _prepare_x_and_weight(data, x, weight, na_rm):
    """Detect type, coerce, optionally drop NAs.

    Returns
    -------
    x_values : pd.Series (numeric or category-coded)
    weight_values : pd.Series or None
    is_categorical : bool
    levels : list or None
        Sorted level list when categorical, else None.
    n_analysis : int
        Rows used after NA handling.
    has_na : bool
        Whether NAs were present in the relevant columns.
    """
    raw = data[x]
    is_categorical = _is_categorical_series(raw)

    if is_categorical:
        cat = _coerce_categorical(raw)
        levels = list(cat.categories)
        x_values = pd.Series(cat, index=data.index)
        x_na = x_values.isna()
    else:
        levels = None
        x_values = pd.to_numeric(raw, errors='coerce')
        x_na = x_values.isna()

    if weight is not None:
        weight_values = pd.to_numeric(data[weight], errors='coerce')
        w_na = weight_values.isna()
    else:
        weight_values = None
        w_na = pd.Series(False, index=data.index)

    combined_na = x_na | w_na
    has_na = bool(combined_na.any())

    if na_rm and has_na:
        keep = ~combined_na
        x_values = x_values[keep]
        if weight_values is not None:
            weight_values = weight_values[keep]
    # If na_rm is False, leave NAs in place; downstream estimators will
    # propagate NaN, mirroring R's svymean(..., na.rm = FALSE).

    n_analysis = int((~combined_na).sum()) if na_rm else len(data)

    return x_values, weight_values, is_categorical, levels, n_analysis, has_na


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def RDSmean(x, data, weight=None, var_est=None, resample_n=None, n_cores=None,
            na_rm=True, return_bootstrap_means=False, return_node_counts=False):
    """Estimating mean with respondent driven sampling sample data

    This function calculates weighted or unweighted means for either a continuous or a
    categorical variable. Standard errors are calculated using naive or resampling approaches from 'RDSboot'.

    Parameters
    ----------
    x : str
        Name of the variable of interest. May be numeric, boolean,
        string/object, or pandas Categorical. Non-numeric columns are
        treated as categorical.
    data : pandas.DataFrame
        The output DataFrame from ``RDSdata``.
    weight : str, optional
        Name of the weight variable for a weighted analysis. When
        ``None`` (the default), the function performs an unweighted
        analysis.
    var_est : str, optional
        Variance estimation method. One of ``'naive'`` (delta method) or
        the bootstrap types ``'chain1'``, ``'chain2'``, ``'tree_uni1'``,
        ``'tree_uni2'``, ``'tree_bi1'``, ``'tree_bi2'``. When ``None``,
        the naive method is used.
    resample_n : int, optional
        Number of resample iterations. Required only for bootstrap
        methods; defaults to 300 when a bootstrap method is specified
        without an explicit value. Must be ``None`` for the naive method.
    n_cores : int, optional
        Number of CPU cores for parallel bootstrap. If ``None``, uses
        the standard sequential bootstrap.
    na_rm : bool, default True
        If ``True``, observations with missing values in ``x`` (or
        ``weight``, when supplied) are removed before estimation. If
        ``False``, NAs are retained and the estimator will return
        ``NaN`` whenever NAs are present.
    return_bootstrap_means : bool, default False
        If ``True``, also return the per-iteration estimates. For
        categorical variables this is a list of proportion arrays
        (one per resample, aligned with the level order); for
        continuous variables it is a list of scalar means.
    return_node_counts : bool, default False
        If ``True``, also return the per-iteration sample sizes.

    Returns
    -------
    RDSResult or tuple
        An RDSResult object containing the following elements:

        mean
            Numeric; Weighted or unweighted mean estimate. For categorical
            variables, one estimated proportion per level.

        se
            Numeric; Standard error of the mean. For categorical
            variables, one standard error per level.

        additional_info
            Information about the estimation:
            (1) SE method: variance estimation method
            (2) Weight: indicator of whether weighted analysis was used
            (3) n_Data: total number of observations in the input data
            (4) n_Analysis: number of observations used in the analysis
                (after NA removal when ``na_rm=True``)
            (5) n_Iteration: number of resampling iterations (if SE method is not 'naive')

        resample_summary
            Descriptive summary of resamples if var_est is not 'naive': mean, SD,
            min, quartiles, and max of resample sizes

        resample_estimates
            Mean estimates for each resampling iteration if var_est is not 'naive'

        When return_bootstrap_means=False and return_node_counts=False (default):
            Returns RDSResult object only

        When return_bootstrap_means=True and return_node_counts=False:
            Returns (RDSResult, bootstrap_means_list)

        When return_bootstrap_means=False and return_node_counts=True:
            Returns (RDSResult, node_counts_list)

        When return_bootstrap_means=True and return_node_counts=True:
            Returns (RDSResult, bootstrap_means_list, node_counts_list)

    Notes
    -----
    The ``RDSResult`` object is a pandas DataFrame subclass that:

    - Retains all DataFrame functionality for downstream analysis.
    - Renders a formatted multi-line summary when printed.

    For categorical variables the reported "Mean" for each level is the
    estimated proportion of observations in that level.

    In all bootstrap methods, versions 1 and 2 differ as version 1 sets
    the number of seeds in a given resample to be consistent with the
    number of seeds in the original sample (s), while version 2 sets
    the sample size of a given resample (n_r) to be at least equal to
    or greater than the original sample (n_s).

    'chain1' selects s seeds using SRSWR from all seeds in the original
    sample and then all nodes in the chains created by each of the
    resampled seeds are retained. With 'chain2', 1 seed is sampled
    using SRSWR from all seeds in the original sample, and all nodes
    from the chain created by this seed are retained. It then compares
    n_r against n_s, and, if n_r < n_s, continues the resampling
    process by drawing 1 seed and its chains one by one until
    n_r >= n_s.

    In the 'tree_uni1' method, s seeds are selected using Simple Random
    Sampling with Replacement (SRSWR) from all seeds. For each selected
    seed, this method (A) checks its recruit counts, (B) selects SRSWR
    of the recruits counts from all recruits identified in (A), and
    (C) for each sampled recruit, this method repeats Steps A and B.
    (D) Steps A, B, and C continue until reaching the last wave of
    each chain. In 'tree_uni2', instead of selecting s seeds, it
    selects one seed, performs Steps B and C for the selected seed.
    It compares the size of the resample (n_r) and the original sample
    (n_s), and, if n_r < n_s, it continues the resampling process by
    drawing 1 seed, performs Steps B and C and checks n_r against n_s.
    If n_r < n_s, the process continues until the sample size of a
    given resample (n_r) is at least equal to the original sample size
    (n_s), i.e., n_r >= n_s.

    'tree_bi1' selects s nodes from the recruitment chains using SRSWR.
    For each selected node, it (A) checks its connected nodes (i.e.,
    both recruiters and recruits) and their count, (B) from all
    connected nodes identified in (A), performs SRSWR of the same node
    count, and (C) for each selected node, performs steps A and B, but
    does not resample already resampled nodes. (D) Steps A, B, and C
    are repeated until the end of the chain. In 'tree_bi2', instead of
    s nodes, it selects 1 node using SRSWR from anywhere in all
    recruitment chains and repeats steps (B), (C), and (D) until
    n_r >= n_s.

    References
    ----------
    .. [1] Salganik, M. J. (2006). Variance estimation, design effects,
       and sample size calculations for respondent-driven sampling.
       Journal of Urban Health, 83(1), 98-112.
       https://doi.org/10.1007/s11524-006-9106-x
    .. [2] Volz, E., & Heckathorn, D. D. (2008). Probability based
       estimation theory for respondent driven sampling. Journal of
       Official Statistics, 24(1), 79-97.
    """

    # ---- Argument validation ------------------------------------------------
    if n_cores is not None:
        if not isinstance(n_cores, int) or n_cores < 1:
            raise ValueError("n_cores must be a positive integer")

    resample_methods = ['chain1', 'chain2',
                        'tree_uni1', 'tree_uni2',
                        'tree_bi1', 'tree_bi2']

    if resample_n is None and var_est in resample_methods:
        resample_n = 300

    if resample_n is not None and var_est not in resample_methods:
        raise ValueError(
            "resample_n argument should only be applied when var_est is a bootstrap method."
        )

    n_original = len(data)

    # ---- Type detection, coercion, NA handling ------------------------------
    x_values, weight_values, is_categorical, levels, n_analysis, has_na = \
        _prepare_x_and_weight(data, x, weight, na_rm)

    method_label = "Naive" if var_est is None else var_est

    # =========================================================================
    # Naive variance estimation (with or without weights)
    # =========================================================================
    if var_est is None or var_est == 'naive':

        if not na_rm and has_na:
            # Mirror R's svymean(..., na.rm = FALSE): propagate NaN.
            if is_categorical:
                means = np.full(len(levels), np.nan)
                ses = np.full(len(levels), np.nan)
            else:
                means, ses = float('nan'), float('nan')
            result = RDSResult(
                means, ses, method_label, n_original, n_analysis,
                is_weighted=(weight is not None),
                is_categorical=is_categorical, levels=levels,
            )
        else:
            if is_categorical:
                if len(x_values) == 0:
                    raise ValueError(f"No valid values found for variable '{x}'")
                means, ses = _categorical_props_se(x_values, levels, weight_values)
            else:
                if len(x_values) == 0:
                    raise ValueError(f"No valid numeric values found for variable '{x}'")
                means, ses = _numeric_mean_se(x_values, weight_values)

            result = RDSResult(
                means, ses, method_label, n_original, n_analysis,
                is_weighted=(weight is not None),
                is_categorical=is_categorical, levels=levels,
            )

        return _pack_return(result, [], [], return_bootstrap_means, return_node_counts)

    # =========================================================================
    # Bootstrap variance estimation
    # =========================================================================
    # Run the bootstrap once (same call regardless of weight/type)
    if n_cores is not None:
        from parallel_bootstrap import RDSBootOptimizedParallel
        boot_out = RDSBootOptimizedParallel(
            data=data, respondent_id_col='ID', seed_id_col='S_ID',
            seed_col='SEED', recruiter_id_col='R_ID',
            type=var_est, resample_n=resample_n, n_cores=n_cores,
        )
    else:
        from bootstrap import RDSboot
        boot_out = RDSboot(
            data=data, respondent_id_col='ID', seed_id_col='S_ID',
            seed_col='SEED', recruiter_id_col='R_ID',
            type=var_est, resample_n=resample_n,
        )

    merged_data = pd.merge(data, boot_out, on='ID')
    if len(merged_data) == 0:
        raise ValueError("No data after merging with bootstrap results")

    # Per-resample estimates
    bootstrap_estimates = []  # list of float (numeric) or 1D array (categorical)
    node_counts = []

    for resample_id in merged_data['RESAMPLE.N'].unique():
        group = merged_data[merged_data['RESAMPLE.N'] == resample_id]

        # Apply same NA handling per resample
        gx_raw = group[x]
        if is_categorical:
            gx = pd.Categorical(gx_raw, categories=levels)
            gx_series = pd.Series(gx, index=group.index)
        else:
            gx_series = pd.to_numeric(gx_raw, errors='coerce')

        if weight is not None:
            gw = pd.to_numeric(group[weight], errors='coerce')
            valid = ~(gx_series.isna() | gw.isna())
            gx_series = gx_series[valid]
            gw = gw[valid]
        else:
            gw = None
            valid = ~gx_series.isna()
            gx_series = gx_series[valid]

        if len(gx_series) == 0:
            continue

        if is_categorical:
            props, _ = _categorical_props_se(gx_series, levels, gw)
            bootstrap_estimates.append(props)
        else:
            try:
                m, _ = _numeric_mean_se(gx_series, gw)
            except ValueError:
                continue
            if not math.isnan(m):
                bootstrap_estimates.append(m)
        node_counts.append(len(gx_series))

    # Point estimate on the original (NA-handled) data
    if not na_rm and has_na:
        if is_categorical:
            point_means = np.full(len(levels), np.nan)
        else:
            point_means = float('nan')
    else:
        if is_categorical:
            if len(x_values) == 0:
                raise ValueError(f"No valid values found for variable '{x}'")
            point_means, _ = _categorical_props_se(x_values, levels, weight_values)
        else:
            if len(x_values) == 0:
                raise ValueError(f"No valid numeric values found for variable '{x}'")
            point_means, _ = _numeric_mean_se(x_values, weight_values)

    # Bootstrap-based SEs
    if is_categorical:
        if len(bootstrap_estimates) > 1:
            est_matrix = np.vstack(bootstrap_estimates)
            ses = np.sqrt(np.var(est_matrix, axis=0, ddof=0))
        else:
            ses = np.full(len(levels), np.nan)
    else:
        if len(bootstrap_estimates) > 1:
            ses = math.sqrt(float(np.var(bootstrap_estimates, ddof=0)))
        else:
            ses = float('nan')

    result = RDSResult(
        point_means, ses, var_est, n_original, n_analysis,
        is_weighted=(weight is not None),
        is_categorical=is_categorical, levels=levels,
        bootstrap_estimates=bootstrap_estimates,
        node_counts=node_counts, n_iterations=resample_n,
    )

    return _pack_return(result, bootstrap_estimates, node_counts,
                        return_bootstrap_means, return_node_counts)


def _pack_return(result, bootstrap_estimates, node_counts,
                 return_bootstrap_means, return_node_counts):
    """Apply the legacy return-tuple convention."""
    if return_bootstrap_means and return_node_counts:
        return result, bootstrap_estimates, node_counts
    if return_bootstrap_means:
        return result, bootstrap_estimates
    if return_node_counts:
        return result, node_counts
    return result
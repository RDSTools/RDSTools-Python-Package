import pandas as pd
import numpy as np
from bootstrap import RDSboot
from parallel_bootstrap import RDSBootOptimizedParallel


class RDSTableResult:
    """Custom class that displays RDS table results in a formatted way"""

    def __init__(self, prop_table=None, se_table=None, results=None, formula=None,
                 n_original=None, is_weighted=False, var_est_method='naive',
                 bootstrap_tables=None, node_counts=None, n_iterations=None,
                 margins=3, cross_tab=None):

        # Store calculated values as attributes
        self.prop_table = prop_table
        self.se_table = se_table
        self.results = results  # For one-way tables
        self.formula = formula
        self.n_original = n_original  # Always len(data)
        self.is_weighted = is_weighted
        self.var_est_method = var_est_method
        self.bootstrap_tables = bootstrap_tables or []
        self.node_counts = node_counts or []
        self.n_iterations = n_iterations
        self.margins = margins
        self.cross_tab = cross_tab  # Actual unweighted counts

        # Cell counts are ALWAYS the actual unweighted counts
        if prop_table is not None:
            self.cell_counts = {}
            for row_idx in prop_table.index:
                for col_idx in prop_table.columns:
                    if cross_tab is not None and row_idx in cross_tab.index and col_idx in cross_tab.columns:
                        self.cell_counts[(row_idx, col_idx)] = int(cross_tab.loc[row_idx, col_idx])
                    else:
                        self.cell_counts[(row_idx, col_idx)] = 0
        else:
            self.cell_counts = None

        # Calculate node statistics if bootstrap data exists
        if node_counts:
            self.mean_nodes = round(np.mean(node_counts), 1)
            self.min_nodes = int(np.min(node_counts))
            self.q1_nodes = int(np.percentile(node_counts, 25))
            self.median_nodes = int(np.median(node_counts))
            self.q3_nodes = int(np.percentile(node_counts, 75))
            self.max_nodes = int(np.max(node_counts))
        else:
            self.mean_nodes = None
            self.min_nodes = None
            self.q1_nodes = None
            self.median_nodes = None
            self.q3_nodes = None
            self.max_nodes = None

    def __repr__(self):
        return self._format_output()

    def __str__(self):
        return self._format_output()

    def _format_output(self):
        lines = []

        # Add formula
        if self.formula:
            lines.append("Call:")
            lines.append(f"{self.formula}")
            # Calculate appropriate line length based on table type
            if self.results is not None:  # One-way table
                line_length = 41
            else:  # Two-way table
                line_length = self._calculate_table_width(self.prop_table.columns)
            lines.append("-" * line_length)

        # Determine if one-way or two-way table
        if self.results is not None:
            # One-way table
            lines.extend(self._format_one_way_table())
        else:
            # Two-way table
            lines.extend(self._format_two_way_table())

        lines.append("")

        # Add metadata
        lines.extend(self._format_metadata())

        return "\n".join(lines)

    def _format_one_way_table(self):
        """Format one-way table with n, %, (SE) columns"""
        lines = []

        # Get proportions and standard errors
        proportions = self.results['Proportions']
        se_values = self.results['SE']
        counts = self.results['Counts']  # Actual unweighted counts

        # Create formatted table
        lines.append(f"{'':>15} {'n':>8} {'%':>8} {'(SE)'}")
        lines.append("-" * 41)

        for category in proportions.index:
            count = int(counts[category])  # Use actual count
            prop = proportions[category] * 100  # Convert to percentage
            se = se_values[category] * 100  # Convert to percentage

            lines.append(f"{str(category):>15} {count:>8} {prop:>7.1f} ({se:.2f})")

        # Add total row - no SE for total
        lines.append("-" * 41)
        lines.append(f"{'TOTAL':>15} {round(self.n_original):>8} {'100.0':>8}")

        return lines

    def _format_two_way_table(self):
        """Format two-way table with n, %, (SE) for each cell"""
        lines = []

        prop_table = self.prop_table
        se_table = self.se_table

        # Create header
        header_parts = [""]
        for col in prop_table.columns:
            header_parts.extend([str(col), "", ""])  # Three columns per category
        header_parts.extend(["TOTAL", "", ""])

        lines.append(self._create_table_header(header_parts))
        lines.append(self._create_subheader(prop_table.columns))
        lines.append("-" * self._calculate_table_width(prop_table.columns))

        # Add data rows
        for row_idx in prop_table.index:
            row_parts = [str(row_idx)]

            # Use actual unweighted counts
            for col_idx in prop_table.columns:
                count = self.cell_counts[(row_idx, col_idx)]
                prop = prop_table.loc[row_idx, col_idx] * 100
                se = se_table.loc[row_idx, col_idx] * 100
                row_parts.extend([f"{count}", f"{prop:.1f}", f"({se:.2f})"])

            # Row total - calculate from actual counts
            row_total_count = sum(self.cell_counts[(row_idx, col_idx)] for col_idx in prop_table.columns)

            if self.margins == 1:  # Row proportions - always 100%
                row_total_pct = "100.0"
            elif self.margins == 2:  # Column proportions - row totals should be N/A
                row_total_pct = "N/A"
            else:  # Cell proportions (margins == 3)
                row_total_pct = f"{sum(prop_table.loc[row_idx, col_idx] * 100 for col_idx in prop_table.columns):.1f}"

            row_parts.extend([f"{row_total_count}", row_total_pct])
            lines.append(self._format_table_row(row_parts))

        # Add total row
        lines.append("-" * self._calculate_table_width(prop_table.columns))
        total_parts = ["TOTAL"]

        # Column totals - calculate from actual counts
        for col_idx in prop_table.columns:
            col_total_count = sum(self.cell_counts[(row_idx, col_idx)] for row_idx in prop_table.index)

            if self.margins == 2:  # Column proportions - always 100%
                col_total_pct = "100.0"
            elif self.margins == 1:  # Row proportions - column totals should be N/A
                col_total_pct = "N/A"
            else:  # Cell proportions (margins == 3)
                col_total_pct = f"{sum(prop_table.loc[row_idx, col_idx] * 100 for row_idx in prop_table.index):.1f}"

            total_parts.extend([f"{col_total_count}", col_total_pct, ""])

        # Grand total
        total_parts.extend([f"{round(self.n_original)}", "100.0"])

        lines.append(self._format_total_row(total_parts))
        return lines

    def _create_table_header(self, header_parts):
        """Create the main header row for two-way table"""
        formatted_parts = []
        formatted_parts.append(f"{header_parts[0]:>12}")

        for i in range(1, len(header_parts) - 3, 3):
            col_name = header_parts[i]
            formatted_parts.append(f"{col_name:>20}")

        formatted_parts.append(f"{'TOTAL':>20}")

        return "".join(formatted_parts)

    def _create_subheader(self, columns):
        """Create the subheader row with n, %, (SE)"""
        parts = []
        parts.append(f"{'':>12}")

        for _ in columns:
            parts.append(f"{'n':>6}{'%':>7}{'(SE)':>7}")

        parts.append(f"{'n':>10}{'%':>10}")

        return "".join(parts)

    def _format_table_row(self, row_parts):
        """Format a data row for the two-way table"""
        formatted = []
        formatted.append(f"{row_parts[0]:>12}")

        i = 1
        col_count = 0
        num_data_cols = len(self.prop_table.columns)

        while i < len(row_parts):
            if col_count < num_data_cols:
                if i + 2 < len(row_parts):
                    formatted.append(f"{row_parts[i]:>6}{row_parts[i + 1]:>7}{row_parts[i + 2]:>7}")
                    i += 3
                else:
                    break
            else:
                if i + 1 < len(row_parts):
                    pct_value = row_parts[i + 1]
                    if pct_value == "N/A":
                        formatted.append(f"{row_parts[i]:>10}{'N/A':>10}")
                    else:
                        formatted.append(f"{row_parts[i]:>10}{pct_value:>10}")
                    i += 2
                else:
                    break
            col_count += 1

        return "".join(formatted)

    def _calculate_table_width(self, columns):
        """Calculate total width needed for the table"""
        return 12 + (len(columns) + 1) * 20

    def _format_metadata(self):
        """Format the metadata section"""
        lines = []

        lines.append(f"n_Data                  {round(self.n_original)}")

        weight_text = "Weighted" if self.is_weighted else "Not weighted"
        lines.append(f"Weight                  {weight_text}")

        lines.append(f"SE Method               {self.var_est_method}")

        if self.n_iterations is not None and self.node_counts:
            lines.append("")
            lines.append("- Resample Summary -")
            lines.append(f"n_Iteration     {self.n_iterations}")
            lines.append("n_Resample      Mean    SD")

            node_sd = round(np.std(self.node_counts), 2) if self.node_counts else 'NA'
            lines.append(f"                {self.mean_nodes:.1f}   {node_sd}")
            lines.append("                Min     1Q      Med     3Q      Max")
            lines.append(
                f"                 {self.min_nodes}     {self.q1_nodes}     {self.median_nodes}     {self.q3_nodes}     {self.max_nodes}")

        return lines

    def _format_total_row(self, row_parts):
        """Format the TOTAL row for the two-way table"""
        formatted = []
        formatted.append(f"{row_parts[0]:>12}")

        i = 1
        col_count = 0
        num_data_cols = len(self.prop_table.columns)

        while i < len(row_parts):
            if col_count < num_data_cols:
                if i + 2 < len(row_parts):
                    pct_value = row_parts[i + 1]
                    if pct_value == "N/A":
                        formatted.append(f"{row_parts[i]:>6}{'N/A':>7}{'':>7}")
                    else:
                        formatted.append(f"{row_parts[i]:>6}{pct_value:>7}{'':>7}")
                    i += 3
                else:
                    break
            else:
                if i + 1 < len(row_parts):
                    formatted.append(f"{row_parts[i]:>10}{row_parts[i + 1]:>10}")
                    i += 2
                else:
                    break
            col_count += 1

        return "".join(formatted)

def RDStable(formula, data, weight=None, var_est=None, resample_n=None, margins=3, n_cores=None,
             return_bootstrap_tables=False, return_node_counts=False):
    """
    1-2 way tables from an RDS study

    Create contingency tables from RDS data with optional weighting
    and different variance estimation methods.

    Parameters:
    -----------
    formula : str
        For a 2-way table, 2 categorical variables of interest. The variables should be coded as factors. For a 1-way table, specify one variable.
        (e.g., "~Sex" for one-way, "~Sex+Race" for two-way)
    data : pandas.DataFrame
        The output from RDSdata
    weight : str, optional
        Name of the weight variable.
        A user specified weights to calculate weighted proportions and standard errors.
        When set to None the function calculates unweighted proportions and standard errors
    var_est : str, optional
        One of the six bootstrap types or the delta (naive) method.
        By default the function calculates naive standard errors.
        Variance estimation options include 'naive' or bootstrap methods like 'resample_chain1', 'resample_chain2', 'resample_tree_uni1', 'resample_tree_uni2',
        'resample_tree_bi1', 'resample_tree_bi2'
    resample_n : int, optional
        Specifies the number of resample iterations.
        Note that this argument is None when var_est = 'naive'.
        Required for bootstrap methods, default 300
    margins : int, optional
        For two-way tables: 1=row proportions, 2=column proportions, 3=cell proportions (default)
    n_cores : int, optional
        Number of CPU cores to use for parallel bootstrap processing.
        If specified, uses optimized parallel bootstrap. If None, uses
        standard sequential bootstrap.
    return_bootstrap_tables : bool, optional
        If True, return bootstrap table estimates along with main results (only for bootstrap methods)
    return_node_counts : bool, optional
        If True, return node counts per iteration along with main results (only for bootstrap methods)

    Returns:
    --------
    RDSTableResult : RDSTableResult object (custom class)
        When return_bootstrap_tables and return_node_counts are both False (default).
        Object with the following elements; weighted or unweighted proportions and their standard errors,
        additional information about the analysis: (1) var_est method, (2) weighted or not, (3) n_Data, (4) n_Analysis, (5) n_Iteration if var_est is not naive.
        descriptive summary of resamples if var_est is not naive, resample estimates
        Contains formatted table output with proportions, standard errors, and cell counts.

    tuple : (RDSTableResult, list)
        When return_bootstrap_tables is True and return_node_counts is False.
        Returns (formatted_result, bootstrap_tables_list).

    tuple : (RDSTableResult, list)
        When return_bootstrap_tables is False and return_node_counts is True.
        Returns (formatted_result, node_counts_list).

    tuple : (RDSTableResult, list, list)
        When both return_bootstrap_tables and return_node_counts are True.
        Returns (formatted_result, bootstrap_tables_list, node_counts_list).

    Notes
    -----
    The RDSTableResult object is a custom class that:
    - Displays nicely formatted output when printed
    - Contains proportions and standard errors from contingency table analysis
    - Uses actual unweighted counts for cell counts regardless of weighting
    - Weights only affect standard error calculations

    For bootstrap methods, the object includes additional bootstrap statistics:
    - n_iterations: number of bootstrap resamples performed
    - mean_nodes: average number of nodes (observations) across all bootstrap samples
    - min_nodes: minimum number of nodes in any bootstrap sample
    - q1_nodes: 25th percentile of nodes across bootstrap samples
    - median_nodes: median number of nodes across bootstrap samples
    - q3_nodes: 75th percentile of nodes across bootstrap samples
    - max_nodes: maximum number of nodes in any bootstrap sample

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

    # Calculate RDStable using data preprocessed by RDSdata
    out = RDStable("~Sex", data = rds_data, weight = 'DEGREE',
                   var_est = 'resample_chain1',
                   resample_n = 100)
    print(out)
    """

    if n_cores is not None:
        if not isinstance(n_cores, int) or n_cores < 1:
            raise ValueError("n_cores must be a positive integer")

    # Parse formula
    variables = formula.strip().split("~")[1].split("+")
    variables = [var.strip() for var in variables]

    if len(variables) > 2:
        raise ValueError('The function supports only 1 and 2-way tables')

    if var_est is None:
        var_est = "naive"

    resample_methods = [
        "resample_chain1", "resample_chain2", "resample_tree_uni1",
        "resample_tree_uni2", "resample_tree_bi1", "resample_tree_bi2"
    ]

    if resample_n is not None and var_est not in resample_methods:
        raise ValueError("resample_n should only be used with bootstrap methods.")

    if resample_n is None and var_est in resample_methods:
        resample_n = 300

    # One-way table
    if len(variables) == 1:
        if var_est == "naive":
            result = compute_naive_one_way(variables[0], data, weight, formula)
            if return_bootstrap_tables:
                if return_node_counts:
                    return result, [], []
                return result, []
            if return_node_counts:
                return result, []
            return result
        else:
            return compute_bootstrap_one_way(variables[0], data, weight, var_est, resample_n, n_cores,
                                             formula, return_bootstrap_tables, return_node_counts)

    # Two-way table
    else:
        if var_est == "naive":
            result = compute_naive_two_way(variables[0], variables[1], data, weight, margins, formula)
            if return_bootstrap_tables:
                if return_node_counts:
                    return result, [], []
                return result, []
            if return_node_counts:
                return result, []
            return result
        else:
            return compute_bootstrap_two_way(variables[0], variables[1], data, weight, var_est, resample_n, margins,
                                             n_cores, formula, return_bootstrap_tables, return_node_counts)


# Helper functions for one-way tables
def compute_naive_one_way(variable, data, weight, formula):
    """Compute proportions and SEs for one-way table using naive method.
    Counts and proportions are always from unweighted data.
    Weights only affect SE calculation."""

    # Calculate proportions from UNWEIGHTED data
    counts = data[variable].value_counts()
    n = len(data)
    proportions = counts / n

    # Calculate SEs based on weight
    if weight is None:
        # Unweighted SE
        se = np.sqrt(proportions * (1 - proportions) / n)
    else:
        # Weighted SE calculation
        se = pd.Series(index=proportions.index)
        total_weight = data[weight].sum()
        for val in proportions.index:
            mask = (data[variable] == val).astype(int)
            p_i = proportions[val]
            weights = data[weight] / total_weight
            se_i = np.sqrt(np.sum(weights ** 2 * (mask - p_i) ** 2))
            se[val] = se_i

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Counts': counts,
        'Proportions': proportions,
        'SE': se
    })

    return RDSTableResult(
        results=results_df,
        formula=formula,
        n_original=n,
        is_weighted=(weight is not None),
        var_est_method='naive',
    )


def compute_bootstrap_one_way(variable, data, weight, var_est, resample_n, n_cores, formula,
                              return_bootstrap_tables=False, return_node_counts=False):
    """Compute proportions and SEs for one-way table using bootstrap methods.
    Counts and proportions are always from unweighted data.
    Weights only affect SE calculation."""

    # STEP 1: Calculate proportions from UNWEIGHTED original data
    original_counts = data[variable].value_counts()
    proportions = original_counts / len(data)
    n_original = len(data)

    # STEP 2: Generate bootstrap samples
    if n_cores is not None:
        boot_results = RDSBootOptimizedParallel(
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
        boot_results = RDSboot(
            data=data,
            respondent_id_col='ID',
            seed_id_col='S_ID',
            seed_col='SEED',
            recruiter_id_col='R_ID',
            type=var_est,
            resample_n=resample_n
        )

    data_copy = data.copy()
    data_copy = data_copy.rename(columns={'ID': 'RESPONDENT_ID'})
    merged = pd.merge(data_copy, boot_results, on='RESPONDENT_ID')

    # STEP 3: Compute bootstrap estimates
    bootstrap_props = []
    node_counts = []

    for i in range(1, resample_n + 1):
        sample = merged[merged['RESAMPLE.N'] == i]
        node_counts.append(len(sample))

        # Always use UNWEIGHTED proportions for bootstrap
        if len(sample) > 0:
            counts = sample[variable].value_counts()
            props = counts / len(sample)
        else:
            props = pd.Series()

        props = props.reindex(proportions.index, fill_value=0)
        bootstrap_props.append(props)

    # STEP 4: Calculate SEs from bootstrap distribution
    se = pd.Series(index=proportions.index)
    for val in proportions.index:
        values = [float(p[val]) for p in bootstrap_props if pd.notna(p[val])]
        if len(values) > 0:
            se[val] = np.std(values)
        else:
            se[val] = 0.0

    # Create results DataFrame using ORIGINAL unweighted data
    results_df = pd.DataFrame({
        'Counts': original_counts,
        'Proportions': proportions,
        'SE': se
    })

    result = RDSTableResult(
        results=results_df,
        formula=formula,
        n_original=n_original,
        is_weighted=(weight is not None),
        var_est_method=var_est,
        bootstrap_tables=bootstrap_props,
        node_counts=node_counts,
        n_iterations=resample_n,
    )

    if return_bootstrap_tables:
        if return_node_counts:
            return result, bootstrap_props, node_counts
        return result, bootstrap_props
    if return_node_counts:
        return result, node_counts
    return result


# Helper functions for two-way tables
def compute_naive_two_way(var1, var2, data, weight, margins, formula):
    """Compute proportions and SEs for two-way table using naive method.
    Counts and proportions are always from unweighted data.
    Weights only affect SE calculation."""

    # Create UNWEIGHTED cross-tabulation
    cross_tab = pd.crosstab(data[var1], data[var2])

    # Calculate proportions from UNWEIGHTED data based on margins
    if margins == 3:  # Cell proportions
        prop_table = cross_tab / cross_tab.sum().sum()
    elif margins == 1:  # Row proportions
        prop_table = cross_tab.div(cross_tab.sum(axis=1), axis=0)
    elif margins == 2:  # Column proportions
        prop_table = cross_tab.div(cross_tab.sum(axis=0), axis=1)

    # Calculate SEs
    n = len(data)
    se_table = pd.DataFrame(0.0, index=prop_table.index, columns=prop_table.columns)

    if weight is None:
        # Unweighted SEs
        for i in prop_table.index:
            for j in prop_table.columns:
                p = prop_table.loc[i, j]

                if margins == 3:
                    se = np.sqrt(p * (1 - p) / n)
                elif margins == 1:
                    n_i = cross_tab.loc[i].sum()
                    se = np.sqrt(p * (1 - p) / n_i)
                elif margins == 2:
                    n_j = cross_tab[j].sum()
                    se = np.sqrt(p * (1 - p) / n_j)

                se_table.loc[i, j] = se
    else:
        # Weighted SEs
        total_weight = data[weight].sum()
        indicators = {}

        for i in prop_table.index:
            for j in prop_table.columns:
                indicator = ((data[var1] == i) & (data[var2] == j)).astype(int)
                indicators[(i, j)] = indicator

        for i in prop_table.index:
            for j in prop_table.columns:
                p = prop_table.loc[i, j]
                indicator = indicators[(i, j)]

                if margins == 3:
                    weights = data[weight] / total_weight
                    se = np.sqrt(np.sum(weights ** 2 * (indicator - p) ** 2))
                elif margins == 1:
                    row_data = data[data[var1] == i]
                    row_weight = row_data[weight].sum()
                    weights = row_data[weight] / row_weight
                    indicator_row = indicator[data[var1] == i]
                    se = np.sqrt(np.sum(weights ** 2 * (indicator_row - p) ** 2))
                elif margins == 2:
                    col_data = data[data[var2] == j]
                    col_weight = col_data[weight].sum()
                    weights = col_data[weight] / col_weight
                    indicator_col = indicator[data[var2] == j]
                    se = np.sqrt(np.sum(weights ** 2 * (indicator_col - p) ** 2))

                se_table.loc[i, j] = se

    return RDSTableResult(
        prop_table=prop_table,
        se_table=se_table,
        formula=formula,
        n_original=n,
        is_weighted=(weight is not None),
        var_est_method='naive',
        margins=margins,
        cross_tab=cross_tab
    )


def compute_bootstrap_two_way(var1, var2, data, weight, var_est, resample_n, margins, n_cores, formula,
                              return_bootstrap_tables=False, return_node_counts=False):
    """Compute proportions and SEs for two-way table using bootstrap methods.
    Counts and proportions are always from unweighted data.
    Weights only affect SE calculation."""

    # STEP 1: Calculate from UNWEIGHTED original data
    original_cross_tab = pd.crosstab(data[var1], data[var2])
    n_original = len(data)

    # Calculate proportions from UNWEIGHTED data
    if margins == 3:
        prop_table = original_cross_tab / original_cross_tab.sum().sum()
    elif margins == 1:
        prop_table = original_cross_tab.div(original_cross_tab.sum(axis=1), axis=0)
    elif margins == 2:
        prop_table = original_cross_tab.div(original_cross_tab.sum(axis=0), axis=1)

    # STEP 2: Generate bootstrap samples
    if n_cores is not None:
        boot_results = RDSBootOptimizedParallel(
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
        boot_results = RDSboot(
            data=data,
            respondent_id_col='ID',
            seed_id_col='S_ID',
            seed_col='SEED',
            recruiter_id_col='R_ID',
            type=var_est,
            resample_n=resample_n
        )

    data_copy = data.copy()
    data_copy = data_copy.rename(columns={'ID': 'RESPONDENT_ID'})
    merged = pd.merge(data_copy, boot_results, on='RESPONDENT_ID')

    # STEP 3: Compute bootstrap estimates using UNWEIGHTED data
    bootstrap_tables = []
    node_counts = []

    for i in range(1, resample_n + 1):
        sample = merged[merged['RESAMPLE.N'] == i]
        node_counts.append(len(sample))

        if len(sample) == 0:
            sample_props = pd.DataFrame(0.0, index=prop_table.index, columns=prop_table.columns)
        else:
            # Always use UNWEIGHTED cross-tab
            sample_tab = pd.crosstab(sample[var1], sample[var2])
            sample_tab = sample_tab.reindex(index=original_cross_tab.index, columns=original_cross_tab.columns,
                                            fill_value=0)

            # Calculate proportions using same margins
            if margins == 3:
                sample_props = sample_tab / sample_tab.sum().sum() if sample_tab.sum().sum() > 0 else pd.DataFrame(0.0,
                                                                                                                   index=prop_table.index,
                                                                                                                   columns=prop_table.columns)
            elif margins == 1:
                sample_props = sample_tab.div(sample_tab.sum(axis=1), axis=0).fillna(0)
            elif margins == 2:
                sample_props = sample_tab.div(sample_tab.sum(axis=0), axis=1).fillna(0)

        sample_props = sample_props.reindex(index=prop_table.index, columns=prop_table.columns, fill_value=0)
        bootstrap_tables.append(sample_props)

    # STEP 4: Calculate SEs from bootstrap distribution
    se_table = pd.DataFrame(0.0, index=prop_table.index, columns=prop_table.columns)

    for i in prop_table.index:
        for j in prop_table.columns:
            values = [float(tab.loc[i, j]) for tab in bootstrap_tables if pd.notna(tab.loc[i, j])]
            if len(values) > 0:
                se_table.loc[i, j] = np.std(values)
            else:
                se_table.loc[i, j] = 0.0

    result = RDSTableResult(
        prop_table=prop_table,
        se_table=se_table,
        formula=formula,
        n_original=n_original,
        is_weighted=(weight is not None),
        var_est_method=var_est,
        bootstrap_tables=bootstrap_tables,
        node_counts=node_counts,
        n_iterations=resample_n,
        margins=margins,
        cross_tab=original_cross_tab
    )

    if return_bootstrap_tables:
        if return_node_counts:
            return result, bootstrap_tables, node_counts
        return result, bootstrap_tables
    if return_node_counts:
        return result, node_counts
    return result
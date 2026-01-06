import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from bootstrap import RDSboot
from parallel_bootstrap import RDSBootOptimizedParallel
from scipy import stats


class RDSRegressionResult:
    """Custom class that displays RDS regression results in a formatted way"""

    def __init__(self, coefficients, r_squared, f_statistic, formula, n_original,
                 is_weighted=False, var_est_method='naive', bootstrap_estimates=None,
                 node_counts=None, n_iterations=None, residual_std_error=None, results_object=None,
                 df_resid=None, logistic_info=None):

        # Store calculated values as attributes
        self.coefficients = coefficients
        self.r_squared = r_squared
        self.f_statistic = f_statistic
        self.formula = formula
        self.n_original = n_original
        self.is_weighted = is_weighted
        self.var_est_method = var_est_method
        self.bootstrap_estimates = bootstrap_estimates or []
        self.node_counts = node_counts or []
        self.n_iterations = n_iterations
        self.residual_std_error = residual_std_error
        self.results_object = results_object
        self.df_resid = df_resid
        self.logistic_info = logistic_info

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
        lines.append("Call:")
        lines.append(f"{self.formula}")
        lines.append("--------------------------------------------------------")

        # Format coefficients
        lines.append("Coefficients:")

        # Make a copy of the coefficients and filter out confidence interval columns
        coef_df = self.coefficients.copy()

        # Filter out confidence interval columns
        cols_to_keep = []
        for col in coef_df.columns:
            # Skip any column that has brackets (like [0.025, 0.975])
            if isinstance(col, str) and ('[' in col or ']' in col or 'conf' in col.lower()):
                continue
            # Skip tuple columns which are often used for confidence intervals
            if isinstance(col, tuple):
                continue
            cols_to_keep.append(col)

        # Keep only standard columns - UPDATED to include 'z value'
        standard_cols = ['Estimate', 'Std.Error', 't_values', 'p_values', 'z value']
        cols_to_keep = [col for col in cols_to_keep if col in standard_cols]

        if cols_to_keep:
            coef_df = coef_df[cols_to_keep]

        coef_str = coef_df.to_string(float_format=lambda x: f"{x:.4f}")
        lines.append(coef_str)
        lines.append("")

        # Different output for linear vs logistic regression
        if self.residual_std_error is not None:  # Linear regression
            # Add significance codes
            lines.append("")
            lines.append(
                f"Residual standard error: {self.residual_std_error:.4f} on {self.df_resid:.2f} degrees of freedom")

            # Format R-squared for linear regression
            r_sq = self.r_squared
            lines.append(f"Multiple R-squared:  {r_sq[0]:.4f}, Adjusted R-squared:  {r_sq[1]:.4f}")

            # Format F-statistic for linear regression
            f_stat = self.f_statistic
            if not np.isnan(f_stat[0]):
                f_pvalue = 1 - stats.f.cdf(f_stat[0], f_stat[1], f_stat[2])
                lines.append(f"F-statistic: {f_stat[0]:.4f} with p-value: {f_pvalue:.4f}")

        else:  # Logistic regression
            # Add deviance information for logistic regression
            if self.logistic_info is not None:
                null_deviance = self.logistic_info.get('null_deviance', 0)
                residual_deviance = self.logistic_info.get('residual_deviance', 0)
                df_null = self.logistic_info.get('df_null', 0)
                df_resid = self.logistic_info.get('df_resid', 0)
                aic = self.logistic_info.get('aic', 0)

                lines.append(f"    Null deviance: {null_deviance:.1f}  on {df_null:.4f}  degrees of freedom")
                lines.append(f"Residual deviance: {residual_deviance:.1f}  on {df_resid:.4f}  degrees of freedom")
                lines.append(f"AIC: {aic:.1f}")

        lines.append("--------------------------------------------------------")
        lines.append("")

        # Determine weight text
        weight_text = "Weighted" if self.is_weighted else "Not weighted"

        # Basic output (similar to mean.py format)
        lines.append(f"n_Data                  {self.n_original}")
        lines.append(f"Weight                  {weight_text}")
        lines.append(f"SE Method               {self.var_est_method}")

        # Bootstrap summary if available
        if self.n_iterations is not None and self.node_counts:
            lines.append("")
            lines.append("— Resample Summary —")
            lines.append(f"n_Iteration     {self.n_iterations}")
            lines.append("n_Resample      Mean    SD")

            node_sd = round(np.std(self.node_counts), 2) if self.node_counts else 'NA'
            lines.append(f"                {self.mean_nodes:.1f}   {node_sd}")
            lines.append("                Min     1Q      Med     3Q      Max")
            lines.append(
                f"                 {self.min_nodes}     {self.q1_nodes}     {self.median_nodes}     {self.q3_nodes}     {self.max_nodes}")

        return "\n".join(lines)


def RDSlm(data, formula, weight=None, var_est=None, resample_n=None, n_cores=None,
                  return_bootstrap_estimates=False, return_node_counts=False):
    """
    Estimating linear and logistic regression models with respondent driven sampling sample data.
    Equivalent to lm in R stats package with capabilities to handle RDS data in model estimation.

    Parameters:
    -----------
    data : pandas.DataFrame
        The output DataFrame from RDSdata
    formula : str
        Description of the model with dependent and independent variables. (e.g., "y ~ x1 + x2")
        Note that the function performs linear regression when the dependent variable is numeric
        and logistic regression with binomial link function when the dependent variable is either character or factor
    weight : str, optional
        Name of the weight variable.
        User specified weight variable for a weighted analysis.
        When set to NULL, the function performs an unweighted analysis.
    var_est : str, optional
        One of the six bootstrap types or the delta (naive) method.
        By default, the function calculates naive standard errors.
        Variance estimation options include 'naive' or bootstrap methods like 'chain1', 'chain2', 'tree_uni1', 'tree_uni2',
        'tree_bi1', 'tree_bi2'
    resample_n : int, optional
        Specifies the number of resample iterations.
        Note that this argument is None when var_est = 'naive'.
        Required for bootstrap methods, default 300
    n_cores : int, optional
        Number of CPU cores to use for parallel bootstrap processing.
        If specified, uses optimized parallel bootstrap. If None, uses
        standard sequential bootstrap.
    return_bootstrap_estimates : bool, optional
        If True, return bootstrap coefficient estimates along with main results (only for bootstrap methods)
    return_node_counts : bool, optional
        If True, return node counts per iteration along with main results (only for bootstrap methods)

    Returns:
    --------
    dict or tuple
        - If both return_bootstrap_estimates and return_node_counts are False (default):
          Dictionary containing regression results
        - If return_bootstrap_estimates is True and return_node_counts is False:
          Tuple of (results dict, list of bootstrap estimates)
        - If return_bootstrap_estimates is False and return_node_counts is True:
          Tuple of (results dict, list of node counts)
        - If both return_bootstrap_estimates and return_node_counts are True:
          Tuple of (results dict, list of bootstrap estimates, list of node counts)

        Results dictionary contains:
        - coefficients: DataFrame with point estimates, se, t-values and p-values
        - r_squared: List containing [R-squared, Adjusted R-squared]
        - f_statistic: List containing [F-statistic, df_model, df_resid] (NaN for logistic regression)
        - metadata: DataFrame with additional information about the analysis

        For bootstrap methods, metadata includes additional fields:
        - n_iterations: number of bootstrap iterations
        - mean_nodes: mean nodes per bootstrap iteration
        - min_nodes: minimum nodes per bootstrap iteration
        - q1_nodes: 25th percentile nodes per bootstrap iteration
        - median_nodes: median nodes per bootstrap iteration
        - q3_nodes: 75th percentile nodes per bootstrap iteration
        - max_nodes: maximum nodes per bootstrap iteration

    Examples:
    --------
    # Preprocess data with RDSdata function
    rds_data = RDSdata(data = RDSToolsToyData,
                       unique_id = "ID",
                       redeemed_coupon = "CouponR",
                       issued_coupons = ["Coupon1",
                                         "Coupon2",
                                         "Coupon3"],
                       degree = "Degree")

    # Run the model using data preprocessed by RDSData
    out = RDSlm(data = rds_data,
                        formula = "Age~Sex",
                        weight = 'WEIGHT',
                        var_est = 'chain1',
                        resample_n = 100)
    print(out)
    """
    # CPU core count check
    if n_cores is not None:
        if not isinstance(n_cores, int) or n_cores < 1:
            raise ValueError("n_cores must be a positive integer")

    # Valid bootstrap methods
    resample_methods = ['chain1', 'chain2',
                        'tree_uni1', 'tree_uni2',
                        'tree_bi1', 'tree_bi2']

    # Check if bootstrap is specified but not a valid method
    if resample_n is not None and var_est not in resample_methods:
        raise ValueError("resample_n argument should only be applied when var_est is a bootstrap method.")

    # Default resample_n if bootstrap method is specified but resample_n is not
    if resample_n is None and var_est in resample_methods:
        resample_n = 300

    # Parse formula to get dependent variable
    formula_parts = formula.split('~')
    if len(formula_parts) != 2:
        raise ValueError("Formula must be in the form 'y ~ x1 + x2'")

    y_var = formula_parts[0].strip()

    # Store original row count
    n_original = len(data)

    # Drop missing values in y_var
    y_values = data[y_var].dropna()
    data_valid = data.dropna(subset=[y_var])
    n_analysis = len(data_valid)

    # Check if y_var is categorical
    if pd.api.types.is_categorical_dtype(data[y_var]) or len(data[y_var].unique()) <= 2:
        model_type = 'logistic'
    else:
        model_type = 'linear'

    # Helper function to generate metadata for the output
    def generate_metadata(method, n_original, n_iterations=None, is_weighted=False,
                          mean_nodes=None, min_nodes=None, q1_nodes=None,
                          median_nodes=None, q3_nodes=None, max_nodes=None):
        point_est = "Weighted" if is_weighted else "Unweighted"

        metadata = {
            'Point.est': point_est,
            'SE_Method': method,
            'n_original': n_original
        }

        if n_iterations is not None:
            metadata['n_iterations'] = n_iterations

        if mean_nodes is not None:
            metadata['mean_nodes'] = mean_nodes
            metadata['min_nodes'] = min_nodes
            metadata['q1_nodes'] = q1_nodes
            metadata['median_nodes'] = median_nodes
            metadata['q3_nodes'] = q3_nodes
            metadata['max_nodes'] = max_nodes

        return pd.DataFrame({'Metric': list(metadata.keys()), 'Value': list(metadata.values())})

    # 1. Unweighted or weighted with naive variance estimation
    if var_est == 'naive' or var_est is None:
        # Fit the model using statsmodels
        if model_type == 'linear':
            # Linear regression
            if weight is None:
                model = smf.ols(formula=formula, data=data_valid)
                results = model.fit()
            else:
                model = smf.wls(formula=formula, data=data_valid, weights=data_valid[weight])
                results = model.fit()
        else:
            # Logistic regression using GLM instead of smf.logit for proper weighting
            y, X = patsy.dmatrices(formula, data_valid, return_type='dataframe')

            if weight is None:
                model = sm.GLM(y, X, family=sm.families.Binomial())
                try:
                    results = model.fit()
                except:
                    results = model.fit_regularized(alpha=0.01)
            else:
                model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=data_valid[weight])
                try:
                    results = model.fit()
                except:
                    results = model.fit_regularized(alpha=0.01)

        # Extract coefficient table
        coef_df = results.summary2().tables[1].copy()

        # Rename columns to match R output - handle logistic vs linear differently
        if model_type == 'linear':
            coef_df = coef_df.rename(columns={
                'Coef.': 'Estimate',
                'Std.Err.': 'Std.Error',
                't': 't_values',
                'P>|t|': 'p_values'
            })
        else:  # logistic regression
            coef_df = coef_df.rename(columns={
                'Coef.': 'Estimate',
                'Std.Err.': 'Std.Error',
                'z': 'z value',
                'P>|z|': 'p_values'
            })

        # Extract R-squared and F-statistic
        if model_type == 'linear':
            r_squared = [results.rsquared, results.rsquared_adj]
            f_statistic = [results.fvalue, results.df_model, results.df_resid]
            residual_std_error = np.sqrt(results.mse_resid)
            df_resid_val = results.df_resid
            logistic_info = None
        else:
            # For logistic regression - GLM doesn't have prsquared
            r_squared = [np.nan, np.nan]  # No R-squared for logistic
            f_statistic = [np.nan, np.nan, np.nan]
            residual_std_error = None
            df_resid_val = results.df_resid

            logistic_info = {
                'null_deviance': results.llnull * -2,
                'residual_deviance': results.llf * -2,
                'df_null': results.df_model + results.df_resid,
                'df_resid': results.df_resid,
                'aic': results.aic
            }

        # Create RDSRegressionResult object
        result = RDSRegressionResult(
            coefficients=coef_df,
            r_squared=r_squared,
            f_statistic=f_statistic,
            formula=formula,
            n_original=n_original,
            is_weighted=(weight is not None),
            var_est_method="Naive" if var_est is None else var_est,
            residual_std_error=residual_std_error,
            df_resid=df_resid_val,
            logistic_info=logistic_info
        )

        if return_bootstrap_estimates:
            if return_node_counts:
                return result, [], []
            return result, []
        if return_node_counts:
            return result, []
        return result

    # 2. Bootstrapped variance estimation
    elif var_est in resample_methods:
        # Fit the model for point estimates
        if model_type == 'linear':
            # Linear regression
            if weight is None:
                point_model = smf.ols(formula=formula, data=data_valid)
                point_results = point_model.fit()
            else:
                point_model = smf.wls(formula=formula, data=data_valid, weights=data_valid[weight])
                point_results = point_model.fit()
        else:
            # Logistic regression using GLM for proper weighting
            y, X = patsy.dmatrices(formula, data_valid, return_type='dataframe')

            if weight is None:
                point_model = sm.GLM(y, X, family=sm.families.Binomial())
                try:
                    point_results = point_model.fit()
                except:
                    point_results = point_model.fit_regularized(alpha=0.01)
            else:
                point_model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=data_valid[weight])
                try:
                    point_results = point_model.fit()
                except:
                    point_results = point_model.fit_regularized(alpha=0.01)

        # Run bootstrap - FIXED: Use new column-based signature
        if n_cores is not None:
            # Use optimized parallel bootstrap
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
            boot_out = RDSboot(
                data=data,
                respondent_id_col='ID',
                seed_id_col='S_ID',
                seed_col='SEED',
                recruiter_id_col='R_ID',
                type=var_est,
                resample_n=resample_n
            )

        merged_data = pd.merge(data, boot_out, on='ID')

        # Calculate bootstrap estimates and collect model fit statistics
        bootstrap_estimates = []
        bootstrap_r_squared = []
        bootstrap_adj_r_squared = []
        bootstrap_f_stats = []
        bootstrap_rse = []
        bootstrap_df_resid = []
        bootstrap_null_deviance = []
        bootstrap_residual_deviance = []
        bootstrap_aic = []
        node_counts = []

        for resample_id in merged_data['RESAMPLE.N'].unique():
            resample_data = merged_data[merged_data['RESAMPLE.N'] == resample_id]
            node_counts.append(len(resample_data))

            try:
                if model_type == 'linear':
                    if weight is None:
                        resample_model = smf.ols(formula=formula, data=resample_data)
                        resample_results = resample_model.fit()
                    else:
                        resample_model = smf.wls(formula=formula, data=resample_data, weights=resample_data[weight])
                        resample_results = resample_model.fit()

                    # Collect model fit statistics for linear regression
                    bootstrap_r_squared.append(resample_results.rsquared)
                    bootstrap_adj_r_squared.append(resample_results.rsquared_adj)
                    bootstrap_f_stats.append(resample_results.fvalue)
                    bootstrap_rse.append(np.sqrt(resample_results.mse_resid))
                    bootstrap_df_resid.append(resample_results.df_resid)

                else:
                    # Logistic regression using GLM for proper weighting
                    y_resample, X_resample = patsy.dmatrices(formula, resample_data, return_type='dataframe')

                    if weight is None:
                        resample_model = sm.GLM(y_resample, X_resample, family=sm.families.Binomial())
                        try:
                            resample_results = resample_model.fit()
                        except:
                            resample_results = resample_model.fit_regularized(alpha=0.01)
                    else:
                        resample_model = sm.GLM(y_resample, X_resample, family=sm.families.Binomial(),
                                                freq_weights=resample_data[weight])
                        try:
                            resample_results = resample_model.fit()
                        except:
                            resample_results = resample_model.fit_regularized(alpha=0.01)

                    # Collect logistic regression statistics
                    bootstrap_df_resid.append(resample_results.df_resid)
                    bootstrap_null_deviance.append(resample_results.llnull * -2)
                    bootstrap_residual_deviance.append(resample_results.llf * -2)
                    bootstrap_aic.append(resample_results.aic)

                bootstrap_estimates.append(resample_results.params)
            except Exception as e:
                # Skip this bootstrap sample if there's an error
                print(f"Error in bootstrap sample {resample_id}: {str(e)}")
                continue

        # Convert to DataFrame for easier handling
        bootstrap_df = pd.DataFrame(bootstrap_estimates)

        # Calculate bootstrap standard errors: se_B(β̂) = sqrt(1/(B-1) * Σ(β̂_r - mean(β̂))^2)
        B = len(bootstrap_estimates)
        bootstrap_mean = bootstrap_df.mean()
        bootstrap_se = bootstrap_df.std(ddof=1)  # ddof=1 for 1/(B-1) divisor

        # Use ORIGINAL point estimates (not bootstrap mean)
        point_estimates = point_results.params

        # Calculate average degrees of freedom: n̄ - p
        avg_df_resid = np.mean(bootstrap_df_resid)

        # Calculate t-statistics or z-statistics: β̂ / se_B(β̂)
        t_z_values = point_estimates / bootstrap_se

        # Calculate p-values using t-distribution with average df for linear, normal for logistic
        if model_type == 'linear':
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_z_values), df=avg_df_resid))
        else:
            p_values = 2 * (1 - stats.norm.cdf(np.abs(t_z_values)))

        # Create coefficient table with proper column names for model type
        if model_type == 'linear':
            coef_df = pd.DataFrame({
                'Estimate': point_estimates,
                'Std.Error': bootstrap_se,
                't_values': t_z_values,
                'p_values': p_values
            })
        else:  # logistic regression
            coef_df = pd.DataFrame({
                'Estimate': point_estimates,
                'Std.Error': bootstrap_se,
                'z value': t_z_values,
                'p_values': p_values
            })

        # Calculate averaged model fit statistics
        if model_type == 'linear':
            # R̄² = mean(R²_r) across bootstrap samples
            avg_r_squared = np.mean(bootstrap_r_squared)
            avg_adj_r_squared = np.mean(bootstrap_adj_r_squared)
            r_squared = [avg_r_squared, avg_adj_r_squared]

            # F̄ = mean(F_r) across bootstrap samples
            avg_f_stat = np.mean(bootstrap_f_stats)
            # Use average df from bootstrap samples
            avg_df_model = point_results.df_model  # This stays constant
            f_statistic = [avg_f_stat, avg_df_model, avg_df_resid]

            # RSE = mean(RSE_r) across bootstrap samples
            residual_std_error = np.mean(bootstrap_rse)

            logistic_info = None
        else:
            r_squared = [np.nan, np.nan]  # No R-squared for logistic
            f_statistic = [np.nan, np.nan, np.nan]
            residual_std_error = None

            # Calculate averaged logistic regression statistics
            # D̄₀ = mean(D₀_r) - average null deviance
            avg_null_deviance = np.mean(bootstrap_null_deviance)
            # D̄_res = mean(D_res_r) - average residual deviance
            avg_residual_deviance = np.mean(bootstrap_residual_deviance)
            # ĀIC = mean(AIC_r) - average AIC
            avg_aic = np.mean(bootstrap_aic)

            # df_null stays constant (it's based on model structure)
            df_null = point_results.df_model + point_results.df_resid

            logistic_info = {
                'null_deviance': avg_null_deviance,
                'residual_deviance': avg_residual_deviance,
                'df_null': df_null,
                'df_resid': avg_df_resid,
                'aic': avg_aic
            }

        # Create RDSRegressionResult object (node statistics calculated in the class)
        result = RDSRegressionResult(
            coefficients=coef_df,
            r_squared=r_squared,
            f_statistic=f_statistic,
            formula=formula,
            n_original=n_original,
            is_weighted=(weight is not None),
            var_est_method=var_est,
            bootstrap_estimates=bootstrap_estimates,
            node_counts=node_counts,
            n_iterations=resample_n,
            residual_std_error=residual_std_error,
            results_object=point_results,
            df_resid=avg_df_resid,
            logistic_info=logistic_info
        )

        if return_bootstrap_estimates:
            if return_node_counts:
                return result, bootstrap_estimates, node_counts
            return result, bootstrap_estimates
        if return_node_counts:
            return result, node_counts
        return result

    raise ValueError("Invalid var_est option provided. Use 'naive' or a bootstrap method.")
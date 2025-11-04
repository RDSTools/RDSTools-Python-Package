import pandas as pd
import numpy as np
import warnings


def RDS_data(data, unique_id, redeemed_coupon, issued_coupons, degree, zero_degree="hotdeck", NA_degree="hotdeck"):
    """
    Data processing in an RDS study.

    Parameters
    ----------
    data : pandas.DataFrame
        Should contain ID numbers for the nodes in the social network, corresponding redeemed coupon number, and issued coupon number
    unique_id : str
        The column name of the column in the data that represents the ID numbers for the nodes in the social network
    redeemed_coupon : str
        The column name of the column in the data that represents coupon numbers of coupons redeemed by respondents when participating in the survey
    issued_coupons : list of str
        The column name of the column in the data that represents the coupon numbers of coupons issued to respondents
    degree : str
        The column name of the column in the data that represents the degree of the respondents
    zero_degree : str, optional
        This parameter is used to set the method for imputing zero values in the 'degree' variable. Three methods are available for selection: mean, median, and hotdeck. If this parameter is not set, the default imputation method is hotdeck
    NA_degree : str, optional
        This parameter is used to set the method for imputing missing values in the 'degree' variable. There are three methods to choose from: mean, median, and hotdeck. If this parameter is not set, the default method is hotdeck.

    Returns
    -------
    pandas.DataFrame
        A data frame with all original variables except ID and new RDS related information:

        ID : str
            Renamed unique_id variable
        R_CP : str
            Renamed redeemed coupon variable
        T_CP1 - T_CPn : str
            Renamed issued coupon variable
        DEGREE : original type
            Original degree variable
        DEGREE_IMP : float
            Imputed degree variable
        WEIGHT : float
            Weight variable calculated as 1/DEGREE_IMP
        WAVE : int
            Indicates which round the node was introduced into the survey, and the value of Seed is 0
        S_ID : str
            Indicates the ID of the seed corresponding to the node. The value of the seed is itself
        R_ID : str
            Indicates the ID of the node who recruits the node joining the survey. The value of seed is NA
        SEED : int
            Values are only 0 and 1, they are used to indicate whether the node is seed or not. If it is seed, the value is 1, if not, it is 0.
        CT_T_CP : int
            Indicates how many coupons have been issued to this node to invite others to join the survey
        CT_T_CP_USED : int
            Indicates how many coupons issued to this node have been used to invite others to join the survey.

    Examples
    --------
    # for preprocessing use RDStoydata
    rds_data = RDS_data(data = RDSToolsToyData,
                       unique_id = "ID",
                       redeemed_coupon = "CouponR",
                       issued_coupons = ["Coupon1",
                                         "Coupon2",
                                         "Coupon3"],
                       degree = "Degree")
    """
    # ---- 1. Data Preprocessing ----
    # Suppress FutureWarning about pandas replace() downcasting deprecation
    # In pandas 2.1+, replace() will stop automatically downcasting object types to numeric types
    # This warning appears even though we explicitly call infer_objects() which will continue
    # to handle the downcasting after the replace() behavior changes
    # Our code will work the same way before and after the pandas update
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        data = data.replace({
            "": pd.NA,
            "<NA>": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA
        }).infer_objects(copy=False)

    if data[unique_id].isna().any():
        raise ValueError(
            f"Function operation has been interrupted. "
            f"Please make sure there are no missing values in '{unique_id}' before trying again."
        )

    columns = [unique_id, redeemed_coupon] + issued_coupons + [degree]
    df_main = data[columns].copy()
    df_main = df_main.astype("string")

    issued_coupons_columns = [f"T_CP{i + 1}" for i in range(len(issued_coupons))]
    df_main.columns = ["ID", "R_CP"] + issued_coupons_columns + ["DEGREE"]

    unmodified_columns = data.drop(columns=columns, errors='ignore').copy()
    df_final = pd.concat([df_main, unmodified_columns], axis=1)

    # ---- 2. Add R_ID and SEED Columns ----
    coupon_columns = [col for col in df_final.columns if col.startswith("T_CP")]
    issued_coupons = df_final[coupon_columns].stack().dropna().unique()

    def find_recruiter(row):
        coupon_r = row["R_CP"]
        if pd.isna(coupon_r) or coupon_r not in issued_coupons:
            return pd.NA
        recruiter_mask = (df_final[coupon_columns] == coupon_r).any(axis=1)
        recruiter_row = df_final[recruiter_mask]
        if len(recruiter_row) == 1:
            return recruiter_row["ID"].iloc[0]
        else:
            return pd.NA

    df_final["R_ID"] = df_final.apply(find_recruiter, axis=1)
    df_final["SEED"] = df_final["R_ID"].isna().astype(int)

    # ---- 3. Calculate WAVE Column ----RDS_data
    df_final["WAVE"] = pd.NA
    df_final.loc[df_final["SEED"] == 1, "WAVE"] = 0

    current_wave = 0
    while df_final["WAVE"].isna().any():
        nodes_in_current_wave = df_final.loc[df_final["WAVE"] == current_wave, "ID"].tolist()
        if not nodes_in_current_wave:
            break
        df_final.loc[df_final["R_ID"].isin(nodes_in_current_wave) & df_final["WAVE"].isna(), "WAVE"] = current_wave + 1
        current_wave += 1

    # ---- 4. Assign S_ID Column ----
    df_final["S_ID"] = df_final["ID"].where(df_final["SEED"] == 1, pd.NA)

    while df_final["S_ID"].isna().any():
        df_final["S_ID"] = df_final["S_ID"].combine_first(
            df_final["R_ID"].map(df_final.set_index("ID")["S_ID"])
        )

    # ---- 5. Calculate Coupon Statistics Columns ----
    df_final["CT_T_CP"] = df_final[coupon_columns].notna().sum(axis=1)
    used_coupons = set(df_final["R_CP"].dropna())

    def count_used_coupons(row):
        return sum(coupon in used_coupons for coupon in row[coupon_columns] if pd.notna(coupon))

    df_final["CT_T_CP_USED"] = df_final.apply(count_used_coupons, axis=1)

    # ---- 6. Impute DEGREE Column ----
    if "DEGREE" not in df_final.columns:
        raise ValueError("Must have 'DEGREE' column")

    df_final["_DEGREE_ORIGINAL"] = df_final["DEGREE"]

    na_indices = df_final.index[df_final["DEGREE"].isna()].tolist()
    df_final["DEGREE"] = pd.to_numeric(df_final["DEGREE"], errors="coerce")
    zero_indices = df_final.index[df_final["DEGREE"] == 0].tolist()

    # Collect rows to drop
    rows_to_drop = []
    if zero_degree == "drop":
        rows_to_drop.extend(zero_indices)
    if NA_degree == "drop":
        rows_to_drop.extend(na_indices)

    # Drop rows if specified
    if rows_to_drop:
        rows_to_drop = list(set(rows_to_drop))  # Remove duplicates
        df_final = df_final.drop(rows_to_drop).reset_index(drop=True)
        print(f"Dropped {len(rows_to_drop)} rows")

        # Recalculate indices after dropping rows (for cases when imputation is selected for either)
        na_indices = df_final.index[df_final["DEGREE"].isna()].tolist()
        zero_indices = df_final.index[df_final["DEGREE"] == 0].tolist()

    valid_indices = df_final.index.difference(na_indices + zero_indices).tolist()
    valid_values = df_final.loc[valid_indices, "DEGREE"]

    def get_imputation_values(method, values, size):
        if method == "mean":
            return [values.mean()] * size
        elif method == "median":
            return [values.median()] * size
        elif method == "hotdeck":
            return np.random.choice(values, size=size, replace=True)
        elif method == "drop":
            return []  # No imputation needed for drop
        else:
            raise ValueError(f"Sorry, can not support your method: {method}")

    # Only impute if not dropping
    if zero_degree != "drop" and zero_indices:
        zero_imputation_values = get_imputation_values(zero_degree, valid_values, len(zero_indices))
        df_final.loc[zero_indices, "DEGREE"] = zero_imputation_values

    if NA_degree != "drop" and na_indices:
        na_imputation_values = get_imputation_values(NA_degree, valid_values, len(na_indices))
        df_final.loc[na_indices, "DEGREE"] = na_imputation_values

    # Create imputed degree column
    df_final["DEGREE_IMP"] = df_final["DEGREE"]
    df_final["WEIGHT"] = 1/df_final["DEGREE_IMP"]
    df_final["DEGREE"] = df_final["_DEGREE_ORIGINAL"]
    df_final.drop(columns=["_DEGREE_ORIGINAL"], inplace=True)

    return df_final
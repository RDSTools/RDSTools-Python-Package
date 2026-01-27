Data Processing
===============

The RDSdata function is the foundational function for processing respondent-driven sampling (RDS) survey data. It reconstructs recruitment chains, calculates wave numbers, identifies seeds, and imputes missing degree values by tracking how participants recruited one another through coupon redemption. **Use RDSdata before applying any estimation or plotting functions from the RDSTools package.**

Usage
-----

.. code-block:: python

    RDSdata(data, unique_id, redeemed_coupon, issued_coupons, degree, zero_degree="hotdeck", NA_degree="hotdeck")

Arguments
---------

**data**
    pandas.DataFrame. Should contain an ID variable for sample case, corresponding redeemed coupon code, and issued coupon code.

**unique_id**
    str. The column name of the column with respondent IDs.

**redeemed_coupon**
    str. The column name of the column with coupon codes redeemed by respondents when participating in the study.

**issued_coupons**
    list of str. The column name of the column with coupon codes issued to respondents (i.e., coupons given to respondents to recruit their peers). If multiple coupons are issued, list all coupon code variables.

**degree**
    str. The column name of the column with degree (i.e., network size) reported by respondents.

**zero_degree**
    str, optional. Used to set the method for handling zero values in the 'degree' variable. Three available methods are: mean imputation, median imputation, and hotdeck imputation. The default is hotdeck.

    * **mean**: Impute all positions that require imputation with the average value of all non-zero and non-missing values from the input degree.
    * **median**: Impute all positions that require imputation with the median value of all non-zero and non-missing values from the input degree.
    * **hotdeck**: For each position needing imputation, perform random sampling with replacement from all non-zero and non-missing values in the input degree, where each value has equal probability of being selected. The sampled value is then used as the imputed value.

**NA_degree**
    str, optional. Used to set the method for handling missing values in the 'degree' variable. Three available methods are: mean imputation, median imputation, and hotdeck imputation. The default is hotdeck.

    * **mean**: Impute all positions that require imputation with the average value of all non-zero and non-missing values from the input degree.
    * **median**: Impute all positions that require imputation with the median value of all non-zero and non-missing values from the input degree.
    * **hotdeck**: For each position needing imputation, perform random sampling with replacement from all non-zero and non-missing values in the input degree, where each value has equal probability of being selected. The sampled value is then used as the imputed value.

Returns
-------

**pandas.DataFrame**
    A data frame with all original variables, some renamed, and new RDS-related information:

    * **ID** (str): Renamed unique_id
    * **R_CP** (str): Renamed redeemed coupon ID
    * **T_CP1 - T_CPn** (str): Renamed issued_coupon
    * **DEGREE** (original type): Original degree variable
    * **DEGREE_IMP** (float): Degree variable with missing 0 and/or missing values treated
    * **WEIGHT** (float): Weight variable calculated as 1/DEGREE_IMP
    * **WAVE** (int): Indicates the wave a node was introduced into the data. The value of Seed is 0
    * **S_ID** (str): Indicates the ID of the seed corresponding to the node. For seeds, the value is the same as the value of ID
    * **R_ID** (str): Indicates the ID of the recruiter node. For seeds, the value is NA because there is no recruiter for seeds among respondents
    * **SEED** (int): Values are only 0 and 1, they are used to indicate whether the node is seed or not. If it is seed, the value is 1, if not, it is 0
    * **CP_ISSUED** (int): The count of issued coupons to the respondent
    * **CP_USED** (int): The count of used coupons (i.e., coupons redeemed by recruits) among the issued coupons

Example
-------

.. code-block:: python

    from RDSTools import load_toy_data, RDSdata

    # Using the built-in toy dataset
    data = load_toy_data()

    rds_data = RDSdata(
        data=data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree"
    )

    # With custom imputation methods
    rds_data = RDSdata(
        data=data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree",
        zero_degree="median",
        NA_degree="mean"
    )


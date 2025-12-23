Data Processing
===============

The RDSdata function is used to process data collected through Respondent Driven Sampling (RDS). This function can extract the unique ID, redeemed coupon numbers, and issued coupon numbers from the original dataset. By processing this information, users can obtain the key data typically required for RDS related research.

Usage
-----

.. code-block:: python

    RDSdata(data, unique_id, redeemed_coupon, issued_coupons, degree, zero_degree="hotdeck", NA_degree="hotdeck")

Arguments
---------

**data**
    A pandas.DataFrame, this data frame should contain ID numbers for the nodes in the social network and the corresponding redeemed coupon number and issued coupon number.

**unique_id**
    The column name of the column in the data that represents the ID numbers for the nodes in the social network.

**redeemed_coupon**
    The column name of the column in the data that represents coupon numbers of coupons redeemed by respondents when participating in the survey.

**issued_coupons**
    List of column names of the columns in the data that represent coupon numbers of coupons issued to respondents.

**degree**
    The column name of the column in the data that represents the degree (network size) of respondents.

**zero_degree**
    This parameter is used to set the method for imputing zero values in the 'degree' variable. Four methods are available for selection: mean, median, hotdeck, and drop. If this parameter is not set, the default imputation method is hotdeck.

**NA_degree**
    This parameter is used to set the method for imputing missing values in the 'degree' variable. There are four methods to choose from: mean, median, hotdeck, and drop. If this parameter is not set, the default method is hotdeck.

Returns
-------

**pandas.DataFrame**
    A data frame with all original variables except ID and new RDS related information:

    * **ID** (str): Renamed unique_id variable
    * **R_CP** (str): Renamed redeemed coupon variable
    * **T_CP1 - T_CPn** (str): Renamed issued coupon variable
    * **DEGREE** (original type): Original degree variable
    * **DEGREE_IMP** (float): Imputed degree variable
    * **WEIGHT** (float): Weight variable calculated as 1/DEGREE_IMP
    * **WAVE** (int): Indicates which round the node was introduced into the survey, and the value of Seed is 0
    * **S_ID** (str): Indicates the ID of the seed corresponding to the node. The value of the seed is itself
    * **R_ID** (str): Indicates the ID of the node who recruits the node joining the survey. The value of seed is NA
    * **SEED** (int): Values are only 0 and 1, they are used to indicate whether the node is seed or not. If it is seed, the value is 1, if not, it is 0.
    * **CT_T_CP** (int): Indicates how many coupons have been issued to this node to invite others to join the survey
    * **CT_T_CP_USED** (int): Indicates how many coupons issued to this node have been used to invite others to join the survey.

Example
-------

.. code-block:: python

    import pandas as pd
    from RDSTools import RDSdata

    # Load your data
    data = pd.read_csv("survey_data.csv")

    # Process RDS data
    rds_data = RDSdata(
        data=data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree",
        zero_degree='median',
        NA_degree='hotdeck'
    )

    # For preprocessing use RDStoydata (matching R example)
    rds_data = RDSdata(
        data=RDSToolsToyData,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree"
    )
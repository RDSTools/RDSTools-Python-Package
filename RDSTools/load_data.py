"""
Utility functions for loading example datasets.
"""

import pandas as pd
import os
from pathlib import Path


def load_toy_data():
    """
    Load the RDSTools toy dataset.

    This is a sample dataset that can be used for testing and examples.
    It contains sample RDS data with ID numbers, coupon codes, and degree information.

    Returns
    -------
    pandas.DataFrame
        The toy dataset with the following columns:
        - ID: Respondent ID
        - CouponR: Redeemed coupon code
        - Coupon1, Coupon2, Coupon3: Issued coupon codes
        - Degree: Network size
        - Other demographic variables

    Examples
    --------
    >>> from RDSTools import load_toy_data, RDSdata
    >>>
    >>> # Load the toy dataset
    >>> toy_data = load_toy_data()
    >>> print(f"Loaded {len(toy_data)} observations")
    >>>
    >>> # Process it with RDSdata
    >>> rds_data = RDSdata(
    ...     data=toy_data,
    ...     unique_id="ID",
    ...     redeemed_coupon="CouponR",
    ...     issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
    ...     degree="Degree"
    ... )
    """
    # Get the directory where this module is located
    current_dir = Path(__file__).parent

    # Look for the CSV file in the same directory
    csv_path = current_dir / 'RDSToolsToyData.csv'

    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        # Try to use pkg_resources for installed packages
        try:
            import pkg_resources
            data_path = pkg_resources.resource_filename(__name__, 'RDSToolsToyData.csv')
            return pd.read_csv(data_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find RDSToolsToyData.csv. "
                f"Make sure the CSV file is in the same directory as the RDSTools modules. "
                f"Looked in: {csv_path}"
            ) from e


def get_toy_data_path():
    """
    Get the path to the toy dataset file.

    Returns
    -------
    str
        Full path to the RDSToolsToyData.csv file.

    Examples
    --------
    >>> from RDSTools import get_toy_data_path
    >>> import pandas as pd
    >>>
    >>> # Get the path and load manually
    >>> path = get_toy_data_path()
    >>> print(f"Toy data located at: {path}")
    >>> data = pd.read_csv(path)
    """
    current_dir = Path(__file__).parent
    csv_path = current_dir / 'RDSToolsToyData.csv'

    if csv_path.exists():
        return str(csv_path)
    else:
        try:
            import pkg_resources
            return pkg_resources.resource_filename(__name__, 'RDSToolsToyData.csv')
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find RDSToolsToyData.csv. "
                f"Make sure the CSV file is in the same directory as the RDSTools modules."
            ) from e


# For backwards compatibility with variable-style access
try:
    RDSToolsToyData = load_toy_data()
except FileNotFoundError:
    # If file isn't available at import time, that's ok
    # User can still call load_toy_data() function later
    RDSToolsToyData = None
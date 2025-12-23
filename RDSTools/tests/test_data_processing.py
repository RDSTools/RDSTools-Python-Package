import pandas as pd
from RDSTools import RDSdata


def test_rds_data_works():
    # YOU write this test data
    data = pd.DataFrame({
        'ID': ['A', 'B', 'C'],
        'coupon': ['C1', None, 'C2'],
        'degree': [2, 3, 1]
    })

    # YOU call your function
    result = RDSdata(data, 'ID', 'coupon', [], 'degree')

    # YOU decide what to check
    assert len(result) == 3
    assert 'SEED' in result.columns
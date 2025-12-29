"""
Tests for RDSTools data processing functionality.

This test file demonstrates how to use both custom test data and the built-in
toy dataset for testing purposes.
"""

import pandas as pd
from RDSTools import RDSdata, load_toy_data


def test_rds_data_basic():
    """Test RDSdata with minimal custom data."""
    # Create simple test data
    data = pd.DataFrame({
        'ID': ['A', 'B', 'C'],
        'coupon': ['C1', None, 'C2'],
        'degree': [2, 3, 1]
    })

    # Process the data
    result = RDSdata(data, 'ID', 'coupon', [], 'degree')

    # Check basic requirements
    assert len(result) == 3, "Should have 3 rows"
    assert 'SEED' in result.columns, "Should have SEED column"
    assert 'WAVE' in result.columns, "Should have WAVE column"
    assert 'WEIGHT' in result.columns, "Should have WEIGHT column"

    print("✓ test_rds_data_basic passed")


def test_rds_data_with_toy_dataset():
    """Test RDSdata using the built-in toy dataset."""
    # Load the toy dataset
    toy_data = load_toy_data()

    # Verify toy data loaded correctly
    assert len(toy_data) > 0, "Toy data should not be empty"
    assert 'ID' in toy_data.columns, "Toy data should have ID column"

    # Process the toy data
    result = RDSdata(
        data=toy_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree"
    )

    # Verify processing worked
    assert len(result) > 0, "Processed data should not be empty"
    assert 'SEED' in result.columns, "Should have SEED column"
    assert 'WAVE' in result.columns, "Should have WAVE column"
    assert 'R_ID' in result.columns, "Should have R_ID column"
    assert 'S_ID' in result.columns, "Should have S_ID column"
    assert 'WEIGHT' in result.columns, "Should have WEIGHT column"
    assert 'DEGREE_IMP' in result.columns, "Should have DEGREE_IMP column"
    assert 'CP_ISSUED' in result.columns, "Should have CP_ISSUED column"
    assert 'CP_USED' in result.columns, "Should have CP_USED column"

    # Check for seeds
    num_seeds = result['SEED'].sum()
    assert num_seeds > 0, "Should have at least one seed"

    print(f"✓ test_rds_data_with_toy_dataset passed")
    print(f"  - Processed {len(result)} observations")
    print(f"  - Found {num_seeds} seeds")
    print(f"  - Max wave: {result['WAVE'].max()}")


def test_degree_imputation():
    """Test different degree imputation methods with toy data."""
    toy_data = load_toy_data()

    # Test different imputation methods
    methods = ['mean', 'median', 'hotdeck']

    for method in methods:
        result = RDSdata(
            data=toy_data,
            unique_id="ID",
            redeemed_coupon="CouponR",
            issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
            degree="Degree",
            zero_degree=method,
            NA_degree=method
        )

        # Check that imputation worked
        assert 'DEGREE_IMP' in result.columns, f"Should have DEGREE_IMP column with {method}"
        assert not result['DEGREE_IMP'].isna().any(), f"Should have no NA values with {method}"
        assert not (result['DEGREE_IMP'] == 0).any(), f"Should have no zero values with {method}"

        print(f"✓ Degree imputation with {method} passed")


def test_coupon_statistics():
    """Test that coupon statistics are calculated correctly."""
    toy_data = load_toy_data()

    result = RDSdata(
        data=toy_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree"
    )

    # Check coupon-related columns exist
    assert 'CP_ISSUED' in result.columns, "Should have CP_ISSUED column"
    assert 'CP_USED' in result.columns, "Should have CP_USED column"

    # Check that values make sense
    assert (result['CP_ISSUED'] >= 0).all(), "CP_ISSUED should be non-negative"
    assert (result['CP_USED'] >= 0).all(), "CP_USED should be non-negative"
    assert (result['CP_USED'] <= result['CP_ISSUED']).all(), "CP_USED should not exceed CP_ISSUED"

    print("✓ test_coupon_statistics passed")
    print(f"  - Average coupons issued: {result['CP_ISSUED'].mean():.2f}")
    print(f"  - Average coupons used: {result['CP_USED'].mean():.2f}")


def test_wave_calculation():
    """Test that wave calculation is correct."""
    toy_data = load_toy_data()

    result = RDSdata(
        data=toy_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree"
    )

    # Seeds should be wave 0
    seeds = result[result['SEED'] == 1]
    assert (seeds['WAVE'] == 0).all(), "Seeds should be in wave 0"

    # All waves should be consecutive integers starting from 0
    waves = sorted(result['WAVE'].unique())
    assert waves[0] == 0, "Waves should start at 0"
    for i in range(len(waves) - 1):
        assert waves[i + 1] == waves[i] + 1, "Waves should be consecutive"

    print("✓ test_wave_calculation passed")
    print(f"  - Waves: {waves}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60 + "\n")

    try:
        test_rds_data_basic()
        print()

        test_rds_data_with_toy_dataset()
        print()

        test_degree_imputation()
        print()

        test_coupon_statistics()
        print()

        test_wave_calculation()
        print()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()
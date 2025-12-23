"""
Example usage of the RDS map module using the RDSTools toy dataset.
These examples work with the included RDSToolsToyData.rda file.

QUICK START
===========
from RDSTools import (
    RDSdata,
    RDSmap,
    get_available_seeds,
    get_available_waves,
    print_map_info
)
import pandas as pd
import pyreadr

# Load toy data
data = pyreadr.read_r('RDSToolsToyData.rda')
data = pd.DataFrame(data['RDSToolsToyData'])

# Process
processed_data = RDSdata(
    data=data,
    unique_id="ID",
    redeemed_coupon="CouponR",
    issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
    degree="Degree",
    zero_degree="mean",
    NA_degree="mean"
)

# Explore what's available
print_map_info(processed_data)

# Get all seeds and waves
seeds = get_available_seeds(processed_data)
waves = get_available_waves(processed_data)

# Create and open map
RDSmap(processed_data, seeds, waves, open_browser=True)

===========
"""

import pandas as pd
from RDSTools import (
    RDSdata,
    RDSmap,
    get_available_seeds,
    get_available_waves,
    print_map_info
)

# For loading the .rda file (optional - can also use pre-exported CSV)
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    print("Note: Install pyreadr to load .rda files directly: pip install pyreadr")


def load_toy_data():
    """Load the RDSTools toy dataset."""
    if HAS_PYREADR:
        # Load directly from .rda file
        data = pyreadr.read_r('RDSToolsToyData.rda')
        data = data['RDSToolsToyData']
        data = pd.DataFrame(data)
    else:
        # Load from pre-exported CSV
        data = pd.read_csv('RDSToolsToyData.csv')

    return data


def basic_example():
    """
    Basic example using toy data.
    Shows how to discover available seeds/waves and create a map.
    """
    print("=" * 60)
    print("BASIC EXAMPLE: Discovering and Mapping All Data")
    print("=" * 60)

    # Load toy data
    raw_data = load_toy_data()
    print(f"Loaded {len(raw_data)} rows of raw data\n")

    # Process into RDS format
    processed_data = RDSdata(
        data=raw_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree",
        zero_degree="mean",
        NA_degree="mean"
    )

    # Print comprehensive information
    print_map_info(processed_data)

    # Get all available seeds and waves
    seeds = get_available_seeds(processed_data)
    waves = get_available_waves(processed_data)

    print(f"Creating map with {len(seeds)} seeds and {len(waves)} waves...\n")

    # Create map with all data
    m = RDSmap(
        data=processed_data,
        seed_ids=seeds,
        waves=waves,
        output_file='toy_data_full_map.html',
        open_browser=True
    )

    print("\n✓ Map created successfully!")
    return m


def single_seed_example():
    """
    Example focusing on a single seed and its recruitment chain.
    """
    print("\n" + "=" * 60)
    print("SINGLE SEED EXAMPLE: Tracking One Recruitment Chain")
    print("=" * 60)

    # Load and process data
    raw_data = load_toy_data()
    processed_data = RDSdata(
        data=raw_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree",
        zero_degree="mean",
        NA_degree="mean"
    )

    # Get available seeds and pick the first one
    seeds = get_available_seeds(processed_data)
    selected_seed = seeds[0]

    # Use all waves to see the full recruitment chain
    waves = get_available_waves(processed_data)

    print(f"Visualizing seed '{selected_seed}' across {len(waves)} waves\n")

    # Create map for single seed
    m = RDSmap(
        data=processed_data,
        seed_ids=[selected_seed],
        waves=waves,
        output_file='single_seed_map.html',
        open_browser=True
    )

    print("\n✓ Single seed map created!")
    return m


def first_three_waves_example():
    """
    Example showing only the first three waves of recruitment.
    """
    print("\n" + "=" * 60)
    print("EARLY WAVES EXAMPLE: First Three Waves Only")
    print("=" * 60)

    # Load and process data
    raw_data = load_toy_data()
    processed_data = RDSdata(
        data=raw_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree",
        zero_degree="mean",
        NA_degree="mean"
    )

    # Get all seeds but only first 3 waves
    seeds = get_available_seeds(processed_data)
    waves = [0, 1, 2, 3]  # Wave 0 (seeds) + first 3 recruitment waves

    print(f"Visualizing {len(seeds)} seeds across waves {waves}\n")

    # Create map
    m = RDSmap(
        data=processed_data,
        seed_ids=seeds,
        waves=waves,
        output_file='early_waves_map.html',
        open_browser=True
    )

    print("\n✓ Early waves map created!")
    return m


def multiple_seeds_example():
    """
    Example comparing specific seeds side by side.
    """
    print("\n" + "=" * 60)
    print("MULTIPLE SEEDS EXAMPLE: Comparing Selected Seeds")
    print("=" * 60)

    # Load and process data
    raw_data = load_toy_data()
    processed_data = RDSdata(
        data=raw_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree",
        zero_degree="mean",
        NA_degree="mean"
    )

    # Get available seeds
    all_seeds = get_available_seeds(processed_data)

    # Select first 3 seeds for comparison
    selected_seeds = all_seeds[:3]
    waves = get_available_waves(processed_data)

    print(f"Comparing seeds: {selected_seeds}")
    print(f"Across {len(waves)} waves\n")

    # Create map
    m = create_participant_map(
        data=processed_data,
        seed_ids=selected_seeds,
        waves=waves,
        output_file='multiple_seeds_map.html',
        open_browser=True
    )

    print("\n✓ Multi-seed comparison map created!")
    return m


def custom_coordinates_example():
    """
    Example showing how to handle data with custom coordinate column names.
    Note: The toy data uses 'Latitude' and 'Longitude' by default.
    """
    print("\n" + "=" * 60)
    print("CUSTOM COORDINATES EXAMPLE")
    print("=" * 60)

    # Load and process data
    raw_data = load_toy_data()
    processed_data = RDSdata(
        data=raw_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree",
        zero_degree="mean",
        NA_degree="mean"
    )

    # Check what coordinate columns are available
    print(f"Available columns: {', '.join(processed_data.columns)}\n")

    # Use the default 'Latitude' and 'Longitude' columns
    seeds = get_available_seeds(processed_data)
    waves = get_available_waves(processed_data)

    print(f"Using coordinate columns: 'Latitude' and 'Longitude'\n")

    # Create map specifying coordinate columns explicitly
    m = RDSmap(
        data=processed_data,
        seed_ids=seeds,
        waves=waves,
        lat_column='Latitude',  # Default, but shown explicitly
        lon_column='Longitude',  # Default, but shown explicitly
        output_file='custom_coords_map.html',
        open_browser=True
    )

    print("\n✓ Map created with custom coordinate specification!")
    return m


def explore_data_only():
    """
    Example that just explores the data without creating a map.
    Useful for understanding your dataset before visualization.
    """
    print("\n" + "=" * 60)
    print("DATA EXPLORATION ONLY")
    print("=" * 60)

    # Load and process data
    raw_data = load_toy_data()
    processed_data = RDSdata(
        data=raw_data,
        unique_id="ID",
        redeemed_coupon="CouponR",
        issued_coupons=["Coupon1", "Coupon2", "Coupon3"],
        degree="Degree",
        zero_degree="mean",
        NA_degree="mean"
    )

    # Print summary information
    print_map_info(processed_data)

    # Get seeds and waves programmatically
    seeds = get_available_seeds(processed_data)
    waves = get_available_waves(processed_data)

    # Additional analysis
    print("Detailed Breakdown:")
    print(f"  • Total seeds: {len(seeds)}")
    print(f"  • Seed IDs: {', '.join(seeds)}")
    print(f"  • Total waves: {len(waves)}")
    print(f"  • Wave range: {min(waves)} to {max(waves)}")
    print(f"  • Maximum recruitment depth: {max(waves)} waves from seeds")

    # Count participants per wave
    print("\nParticipants per wave:")
    for wave in waves:
        count = len(processed_data[processed_data['WAVE'] == wave])
        print(f"  Wave {wave}: {count} participants")

    # Count participants per seed
    print("\nParticipants per seed:")
    for seed in seeds:
        count = len(processed_data[processed_data['S_ID'] == seed])
        print(f"  Seed {seed}: {count} participants")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "=" * 60)
    print("RUNNING ALL EXAMPLES WITH TOY DATA")
    print("=" * 60)

    try:
        # 1. Exploration only
        explore_data_only()

        input("\nPress Enter to continue to basic example...")

        # 2. Basic full map
        basic_example()

        input("\nPress Enter to continue to single seed example...")

        # 3. Single seed
        single_seed_example()

        input("\nPress Enter to continue to early waves example...")

        # 4. First three waves
        first_three_waves_example()

        input("\nPress Enter to continue to multiple seeds example...")

        # 5. Multiple seeds comparison
        multiple_seeds_example()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        raise


if __name__ == '__main__':
    # You can run individual examples or all at once

    # Run just the exploration
    explore_data_only()

    # Or run a specific example
    # basic_example()
    # single_seed_example()
    # first_three_waves_example()
    # multiple_seeds_example()

    # Or run all examples
    # run_all_examples()
"""
RDS Map Module
Generates geographic visualizations of Respondent-Driven Sampling network data.
"""

import os
import webbrowser
import folium
import pandas as pd
from typing import List, Optional, Union


class TreeNode:
    """Represents a node in the RDS recruitment tree."""

    def __init__(self, node_id, wave, latitude=None, longitude=None):
        self.node_id = node_id
        self.wave = wave
        self.latitude = latitude
        self.longitude = longitude
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        """Add a child node to this node."""
        self.children.append(child_node)
        child_node.parent = self


def build_tree(edges, lat_key='Latitude', lon_key='Longitude'):
    """
    Build recruitment tree from edge data.

    Parameters
    ----------
    edges : list of dict
        List of dictionaries containing recruitment relationships
    lat_key : str, default 'Latitude'
        Column name for latitude coordinates
    lon_key : str, default 'Longitude'
        Column name for longitude coordinates

    Returns
    -------
    list of TreeNode
        Root nodes (seeds) of the recruitment trees
    """
    nodes = {}

    # Create all nodes
    for edge in edges:
        node_id = str(edge['ID'])
        wave = int(edge['WAVE'])
        lat = edge.get(lat_key)
        lon = edge.get(lon_key)

        if node_id not in nodes:
            nodes[node_id] = TreeNode(node_id, wave, lat, lon)

    # Build tree relationships
    root_nodes = []
    for edge in edges:
        node_id = str(edge['ID'])
        recruiter_id = edge.get('R_ID')

        if pd.isna(recruiter_id):
            # This is a seed node
            root_nodes.append(nodes[node_id])
        else:
            recruiter_id = str(recruiter_id)
            if recruiter_id in nodes:
                nodes[recruiter_id].add_child(nodes[node_id])

    return root_nodes


def traverse_tree(node, max_wave):
    """
    Traverse tree up to a maximum wave.

    Parameters
    ----------
    node : TreeNode
        Starting node for traversal
    max_wave : int
        Maximum wave to include in traversal

    Yields
    ------
    TreeNode
        Nodes in the tree up to max_wave
    """
    if node.wave <= max_wave:
        yield node
        for child in node.children:
            yield from traverse_tree(child, max_wave)


def RDSmap(
    data: pd.DataFrame,
    lat: str,
    long: str,
    seed_ids: Union[List[str], List[int]],
    waves: Optional[List[int]] = None,
    seed_color: str = "red",
    seed_radius: int = 7,
    recruit_color: str = "blue",
    recruit_radius: int = 7,
    line_color: str = "black",
    line_weight: int = 2,
    line_dashArray: Optional[str] = None,
    output_file: str = 'participant_map.html',
    zoom_start: int = 5,
    open_browser: bool = False
) -> folium.Map:
    """
    Mapping respondents in respondent driven sampling sample data overlaying with recruitment chains

    This function creates an interactive Folium map that displays RDS recruitment chains by plotting
    seed participants and their recruits with connecting lines.

    Parameters
    ----------
    data : pd.DataFrame
        The output from RDSdata with latitude and longitude coordinates per respondent
    lat : str
        Column name for latitude coordinates
    long : str
        Column name for longitude coordinates
    seed_ids : list of str or list of int
        List of seed IDs to display. Use get_available_seeds() to see available seeds.
    waves : list of int, optional
        List of wave numbers to display. If not specified, all available waves are included.
        Use get_available_waves() to see available waves, or specify explicitly like
        list(range(0, 4)) or [0, 1, 2, 3] for waves 0-3.
    seed_color : str, default "red"
        Color of seed circles
    seed_radius : int, default 7
        Size of seed circles
    recruit_color : str, default "blue"
        Color of recruit circles
    recruit_radius : int, default 7
        Size of recruit circles
    line_color : str, default "black"
        Color of lines connecting seeds and recruits
    line_weight : int, default 2
        Thickness of lines connecting seeds and recruits
    line_dashArray : str, optional
        Style of connecting lines (e.g., '5,6' for dashed lines)
    output_file : str, default 'participant_map.html'
        Name of the HTML file to save the map in current working directory
    zoom_start : int, default 5
        Initial zoom level for the map
    open_browser : bool, default False
        If True, automatically opens the map in default web browser after creation

    Returns
    -------
    folium.Map

        A map with seeds (in red circle markers) and participants (in blue circle markers) up to maximum availalble number of specified waves.
        Recruits from each seed are connected by edges.

    Raises
    ------
    ValueError
        If seed_ids or waves lists are empty
        If coordinate columns are not found
        If no valid coordinates are found

    Examples
    --------
    >>> import pandas as pd
    >>> from RDSTools import RDSdata, RDSmap, get_available_seeds, get_available_waves
    >>>
    >>> # Preprocess data with RDSdata function
    >>> rds_data = RDSdata(data = RDSToolsToyData,
    ...                     unique_id = "ID",
    ...                     redeemed_coupon = "CouponR",
    ...                     issued_coupon = ["Coupon1", "Coupon2", "Coupon3"],
    ...                     degree = "Degree")
    >>>
    >>> # Method 1: Simple example with all available waves (default)
    >>> out = RDSmap(rds_data,
    ...              lat="Latitude",
    ...              long="Longitude",
    ...              seed_ids=['1', '2'])
    >>> # Automatically uses all available waves
    >>>
    >>> # Method 2: Explicit seed IDs and waves
    >>> out = RDSmap(rds_data,
    ...              lat="Latitude",
    ...              long="Longitude",
    ...              seed_ids=['1', '2'],
    ...              waves=list(range(0, 3)))
    >>> # Map saved to participant_map.html
    >>>
    >>> # Method 3: Using get_available_seeds to select specific seeds
    >>> available_seeds = get_available_seeds(rds_data)
    >>> print(f"Available seeds: {available_seeds}")
    >>> # Available seeds: ['1', '2', '3', '4', '5']
    >>>
    >>> out = RDSmap(rds_data,
    ...              lat="Latitude",
    ...              long="Longitude",
    ...              seed_ids=available_seeds[:4],  # First 4 seeds
    ...              waves=list(range(0, 4)))       # Waves 0, 1, 2, 3
    >>>
    >>> # Method 4: Using get_available_waves to check and select waves
    >>> available_waves = get_available_waves(rds_data)
    >>> print(f"Available waves: {available_waves}")
    >>> # Available waves: [0, 1, 2, 3, 4, 5]
    >>>
    >>> out = RDSmap(rds_data,
    ...              lat="Latitude",
    ...              long="Longitude",
    ...              seed_ids=['1', '2', '3'],
    ...              waves=available_waves[:4])  # Use first 4 available waves
    >>>
    >>> # Method 5: Combine both helper functions for maximum flexibility
    >>> available_seeds = get_available_seeds(rds_data)
    >>> available_waves = get_available_waves(rds_data)
    >>>
    >>> out = RDSmap(rds_data,
    ...              lat="Latitude",
    ...              long="Longitude",
    ...              seed_ids=available_seeds[:2],    # First 2 seeds
    ...              waves=available_waves[1:4])      # Waves 1, 2, 3 (skip wave 0)
    >>>
    >>> # Method 6: Full customization with R-style colors and aesthetics
    >>> out = RDSmap(rds_data,
    ...              lat="Latitude",
    ...              long="Longitude",
    ...              seed_ids=['1', '2', '3', '4'],
    ...              waves=[0, 1, 2, 3],
    ...              seed_color="red",
    ...              seed_radius=5,
    ...              recruit_color="darkred",
    ...              recruit_radius=3,
    ...              line_color="black",
    ...              line_weight=5,
    ...              line_dashArray='5,6',
    ...              open_browser=True)  # Opens map in browser automatically
    """
    # Auto-populate waves if not specified
    if waves is None:
        waves = get_available_waves(data)
        print(f"Waves not specified, using all available waves: {waves}")

    # Input validation
    if not seed_ids or not waves:
        raise ValueError("seed_ids and waves must be non-empty lists")

    # Convert seed_ids to strings for consistency
    seed_ids = [str(sid) for sid in seed_ids]

    # Validate coordinate columns exist
    if lat not in data.columns or long not in data.columns:
        raise ValueError(
            f"Coordinate columns '{lat}' and/or '{long}' not found in data. "
            f"Available columns: {', '.join(data.columns)}"
        )

    # Validate coordinate columns contain numeric data
    if not pd.api.types.is_numeric_dtype(data[lat]) or \
       not pd.api.types.is_numeric_dtype(data[long]):
        raise ValueError(
            f"Columns '{lat}' and '{long}' must contain numeric data"
        )

    # Filter data by selected seed_ids (filter by S_ID to get all recruits from those seeds)
    seed_filtered_data = data[data['S_ID'].isin(seed_ids)].copy()

    if seed_filtered_data.empty:
        raise ValueError(
            f"No data found for the specified seed_ids: {seed_ids}"
        )

    # Get wave respondents from the seed-filtered data
    wave_respondents = seed_filtered_data[seed_filtered_data['WAVE'].isin(waves)].copy()

    # Filter to valid coordinates
    valid_data = wave_respondents.dropna(subset=[lat, long])
    valid_data = valid_data[
        (valid_data[lat] >= -90) &
        (valid_data[lat] <= 90) &
        (valid_data[long] >= -180) &
        (valid_data[long] <= 180)
    ]

    if valid_data.empty:
        raise ValueError(
            f"No valid geographic coordinates found in columns '{lat}' "
            f"and '{long}' for the specified waves"
        )

    # Calculate map center
    center_lat = valid_data[lat].mean()
    center_lon = valid_data[long].mean()

    # Create Folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )

    # Add all markers from wave_respondents
    for _, row in wave_respondents.iterrows():
        if pd.notna(row[lat]) and pd.notna(row[long]):
            # Check if this respondent is a seed (their ID is in seed_ids)
            is_seed = str(row['ID']) in seed_ids

            if is_seed:
                # This is a seed marker
                folium.CircleMarker(
                    location=[row[lat], row[long]],
                    radius=seed_radius,
                    color=seed_color,
                    fill=True,
                    fill_color=seed_color,
                    fill_opacity=0.8,
                    popup=f"Seed {row['S_ID']}"
                ).add_to(m)
            else:
                # This is a recruit marker
                folium.CircleMarker(
                    location=[row[lat], row[long]],
                    radius=recruit_radius,
                    color=recruit_color,
                    fill=True,
                    fill_color=recruit_color,
                    fill_opacity=0.8,
                    popup=f"Seed {row['S_ID']} - Respondent {row['ID']}"
                ).add_to(m)

    # Add lines connecting recruits to their recruiters
    # We need to connect each recruit to their recruiter (R_ID)
    for _, recruit_row in wave_respondents.iterrows():
        if pd.isna(recruit_row['R_ID']):
            # This is a seed, skip
            continue

        recruiter_id = recruit_row['R_ID']
        recruit_lat = recruit_row[lat]
        recruit_lon = recruit_row[long]

        if pd.isna(recruit_lat) or pd.isna(recruit_lon):
            continue

        # Find the recruiter in the data
        recruiter_rows = wave_respondents[wave_respondents['ID'] == recruiter_id]

        if not recruiter_rows.empty:
            recruiter_row = recruiter_rows.iloc[0]
            recruiter_lat = recruiter_row[lat]
            recruiter_lon = recruiter_row[long]

            if pd.notna(recruiter_lat) and pd.notna(recruiter_lon):
                folium.PolyLine(
                    locations=[
                        [recruiter_lat, recruiter_lon],
                        [recruit_lat, recruit_lon]
                    ],
                    color=line_color,
                    weight=line_weight,
                    opacity=0.7,
                    dash_array=line_dashArray
                ).add_to(m)

    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 70px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                border-radius: 5px;">
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: {seed_color}; width: 15px; height: 15px; 
                      border-radius: 50%; margin-right: 5px;"></div>
            <span>Seed</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="background-color: {recruit_color}; width: 15px; height: 15px; 
                      border-radius: 50%; margin-right: 5px;"></div>
            <span>Recruit</span>
        </div>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(legend_html))

    # Save to file
    m.save(output_file)
    print(f"Map saved to: {output_file}")

    # Open in browser if requested
    if open_browser:
        webbrowser.open('file://' + os.path.abspath(output_file))
        print(f"Opening map in browser...")

    return m


def get_available_seeds(data: pd.DataFrame) -> List[str]:
    """
    Get list of available seed IDs from RDS data.

    Parameters
    ----------
    data : pd.DataFrame
        RDS data processed by RDSdata function. Must contain 'S_ID' column.

    Returns
    -------
    list of str
        Sorted list of unique seed IDs in the dataset

    Raises
    ------
    ValueError
        If 'S_ID' column is not found in the data

    Examples
    --------
    >>> from rdstools import RDSdata, get_available_seeds
    >>> rds_data = RDSdata(raw_data, ...)
    >>> seeds = get_available_seeds(rds_data)
    >>> print(f"Available seeds: {seeds}")
    ['1', '2', '3', '4']
    """
    if 'S_ID' not in data.columns:
        raise ValueError(
            "Column 'S_ID' not found in data. "
            "Please ensure data has been processed with RDSdata function."
        )

    # Get unique seed IDs, convert to strings, and sort
    seeds = sorted(data['S_ID'].dropna().unique().astype(str))
    return seeds


def get_available_waves(data: pd.DataFrame) -> List[int]:
    """
    Get list of available wave numbers from RDS data.

    Parameters
    ----------
    data : pd.DataFrame
        RDS data processed by RDSdata function. Must contain 'WAVE' column.

    Returns
    -------
    list of int
        Sorted list of unique wave numbers in the dataset

    Raises
    ------
    ValueError
        If 'WAVE' column is not found in the data

    Examples
    --------
    >>> from rdstools import RDSdata, get_available_waves
    >>> rds_data = RDSdata(raw_data, ...)
    >>> waves = get_available_waves(rds_data)
    >>> print(f"Available waves: {waves}")
    [0, 1, 2, 3, 4, 5]
    """
    if 'WAVE' not in data.columns:
        raise ValueError(
            "Column 'WAVE' not found in data. "
            "Please ensure data has been processed with RDSdata function."
        )

    # Get unique waves, convert to int (handling numpy types), and sort
    waves = sorted([int(w) for w in data['WAVE'].dropna().unique()])
    return waves


def print_map_info(data: pd.DataFrame, lat_column: str = 'Latitude',
                   lon_column: str = 'Longitude') -> None:
    """
    Print summary information about the RDS data for mapping.

    This function displays available seeds, waves, and coordinate coverage
    to help users decide what to visualize.

    Parameters
    ----------
    data : pd.DataFrame
        RDS data processed by RDSdata function
    lat_column : str, default 'Latitude'
        Name of latitude column to check
    lon_column : str, default 'Longitude'
        Name of longitude column to check

    Examples
    --------
    >>> from rdstools import RDSdata, print_map_info
    >>> rds_data = RDSdata(raw_data, ...)
    >>> print_map_info(rds_data)

    RDS Mapping Information
    =======================
    Available Seeds: ['1', '2', '3', '4']
    Available Waves: [0, 1, 2, 3, 4, 5]

    Total participants: 250
    Participants with coordinates: 245 (98.0%)
    Participants missing coordinates: 5 (2.0%)

    Coordinate columns: Latitude, Longitude
    """
    print("\nRDS Mapping Information")
    print("=" * 50)

    # Seeds
    try:
        seeds = get_available_seeds(data)
        print(f"Available Seeds: {seeds}")
    except ValueError as e:
        print(f"Seeds: {e}")

    # Waves
    try:
        waves = get_available_waves(data)
        print(f"Available Waves: {waves}")
    except ValueError as e:
        print(f"Waves: {e}")

    print()

    # Total participants
    total = len(data)
    print(f"Total participants: {total}")

    # Coordinate coverage
    if lat_column in data.columns and lon_column in data.columns:
        valid_coords = data.dropna(subset=[lat_column, lon_column])
        valid_coords = valid_coords[
            (valid_coords[lat_column] >= -90) &
            (valid_coords[lat_column] <= 90) &
            (valid_coords[lon_column] >= -180) &
            (valid_coords[lon_column] <= 180)
        ]

        n_valid = len(valid_coords)
        n_missing = total - n_valid
        pct_valid = (n_valid / total * 100) if total > 0 else 0
        pct_missing = (n_missing / total * 100) if total > 0 else 0

        print(f"Participants with coordinates: {n_valid} ({pct_valid:.1f}%)")
        print(f"Participants missing coordinates: {n_missing} ({pct_missing:.1f}%)")
        print(f"\nCoordinate columns: {lat_column}, {lon_column}")
    else:
        missing = []
        if lat_column not in data.columns:
            missing.append(lat_column)
        if lon_column not in data.columns:
            missing.append(lon_column)
        print(f"Warning: Coordinate columns not found: {', '.join(missing)}")
        print(f"Available columns: {', '.join(data.columns)}")

    print("=" * 50 + "\n")
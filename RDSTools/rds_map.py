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


def get_available_seeds(data: pd.DataFrame) -> List[str]:
    """
    Get list of available seed IDs from RDS data.

    Parameters
    ----------
    data : pd.DataFrame
        RDS data processed by RDS_data function. Must contain 'S_ID' column.

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
    >>> from rdstools import RDS_data, get_available_seeds
    >>> rds_data = RDS_data(raw_data, ...)
    >>> seeds = get_available_seeds(rds_data)
    >>> print(f"Available seeds: {seeds}")
    ['1', '2', '3', '4']
    """
    if 'S_ID' not in data.columns:
        raise ValueError(
            "Column 'S_ID' not found in data. "
            "Please ensure data has been processed with RDS_data function."
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
        RDS data processed by RDS_data function. Must contain 'WAVE' column.

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
    >>> from rdstools import RDS_data, get_available_waves
    >>> rds_data = RDS_data(raw_data, ...)
    >>> waves = get_available_waves(rds_data)
    >>> print(f"Available waves: {waves}")
    [0, 1, 2, 3, 4, 5]
    """
    if 'WAVE' not in data.columns:
        raise ValueError(
            "Column 'WAVE' not found in data. "
            "Please ensure data has been processed with RDS_data function."
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
        RDS data processed by RDS_data function
    lat_column : str, default 'Latitude'
        Name of latitude column to check
    lon_column : str, default 'Longitude'
        Name of longitude column to check

    Examples
    --------
    >>> from rdstools import RDS_data, print_map_info
    >>> rds_data = RDS_data(raw_data, ...)
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


def RDSmap(
    data: pd.DataFrame,
    seed_ids: Union[List[str], List[int]],
    waves: List[int],
    lat_column: str = 'Latitude',
    lon_column: str = 'Longitude',
    output_file: str = 'participant_map.html',
    zoom_start: int = 5,
    open_browser: bool = False
) -> folium.Map:
    """
    Create a Folium map visualization of RDS participant locations.

    This function generates an interactive map showing the geographic distribution
    of study participants, with seed nodes highlighted and recruitment relationships
    shown when applicable.

    Parameters
    ----------
    data : pd.DataFrame
        RDS data processed by RDSdata function. Must contain columns:
        - ID: Unique participant identifier
        - S_ID: Seed ID for each participant
        - WAVE: Recruitment wave number
        - R_ID: Recruiter ID
        - Columns specified by lat_column and lon_column
    seed_ids : list of str or int
        List of seed IDs to include in visualization
    waves : list of int
        List of wave numbers to include in visualization
    lat_column : str, default 'Latitude'
        Name of column containing latitude coordinates
    lon_column : str, default 'Longitude'
        Name of column containing longitude coordinates
    output_file : str, default 'participant_map.html'
        Path to save HTML map file. Map is always saved to this file.
    zoom_start : int, default 5
        Initial zoom level for the map
    open_browser : bool, default False
        If True, automatically opens the map in default web browser after creation.

    Returns
    -------
    folium.Map
        Folium map object that has been saved to output_file.

    Raises
    ------
    ValueError
        If coordinate columns don't contain numeric data
        If no valid coordinates are found
        If no data matches the selection criteria

    Examples
    --------
    >>> import pandas as pd
    >>> from RDSTools import RDSdata, RDSmap
    >>>
    >>> # Load and process data
    >>> raw_data = pd.read_csv('survey_data.csv')
    >>> rds_data = RDSdata(raw_data, unique_id='ID', redeemed_coupon='Coupon',
    ...                      issued_coupons=['C1', 'C2', 'C3'], degree='NetworkSize')
    >>>
    >>> # Create and save map (default: participant_map.html)
    >>> m = RDSmap(rds_data, seed_ids=['1', '2'], waves=[0, 1, 2, 3])
    >>> # Map saved to 'participant_map.html' - open this file in your browser
    >>>
    >>> # Save to specific file
    >>> m = RDSmap(rds_data, seed_ids=['1', '2'], waves=[0, 1, 2, 3],
    ...                            output_file='my_map.html')
    >>>
    >>> # Create and open in browser automatically
    >>> m = RDSmap(rds_data, seed_ids=['1', '2'], waves=[0, 1, 2, 3],
    ...                            open_browser=True)
    """
    # Input validation
    if not seed_ids or not waves:
        raise ValueError("At least one seed ID and one wave must be specified")

    # Convert seed_ids to strings for consistency
    seed_ids = [str(sid) for sid in seed_ids]

    # Validate coordinate columns exist
    if lat_column not in data.columns or lon_column not in data.columns:
        raise ValueError(
            f"Coordinate columns '{lat_column}' and/or '{lon_column}' not found in data. "
            f"Available columns: {', '.join(data.columns)}"
        )

    # Validate coordinate columns contain numeric data
    if not pd.api.types.is_numeric_dtype(data[lat_column]) or \
       not pd.api.types.is_numeric_dtype(data[lon_column]):
        raise ValueError(
            f"Columns '{lat_column}' and '{lon_column}' must contain numeric data"
        )

    # Filter to valid geographic coordinates
    valid_data = data.dropna(subset=[lat_column, lon_column])

    # Basic geographical bounds check
    valid_data = valid_data[
        (valid_data[lat_column] >= -90) &
        (valid_data[lat_column] <= 90) &
        (valid_data[lon_column] >= -180) &
        (valid_data[lon_column] <= 180)
    ]

    if valid_data.empty:
        raise ValueError(
            f"No valid geographic coordinates found in columns '{lat_column}' "
            f"and '{lon_column}'. Coordinates must be in range "
            f"latitude: -90 to 90, longitude: -180 to 180"
        )

    # Filter by selected seeds
    seed_filtered = valid_data[valid_data['S_ID'].isin(seed_ids)]

    if seed_filtered.empty:
        raise ValueError(
            f"No data points found for the specified seed IDs: {', '.join(seed_ids)}"
        )

    # Check if waves are consecutive starting from 0
    is_consecutive_from_zero = (
        0 in waves and
        all(w in waves for w in range(min(waves), max(waves) + 1))
    )

    # Prepare nodes and edges data
    if is_consecutive_from_zero:
        # Use tree traversal for consecutive waves
        nodes_data, edges_data = _prepare_tree_data(
            seed_filtered, waves, seed_ids, lat_column, lon_column
        )
    else:
        # Use direct plotting for non-consecutive or single waves
        nodes_data, edges_data = _prepare_direct_data(
            seed_filtered, waves, seed_ids, lat_column, lon_column
        )

    if not nodes_data:
        raise ValueError("No valid nodes found with the specified criteria")

    # Create the map
    folium_map = _create_folium_map(nodes_data, edges_data, zoom_start)

    # Always save to file
    folium_map.save(output_file)
    print(f"Map saved to: {output_file}")

    # Open in browser if requested
    if open_browser:
        webbrowser.open('file://' + os.path.abspath(output_file))
        print(f"Opening map in browser...")

    return folium_map


def _prepare_tree_data(data, waves, seed_ids, lat_column, lon_column):
    """Prepare node and edge data using tree traversal."""
    nodes_data = []
    edges_data = []

    # Build the tree from seed-filtered data
    edges = data.to_dict('records')
    root_nodes = build_tree(edges, lat_key=lat_column, lon_key=lon_column)

    max_wave = max(waves)

    for root_node in root_nodes:
        for node in traverse_tree(root_node, max_wave):
            # Only include nodes in selected waves with valid coordinates
            if (node.wave in waves and
                node.latitude is not None and
                node.longitude is not None):

                # Add node data
                nodes_data.append({
                    'id': node.node_id,
                    'lat': node.latitude,
                    'lon': node.longitude,
                    'is_seed': node.node_id in seed_ids,
                    'wave': node.wave
                })

                # Add edges for each child
                for child in node.children:
                    if (child.wave in waves and
                        child.latitude is not None and
                        child.longitude is not None):

                        edges_data.append({
                            'from_id': node.node_id,
                            'to_id': child.node_id,
                            'from_lat': node.latitude,
                            'from_lon': node.longitude,
                            'to_lat': child.latitude,
                            'to_lon': child.longitude
                        })

    return nodes_data, edges_data


def _prepare_direct_data(data, waves, seed_ids, lat_column, lon_column):
    """Prepare node data for direct plotting (no edges)."""
    # Filter by selected waves
    df = data[data['WAVE'].isin(waves)]

    if df.empty:
        return [], []

    nodes_data = []
    for _, row in df.iterrows():
        # Check if this ID is a seed
        is_seed = str(row['ID']) in seed_ids

        nodes_data.append({
            'id': row['ID'],
            'lat': row[lat_column],
            'lon': row[lon_column],
            'is_seed': is_seed,
            'wave': row['WAVE']
        })

    # No edges for non-consecutive wave selections
    return nodes_data, []


def _create_folium_map(nodes, edges, zoom_start):
    """Create the actual Folium map visualization."""
    # Calculate mean coordinates for map centering
    lat_coords = [node['lat'] for node in nodes]
    lon_coords = [node['lon'] for node in nodes]

    center_lat = sum(lat_coords) / len(lat_coords)
    center_lon = sum(lon_coords) / len(lon_coords)

    # Create Folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )

    # Add nodes to the map
    for node in nodes:
        is_seed = node['is_seed']
        color = 'red' if is_seed else 'blue'

        folium.CircleMarker(
            location=[node['lat'], node['lon']],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"ID: {node['id']}<br>Wave: {node['wave']}<br>"
                  f"Seed: {'Yes' if is_seed else 'No'}"
        ).add_to(m)

    # Add edges (recruitment relationships)
    for edge in edges:
        points = [
            [edge['from_lat'], edge['from_lon']],
            [edge['to_lat'], edge['to_lon']]
        ]

        folium.PolyLine(
            locations=points,
            color='black',
            weight=2,
            opacity=0.7
        ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 70px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                border-radius: 5px;">
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: red; width: 15px; height: 15px; 
                      border-radius: 50%; margin-right: 5px;"></div>
            <span>Seed</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="background-color: blue; width: 15px; height: 15px; 
                      border-radius: 50%; margin-right: 5px;"></div>
            <span>Non-seed</span>
        </div>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(legend_html))

    return m
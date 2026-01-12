Visualization
=============

The RDS Tools package supports visualization of respondents' networks and the geographic distribution of recruitment waves starting from seeds. Users can generate network plots to examine recruitment chains overall and by some characteristic, as well as geographic maps that display participant locations and the spread of recruitment over time or across regions. These visualizations aid in understanding the structure of chains and the geographic reach of RDS studies.

RDSnetgraph - Recruitment Network Visualization
===============================================

The RDSnetgraph function creates network visualizations showing recruitment relationships between participants. This function supports multiple layout algorithms and customization options for coloring nodes by demographic variables.

Usage
-----

.. code-block:: python

    RDSnetgraph(data, seed_ids, waves, layout='Spring', variable=None, category_colors=None,
                vertex_size=6, vertex_size_seed=10, seed_color='#E41A1C', nonseed_color='#377EB8',
                edge_width=1.0, title=None, figsize=(12, 10), show_plot=True, save_path=None)

Arguments
---------

**data**
    pandas.DataFrame. The output from RDSdata function containing preprocessed RDS data with recruitment relationships.

**seed_ids**
    list of str or list of int. List of seed IDs to include in the network visualization. These should match the IDs in the 'S_ID' column of the data.

**waves**
    list of int. List of wave numbers to display in the network. Wave 0 represents seeds. Use list(range(0, n)) to include waves 0 through n-1.

**layout**
    str, optional. Specifies the network layout algorithm to use. Default is 'Spring'. Available options:

    * **Spring**: Force-directed layout (default, uses igraph Fruchterman-Reingold algorithm). Good for general network visualization.
    * **Tree**: Hierarchical tree layout (requires pygraphviz, uses NetworkX). Best for visualizing recruitment chains as a tree structure.
    * **Circular**: Circular arrangement of nodes. Good for networks with cyclical patterns.
    * **Kamada-Kawai**: Force-directed layout with uniform edge lengths. Provides more uniform spacing than Spring.
    * **Grid**: Grid-based layout. Useful for ordered data.
    * **Star**: Star-shaped layout. Centers one node with others radiating outward.
    * **Random**: Random positioning of nodes. Useful for comparison or testing.

**variable**
    str, optional. Name of a categorical variable in the data to color nodes by. When specified, nodes will be colored according to their category values. Variables with 10+ categories trigger a warning as they may be hard to interpret. Default is None (uses seed/nonseed coloring).

**category_colors**
    list of str, optional. Custom colors for categories when using the variable parameter. Must provide exactly one color per category in the variable. Colors are assigned to categories in sorted alphabetical/numerical order. Accepts hex codes (e.g., '#FF0000') or named colors (e.g., 'red'). If not provided, uses the default 20-color palette. Default is None.

**vertex_size**
    int or float, optional. Size of non-seed vertices (nodes) in the network graph. Default is 6.

**vertex_size_seed**
    int or float, optional. Size of seed vertices (nodes) in the network graph. Seeds are typically displayed larger to distinguish them. Default is 10.

**seed_color**
    str, optional. Color for seed nodes when not using the variable parameter for grouping. Accepts hex codes or named colors. Default is '#E41A1C' (red).

**nonseed_color**
    str, optional. Color for non-seed nodes when not using the variable parameter for grouping. Accepts hex codes or named colors. Default is '#377EB8' (blue).

**edge_width**
    float, optional. Width of edges (lines) connecting nodes in the recruitment network. Default is 1.0.

**title**
    str, optional. Title for the network graph. If not provided, a default title is automatically generated showing the seeds and waves included.

**figsize**
    tuple of (int, int), optional. Figure size in inches as (width, height). Default is (12, 10).

**show_plot**
    bool, optional. If True, displays the plot in the current environment. If False, the plot is not shown but can still be saved using save_path. Default is True.

**save_path**
    str, optional. File path to save the network graph image. Supports common image formats (.png, .pdf, .svg, .jpg). If None, the graph is not saved to file. Default is None.

Returns
-------

**Graph object**
    Returns either an igraph.Graph object (for Spring, Circular, Kamada-Kawai, Grid, Star, Random layouts) or a networkx.DiGraph object (for Tree layout). The returned graph object contains all nodes, edges, and attributes and can be further manipulated or analyzed.

Examples
--------

.. code-block:: python

    from RDSTools import RDSnetgraph

    # Basic network graph with Spring layout
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        layout='Spring'
    )

    # Tree layout showing hierarchical recruitment structure
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2],
        layout='Tree'
    )

    # Color nodes by demographic variable
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2],
        layout='Spring',
        variable='Sex',
        title='Recruitment by Sex',
        vertex_size_seed=10,
        vertex_size=6,
        figsize=(14, 12)
    )

    # Use custom colors for categories
    # Colors must match the number of categories in sorted alphabetical/numerical order
    custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # For 3 categories

    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2],
        variable='Race',  # Assuming Race has 3 categories
        category_colors=custom_colors,
        title='Recruitment by Race (Custom Colors)'
    )

    # Customize colors when not grouping by variable
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2],
        seed_color='purple',
        nonseed_color='orange',
        edge_width=2.0
    )

    # Save network graph without displaying
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1'],
        waves=[0, 1, 2, 3, 4],
        layout='Tree',
        save_path='recruitment_tree.png',
        show_plot=False
    )

Color Customization for Network Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the ``variable`` parameter to color nodes by categories:

**Default Color Palette:**
    - A 20-color palette is automatically applied
    - Colors 1-8: Strong, distinct colors (Set1 palette)
    - Colors 9-16: Muted, professional colors (Dark2 palette)
    - Colors 17-20: Soft, light colors (Pastel1 palette)
    - For 20+ categories, colors recycle (with a warning)

**Custom Colors:**
    - Use ``category_colors`` parameter with a list of color codes
    - Must provide exactly one color per category
    - Colors apply in sorted alphabetical/numerical order of categories
    - Accepts hex codes (e.g., '#FF0000') or named colors (e.g., 'red')

**Important Notes:**
    - Variables with 10+ categories trigger a warning (may be hard to interpret)
    - Variables with 20+ categories recycle colors and strongly suggest custom colors
    - Categories are always sorted before color assignment for consistency
    - To check category order: ``sorted(data[variable].dropna().unique())``

RDSmap - Geographic Distribution Mapping
=========================================

When longitude and latitude are available, users can plot distribution of recruitment overall or for each wave. The RDSmap function allows explicit control of the number of waves and seeds in the plot. If waves are not specified, all available waves are automatically included.

Usage
-----

.. code-block:: python

    RDSmap(data, lat, long, seed_ids, waves=None, seed_color="red", seed_radius=7,
           recruit_color="blue", recruit_radius=7, line_color="black", line_weight=2,
           line_dashArray=None, output_file='participant_map.html', zoom_start=5,
           open_browser=False)

Arguments
---------

**data**
    pandas.DataFrame. The output from RDSdata function containing preprocessed RDS data with latitude and longitude coordinates per respondent.

**lat**
    str. Column name for latitude coordinates in the data. Values should be numeric, in the range [-90, 90].

**long**
    str. Column name for longitude coordinates in the data. Values should be numeric, in the range [-180, 180].

**seed_ids**
    list of str or list of int. List of seed IDs to display on the map. Use get_available_seeds() to see available seeds in the dataset.

**waves**
    list of int, optional. List of wave numbers to display on the map. If not specified, all available waves are automatically included. Wave 0 represents seeds. Use get_available_waves() to see available waves, or specify explicitly like list(range(0, 4)) or [0, 1, 2, 3] for waves 0-3. Default is None (all waves).

**seed_color**
    str, optional. Color of seed circle markers on the map. Accepts standard CSS color names (e.g., 'red', 'blue', 'green') or hex codes (e.g., '#FF0000'). Default is 'red'.

**seed_radius**
    int, optional. Radius (size) of seed circle markers in pixels. Larger values create bigger circles. Default is 7.

**recruit_color**
    str, optional. Color of recruit circle markers on the map. Accepts standard CSS color names or hex codes. Default is 'blue'.

**recruit_radius**
    int, optional. Radius (size) of recruit circle markers in pixels. Default is 7.

**line_color**
    str, optional. Color of lines connecting recruiters to recruits, showing recruitment relationships. Accepts standard CSS color names or hex codes. Default is 'black'.

**line_weight**
    int, optional. Thickness (width) of lines connecting recruiters to recruits in pixels. Default is 2.

**line_dashArray**
    str, optional. Dash pattern for lines connecting recruiters to recruits. Format is a string of comma-separated numbers representing dash and gap lengths (e.g., '5,6' creates dashed lines with 5-pixel dashes and 6-pixel gaps). If None, solid lines are used. Default is None.

**output_file**
    str, optional. Name of the HTML file to save the interactive map. The file is saved in the current working directory. Default is 'participant_map.html'.

**zoom_start**
    int, optional. Initial zoom level for the map. Lower values show more area (zoomed out), higher values show less area (zoomed in). Typical range is 1-18. Default is 5.

**open_browser**
    bool, optional. If True, automatically opens the generated map in the default web browser after creation. If False, the map is saved but not opened. Default is False.

Returns
-------

**folium.Map**
    A Folium map object containing the interactive visualization. The map shows:

    * Seed locations as circle markers (red by default)
    * Non-seed locations as circle markers (blue by default)
    * Lines connecting recruiters to recruits for consecutive waves
    * Interactive popups with participant details (ID, seed ID)
    * A legend showing seed and recruit marker types

Raises
------

**ValueError**
    If seed_ids or waves lists are empty, if coordinate columns are not found in the data, or if no valid coordinates are found.

Examples
--------

.. code-block:: python

    from RDSTools import RDSmap, get_available_seeds, get_available_waves, print_map_info

    # Check available data
    print_map_info(rds_data, lat='Latitude', long='Longitude')

    # Get available seeds and waves
    seeds = get_available_seeds(rds_data)
    waves = get_available_waves(rds_data)

    print(f"Available seeds: {seeds}")
    print(f"Available waves: {waves}")

    # Simplest map - uses all available waves automatically
    m = RDSmap(
        data=rds_data,
        seed_ids=['1', '2'],
        lat='Latitude',
        long='Longitude',
        output_file='recruitment_map.html'
    )

    # Create map with specific waves
    m = RDSmap(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        lat='Latitude',
        long='Longitude',
        output_file='recruitment_map.html'
    )

    # Custom map with specific zoom and auto-open
    m = RDSmap(
        data=rds_data,
        seed_ids=['1', '2', '3'],
        waves=[0, 1, 2, 3, 4],
        lat='Latitude',
        long='Longitude',
        output_file='custom_map.html',
        zoom_start=10,
        open_browser=True
    )

    # Customize marker colors and sizes
    m = RDSmap(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=list(range(0, 4)),
        lat='Latitude',
        long='Longitude',
        seed_color='red',
        seed_radius=7,
        recruit_color='blue',
        recruit_radius=7,
        line_color='black',
        line_weight=2,
        line_dashArray='5,6',  # Dashed lines
        output_file='styled_map.html'
    )

    # Use helper functions to explore data
    available_seeds = get_available_seeds(rds_data)
    available_waves = get_available_waves(rds_data)

    m = RDSmap(
        data=rds_data,
        seed_ids=available_seeds[:2],    # First 2 seeds
        waves=available_waves[:4],       # First 4 waves
        lat='Latitude',
        long='Longitude'
    )

Helper Functions
----------------

**get_available_seeds(data)**
    Get list of available seed IDs from RDS data.

    Parameters:
        - data (pandas.DataFrame): RDS data processed by RDSdata function

    Returns:
        - list of str: Sorted list of unique seed IDs

**get_available_waves(data)**
    Get list of available wave numbers from RDS data.

    Parameters:
        - data (pandas.DataFrame): RDS data processed by RDSdata function

    Returns:
        - list of int: Sorted list of unique wave numbers

**print_map_info(data, lat='Latitude', long='Longitude')**
    Print summary information about the RDS data for mapping, including available seeds, waves, and coordinate coverage.

    Parameters:
        - data (pandas.DataFrame): RDS data processed by RDSdata function
        - lat (str): Name of latitude column to check
        - long (str): Name of longitude column to check

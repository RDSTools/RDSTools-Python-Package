Visualization
=============

The RDS Tools package supports visualization of respondents' networks and the geographic distribution of recruitment waves starting from seeds. Users can generate network plots to examine recruitment chains overall and by some characteristic, as well as geographic maps that display participant locations and the spread of recruitment over time or across regions. These visualizations aid in understanding the structure of chains and the geographic reach of RDS studies.

Recruitment Networks
--------------------

The RDSnetgraph function creates network visualizations showing recruitment relationships between participants.

.. code-block:: python

    from RDSTools import RDSnetgraph

    # Basic network graph
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        layout='Spring'
    )

The function supports different layout algorithms:

- **Spring**: Force-directed layout (default, uses igraph)
- **Tree**: Hierarchical tree layout (requires pygraphviz, uses NetworkX)
- **Circular**: Circular arrangement
- **Kamada-Kawai**: Force-directed with uniform edge lengths
- **Grid**: Grid-based layout
- **Star**: Star-shaped layout
- **Random**: Random positioning

You can color nodes by demographic variables:

.. code-block:: python

    # Color nodes by demographic variable
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2],
        layout='Spring',
        group_by='Sex',
        node_size=50,
        figsize=(14, 12)
    )

You can save the network graph to a file:

.. code-block:: python

    # Save network graph
    G = RDSnetgraph(
        data=rds_data,
        seed_ids=['1'],
        waves=[0, 1, 2, 3, 4],
        layout='Tree',
        save_path='recruitment_tree.png',
        show_plot=False
    )

Mapping
-------

When longitude and latitude are available, users can plot distribution of recruitment overall or for each wave. The RDSmap function allows explicit control of the number of waves and seeds in the plot.

.. code-block:: python

    from RDSTools import RDSmap, get_available_seeds, get_available_waves, print_map_info

    # Check available data
    print_map_info(rds_data, lat_column='Latitude', lon_column='Longitude')

    # Get available seeds and waves
    seeds = get_available_seeds(rds_data)
    waves = get_available_waves(rds_data)

    print(f"Available seeds: {seeds}")
    print(f"Available waves: {waves}")

    # Create map
    m = RDSmap(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        lat_column='Latitude',
        lon_column='Longitude',
        output_file='recruitment_map.html'
    )

The mapping function creates interactive HTML maps showing:

- Seed locations (red markers)
- Non-seed locations (blue markers)
- Recruitment relationships (connecting lines for consecutive waves)
- Interactive popups with participant details

You can customize the map display:

.. code-block:: python

    # Custom map with specific zoom and auto-open
    m = RDSmap(
        data=rds_data,
        seed_ids=['1', '2', '3'],
        waves=[0, 1, 2, 3, 4],
        lat_column='lat',
        lon_column='long',
        output_file='custom_map.html',
        zoom_start=10,
        open_browser=True
    )

Helper Functions
----------------

**get_available_seeds(data)**
    Get list of available seed IDs from RDS data.

**get_available_waves(data)**
    Get list of available wave numbers from RDS data.

**print_map_info(data, lat_column='Latitude', lon_column='Longitude')**
    Print summary information about the RDS data for mapping, including available seeds, waves, and coordinate coverage.
Visualization
=============

The RDS Tools package supports visualization of respondents' networks and the geographic distribution of recruitment waves starting from seeds. Users can generate network plots to examine recruitment chains overall and by some characteristic, as well as geographic maps that display participant locations and the spread of recruitment over time or across regions. These visualizations aid in understanding the structure of chains and the geographic reach of RDS studies.

Recruitment Networks
--------------------

The create_network_graph function creates network visualizations showing recruitment relationships between participants.

.. code-block:: python

    from RDSTools.network_graph import create_network_graph

    # Basic network graph
    G = create_network_graph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2, 3],
        layout='Spring'
    )

The function supports different layout algorithms:

- **Spring**: Force-directed layout
- **Tree**: Hierarchical tree layout
- **Circular**: Circular arrangement
- **Kamada-Kawai**: Force-directed with uniform edge lengths
- **Grid**: Grid-based layout
- **Star**: Star-shaped layout
- **Random**: Random positioning

You can color nodes by demographic variables:

.. code-block:: python

    # Color nodes by demographic variable
    G = create_network_graph(
        data=rds_data,
        seed_ids=['1', '2'],
        waves=[0, 1, 2],
        layout='Spring',
        group_by='Sex'
    )

Mapping
-------

When longitude and latitude are available, users can plot distribution of recruitment overall or for each wave. The create_participant_map function allows explicit control of the number of waves and seeds in the plot.

.. code-block:: python

    from RDSTools.rds_map import create_participant_map, print_map_info

    # Check available data
    print_map_info(rds_data, lat_column='Latitude', lon_column='Longitude')

    # Create map
    m = create_participant_map(
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
- Recruitment relationships (connecting lines)
- Interactive popups with participant details

You can customize the map display:

.. code-block:: python

    # Custom map with specific zoom and auto-open
    m = create_participant_map(
        data=rds_data,
        seed_ids=['1', '2', '3'],
        waves=[0, 1, 2, 3, 4],
        lat_column='lat',
        lon_column='long',
        output_file='custom_map.html',
        zoom_start=10,
        open_browser=True
    )
"""
Network graph visualization for RDS data
"""
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # For saving to files
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import igraph as ig
from typing import List, Optional, Union, Literal, Dict, Any
import warnings


# ============================================================================
# Tree Building Classes and Functions
# ============================================================================

class TreeNode:
    """Represents a node in the RDS recruitment tree"""
    def __init__(self, node_id, is_seed=False, wave=None, seed_id=None):
        self.id = node_id
        self.is_seed = is_seed
        self.wave = wave
        self.seed_id = seed_id
        self.children = []
        self.parent = None
        self.data = {}

    def add_child(self, child_node):
        """Add a child node"""
        self.children.append(child_node)
        child_node.parent = self


def build_tree(edges: List[Dict[str, Any]]) -> List[TreeNode]:
    """
    Build a tree structure from RDS edge data.

    Parameters
    ----------
    edges : List[Dict]
        List of edge dictionaries with keys: ID, R_ID, S_ID, WAVE

    Returns
    -------
    List[TreeNode]
        List of root nodes (seeds) in the recruitment tree
    """
    nodes = {}
    root_nodes = []

    # First pass: create all nodes
    for edge in edges:
        node_id = edge.get('ID')
        recruiter_id = edge.get('R_ID')
        seed_id = edge.get('S_ID')
        wave = edge.get('WAVE')

        if node_id not in nodes:
            # Check if this is a seed (wave 0 or R_ID is NaN/None)
            is_seed = (wave == 0) or (recruiter_id is None) or (str(recruiter_id).lower() == 'nan')

            node = TreeNode(
                node_id=node_id,
                is_seed=is_seed,
                wave=wave,
                seed_id=seed_id
            )
            node.data = edge.copy()
            nodes[node_id] = node

            if is_seed:
                root_nodes.append(node)

    # Second pass: establish parent-child relationships
    for edge in edges:
        node_id = edge.get('ID')
        recruiter_id = edge.get('R_ID')

        if recruiter_id and str(recruiter_id).lower() != 'nan':
            if recruiter_id in nodes and node_id in nodes:
                parent_node = nodes[recruiter_id]
                child_node = nodes[node_id]
                parent_node.add_child(child_node)

    return root_nodes


def create_networkx_graph(root_nodes: List[TreeNode], df) -> nx.DiGraph:
    """
    Create a NetworkX directed graph from tree nodes.

    Parameters
    ----------
    root_nodes : List[TreeNode]
        List of root nodes from build_tree
    df : pd.DataFrame
        Original data frame for additional node attributes

    Returns
    -------
    nx.DiGraph
        NetworkX directed graph
    """
    G = nx.DiGraph()

    def add_node_and_edges(node: TreeNode):
        """Recursively add nodes and edges to the graph"""
        # Add node with attributes
        G.add_node(
            node.id,
            is_seed=node.is_seed,
            wave=node.wave,
            seed_id=node.seed_id
        )

        # Add edges to children
        for child in node.children:
            G.add_edge(node.id, child.id)
            add_node_and_edges(child)

    # Process each root node
    for root in root_nodes:
        add_node_and_edges(root)

    return G


def create_igraph_graph(root_nodes: List[TreeNode], df) -> ig.Graph:
    """
    Create an igraph directed graph from tree nodes.

    Parameters
    ----------
    root_nodes : List[TreeNode]
        List of root nodes from build_tree
    df : pd.DataFrame
        Original data frame for additional node attributes

    Returns
    -------
    ig.Graph
        igraph directed graph
    """
    # Collect all nodes and edges
    all_nodes = []
    edges = []
    node_attributes = {
        'is_seed': [],
        'wave': [],
        'seed_id': []
    }

    def collect_nodes_and_edges(node: TreeNode):
        """Recursively collect nodes and edges"""
        if node.id not in all_nodes:
            all_nodes.append(node.id)
            node_attributes['is_seed'].append(node.is_seed)
            node_attributes['wave'].append(node.wave)
            node_attributes['seed_id'].append(node.seed_id)

        for child in node.children:
            edges.append((node.id, child.id))
            collect_nodes_and_edges(child)

    # Process all root nodes
    for root in root_nodes:
        collect_nodes_and_edges(root)

    # Create igraph
    G = ig.Graph(directed=True)

    # Add vertices
    G.add_vertices(len(all_nodes))
    G.vs["name"] = all_nodes
    G.vs["is_seed"] = node_attributes['is_seed']
    G.vs["wave"] = node_attributes['wave']
    G.vs["seed_id"] = node_attributes['seed_id']

    # Add edges (convert node IDs to indices)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(all_nodes)}
    edge_indices = [(node_to_idx[src], node_to_idx[dst]) for src, dst in edges]
    G.add_edges(edge_indices)

    return G


# ============================================================================
# Color Palette Management
# ============================================================================

def get_default_color_palette():
    """
    Get the default color palette for categorical variables.

    Returns an extended 20-color palette combining multiple ColorBrewer palettes.
    Colors are ordered as follows:

    Positions 1-8: Set1 palette (strong, distinct colors)
    - 1. Red (#E41A1C)
    - 2. Blue (#377EB8)
    - 3. Green (#4DAF4A)
    - 4. Purple (#984EA3)
    - 5. Orange (#FF7F00)
    - 6. Yellow (#FFFF33)
    - 7. Brown (#A65628)
    - 8. Pink (#F781BF)

    Positions 9-16: Dark2 palette (muted, professional colors)
    - 9. Teal (#1B9E77)
    - 10. Dark Orange (#D95F02)
    - 11. Olive (#7570B3)
    - 12. Magenta (#E7298A)
    - 13. Lime (#66A61E)
    - 14. Gold (#E6AB02)
    - 15. Cyan (#A6761D)
    - 16. Gray (#666666)

    Positions 17-20: Pastel1 palette (soft, light colors)
    - 17. Light Red (#FBB4AE)
    - 18. Light Blue (#B3CDE3)
    - 19. Light Green (#CCEBC5)
    - 20. Light Purple (#DECBE4)

    Returns
    -------
    list of str
        List of 20 hex color codes
    """
    # Set1 - strong distinct colors (8 colors)
    set1 = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
            '#FF7F00', '#FFFF33', '#A65628', '#F781BF']

    # Dark2 - muted professional colors (8 colors)
    dark2 = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A',
             '#66A61E', '#E6AB02', '#A6761D', '#666666']

    # Pastel1 - soft light colors (4 colors, subset)
    pastel1 = ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4']

    return set1 + dark2 + pastel1


def validate_and_get_colors(unique_categories, custom_colors=None):
    """
    Validate custom colors or generate default colors for categories.

    Parameters
    ----------
    unique_categories : list
        Sorted list of unique category values
    custom_colors : list of str, optional
        User-provided list of colors (hex codes or named colors)

    Returns
    -------
    dict
        Mapping of categories to colors

    Raises
    ------
    ValueError
        If custom_colors is provided but has wrong length
    """
    n_categories = len(unique_categories)

    if custom_colors is not None:
        if len(custom_colors) != n_categories:
            raise ValueError(
                f"Number of custom colors ({len(custom_colors)}) must match "
                f"number of categories ({n_categories}). "
                f"Categories: {unique_categories}"
            )
        color_palette = custom_colors
    else:
        color_palette = get_default_color_palette()

        # Warn if we have more categories than colors in palette
        if n_categories > len(color_palette):
            warnings.warn(
                f"Variable has {n_categories} categories but default palette only has "
                f"{len(color_palette)} colors. Categories beyond position {len(color_palette)} "
                f"will recycle colors, which may cause visual confusion. "
                f"Consider providing custom colors via the 'category_colors' parameter.",
                UserWarning
            )
            # Extend palette by cycling
            color_palette = color_palette * ((n_categories // len(color_palette)) + 1)

    # Create mapping
    color_mapping = {category: color_palette[i] for i, category in enumerate(unique_categories)}

    return color_mapping


# ============================================================================
# Main Network Graph Function
# ============================================================================

def RDSnetgraph(
        data: pd.DataFrame,
        seed_ids: list,
        waves: list,
        variable: Optional[str] = None,
        category_colors: Optional[List[str]] = None,
        title: Optional[str] = None,
        vertex_size_seed: int = 45,
        vertex_size: int = 30,
        seed_color: str = "#E41A1C",
        nonseed_color: str = "#377EB8",
        edge_width: float = 1.5,
        layout: Literal["Spring", "Circular", "Kamada-Kawai", "Grid", "Star", "Random", "Tree"] = "Spring",
        figsize: tuple = (14, 12),
        show_plot: bool = True,
        save_path: Optional[str] = None
) -> Union[ig.Graph, nx.Graph]:
    """
    Visualization of recruitment chains/networks in respondent driven sampling sample data

    Parameters
    ----------
    data : pd.DataFrame
        The output from RDSdata
    seed_ids : list
        List of seed IDs to include in the network graph
    waves : list
        List of wave numbers to include in the network graph
    variable : str, optional
        A factor or character variable of interest for coloring nodes.
        For space considerations, use short names for categories.
        If not specified, nodes are colored by seed status (seed vs non-seed).
    category_colors : list of str, optional
        Custom colors for each category in the variable. Must be provided in the same
        order as sorted category values (alphabetical/numerical order). Can be hex codes
        (e.g., '#FF0000') or named colors (e.g., 'red'). Length must exactly match the
        number of unique categories in the variable. If not specified, uses the default
        20-color palette (see Notes for color order). Only used when 'variable' is specified.
    title : str, optional
        A user-specified title for the network. By default, provides an empty title.
    vertex_size_seed : int, default 45
        Size of the vertices representing seeds in the plot
    vertex_size : int, default 30
        Size of regular (non-seed) vertices in the plot
    seed_color : str, default "#E41A1C" (red)
        Color of seed vertices when variable is not specified.
        Overridden when variable is provided for grouping.
    nonseed_color : str, default "#377EB8" (blue)
        Color of non-seed vertices when variable is not specified.
        Overridden when variable is provided for grouping.
    edge_width : float, default 1.5
        Thickness of edges (connections) in the plot
    layout : str, default "Spring"
        Graph layout algorithm. Options:
        - "Spring" (default, Fruchterman-Reingold spring/force-directed layout, igraph)
        - "Circular" (circular layout, igraph)
        - "Kamada-Kawai" (force-directed layout, igraph)
        - "Grid" (grid layout, igraph)
        - "Star" (star layout, igraph)
        - "Random" (random layout, igraph)
        - "Tree" (hierarchical tree layout, NetworkX with pygraphviz)
    figsize : tuple, default (14, 12)
        Figure size for matplotlib (width, height)
    show_plot : bool, default True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure. If None, figure is not saved

    Returns
    -------
    Union[ig.Graph, nx.Graph]
        The graph object (igraph for non-Tree layouts, NetworkX for Tree)

    Raises
    ------
    ValueError
        If required columns are missing from data
        If custom_colors length doesn't match number of categories
    UserWarning
        If variable has more than 20 categories and no custom colors provided

    Notes
    -----
    Default Color Palette Order (20 colors):
        When 'variable' is specified and 'category_colors' is not provided, categories
        are colored using a 20-color palette in the following order (categories are
        sorted alphabetically/numerically first):

        Positions 1-8 (Set1 - strong, distinct):
            1. Red, 2. Blue, 3. Green, 4. Purple, 5. Orange, 6. Yellow, 7. Brown, 8. Pink

        Positions 9-16 (Dark2 - muted, professional):
            9. Teal, 10. Dark Orange, 11. Olive, 12. Magenta,
            13. Lime, 14. Gold, 15. Cyan, 16. Gray

        Positions 17-20 (Pastel1 - soft, light):
            17. Light Red, 18. Light Blue, 19. Light Green, 20. Light Purple

        For more than 20 categories, colors will cycle (with a warning). Consider
        providing custom colors via 'category_colors' for better visual distinction.

    Examples
    --------
    >>> import pandas as pd
    >>> from RDSTools import load_toy_data, RDSdata, RDSnetgraph, get_available_seeds, get_available_waves
    >>>
    >>> # Preprocess data with RDSdata function
    >>> data = load_toy_data()
    >>> rds_data = RDSdata(data = data,
    ...                     unique_id = "ID",
    ...                     redeemed_coupon = "CouponR",
    ...                     issued_coupon = ["Coupon1", "Coupon2", "Coupon3"],
    ...                     degree = "Degree")
    >>>
    >>> # Check available seeds and waves
    >>> available_seeds = get_available_seeds(rds_data)
    >>> available_waves = get_available_waves(rds_data)
    >>>
    >>> # Method 1: Simple network without grouping
    >>> out = RDSnetgraph(rds_data,
    ...                   seed_ids=['1', '2'],
    ...                   waves=[0, 1, 2])
    >>>
    >>> # Method 2: Network grouped by variable with default colors
    >>> out = RDSnetgraph(rds_data,
    ...                   seed_ids=['1', '2', '3'],
    ...                   waves=list(range(0, 4)),
    ...                   variable='Sex',
    ...                   title='Recruitment Chain by Sex',
    ...                   vertex_size_seed=8,
    ...                   vertex_size=5,
    ...                   edge_width=2)
    >>>
    >>> # Method 3: Network with custom colors for categories
    >>> # Assuming 'Race' has 3 categories (sorted: 'Asian', 'Black', 'White')
    >>> out = RDSnetgraph(rds_data,
    ...                   seed_ids=['1', '2'],
    ...                   waves=[0, 1, 2],
    ...                   variable='Race',
    ...                   category_colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    >>>
    >>> # Method 4: Tree layout with NetworkX
    >>> out = RDSnetgraph(rds_data,
    ...                   seed_ids=available_seeds[:2],
    ...                   waves=available_waves[:3],
    ...                   variable='Race',
    ...                   layout='Tree',
    ...                   save_path='network_tree.png')
    """
    # Validate required columns
    required_cols = ['R_ID', 'ID', 'SEED', 'S_ID', 'WAVE']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert seed_ids to strings for consistency
    seed_ids = [str(sid) for sid in seed_ids]

    # Filter data by seed_ids and waves
    filtered_data = data[data['S_ID'].isin(seed_ids) & data['WAVE'].isin(waves)].copy()

    if filtered_data.empty:
        raise ValueError(f"No data found for the specified seed_ids and waves")

    # Validate variable and colors if provided
    color_mapping = None
    if variable:
        if variable not in filtered_data.columns:
            raise ValueError(f"Variable '{variable}' not found in data columns")

        # Get unique categories (sorted for consistency)
        unique_categories = sorted([cat for cat in filtered_data[variable].dropna().unique()
                                   if pd.notna(cat)])

        # Warn if many categories
        if len(unique_categories) > 10:
            warnings.warn(
                f"Variable '{variable}' has {len(unique_categories)} categories. "
                f"Visualizations with many categories may be difficult to interpret. "
                f"Consider grouping categories or using a different variable.",
                UserWarning
            )

        # Validate and get color mapping
        color_mapping = validate_and_get_colors(unique_categories, category_colors)

    # Build tree structure
    edges = filtered_data.to_dict('records')
    root_nodes = build_tree(edges)

    # Create appropriate graph based on layout
    if layout == "Tree":
        # Tree layout uses NetworkX with pygraphviz
        return _create_networkx_tree(
            root_nodes, filtered_data, seed_ids, waves, variable, color_mapping,
            vertex_size, vertex_size_seed, seed_color, nonseed_color, edge_width, title,
            figsize, show_plot, save_path
        )
    else:
        # All other layouts use igraph
        return _create_igraph_network(
            root_nodes, filtered_data, seed_ids, waves, layout, variable, color_mapping,
            vertex_size, vertex_size_seed, seed_color, nonseed_color, edge_width, title,
            figsize, show_plot, save_path
        )


def _create_networkx_tree(root_nodes, df, seed_ids, waves, variable, color_mapping,
                          vertex_size, vertex_size_seed, seed_color, nonseed_color, edge_width, title,
                          figsize, show_plot, save_path):
    """Create NetworkX tree layout graph (requires pygraphviz)"""
    G = create_networkx_graph(root_nodes, df)

    if len(G.nodes()) == 0:
        raise ValueError("No nodes found for the selected criteria")

    # Create title if not provided
    if not title:
        seed_str = ", ".join(seed_ids)
        wave_str = ", ".join([str(w) for w in waves])
        title = f"Network Graph for Seeds: {seed_str} and Waves: {wave_str}"
        if variable:
            title += f" (Grouped by: {variable})"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Find root nodes
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        potential_roots = [n for n in G.nodes() if G.out_degree(n) > G.in_degree(n)]
        if potential_roots:
            roots = [max(potential_roots, key=lambda n: G.out_degree(n))]
        else:
            roots = [max(G.nodes(), key=lambda n: G.out_degree(n))]

    # Create hierarchical layout using pygraphviz
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', root=roots[0] if roots else None)
    except:
        # Fallback to spring layout if pygraphviz not available
        print("Warning: pygraphviz not available, using spring layout instead")
        pos = nx.spring_layout(G)

    # Handle coloring
    if variable and color_mapping:
        _apply_networkx_grouping(G, df, variable, color_mapping, pos, ax,
                                vertex_size, vertex_size_seed, edge_width)
    else:
        _draw_networkx_default(G, pos, ax, vertex_size, vertex_size_seed,
                              seed_color, nonseed_color, edge_width)

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network graph saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return G


def _create_igraph_network(root_nodes, df, seed_ids, waves, layout_type, variable, color_mapping,
                           vertex_size, vertex_size_seed, seed_color, nonseed_color, edge_width, title,
                           figsize, show_plot, save_path):
    """Create igraph network with specified layout"""
    G = create_igraph_graph(root_nodes, df)

    if G.vcount() == 0:
        raise ValueError("No nodes found for the selected criteria")

    # Apply layout - using igraph layouts with user-friendly names
    layout_map = {
        "Spring": G.layout_fruchterman_reingold,
        "Circular": G.layout_circle,
        "Kamada-Kawai": G.layout_kamada_kawai,
        "Grid": G.layout_grid,
        "Star": G.layout_star,
        "Random": G.layout_random
    }

    layout_func = layout_map.get(layout_type, G.layout_fruchterman_reingold)
    graph_layout = layout_func()

    # Create title if not provided
    if not title:
        seed_str = ", ".join(seed_ids)
        wave_str = ", ".join([str(w) for w in waves])
        title = f"Network Graph for Seeds: {seed_str} and Waves: {wave_str}"
        if variable:
            title += f" (Grouped by: {variable})"

    # Set up colors
    if variable and color_mapping:
        # Assign colors to vertices based on mapping
        vertex_colors = []
        for v in G.vs:
            node_data = df[df['ID'] == v["name"]]
            if not node_data.empty:
                factor_val = node_data[variable].iloc[0]
                if pd.notna(factor_val) and factor_val in color_mapping:
                    vertex_colors.append(color_mapping[factor_val])
                else:
                    vertex_colors.append('#CCCCCC')  # Gray for missing values
            else:
                vertex_colors.append('#CCCCCC')
    else:
        # Default coloring: use seed_color and nonseed_color parameters
        vertex_colors = [seed_color if v['is_seed'] else nonseed_color for v in G.vs]

    # Set vertex sizes based on seed status
    vertex_sizes = [vertex_size_seed if v['is_seed'] else vertex_size for v in G.vs]

    # Create visual style
    visual_style = {
        "layout": graph_layout,
        "vertex_size": vertex_sizes,
        "vertex_color": vertex_colors,
        "vertex_label": G.vs["name"],
        "vertex_label_size": 12,
        "edge_color": "black",
        "edge_width": edge_width,
        "bbox": (figsize[0] * 80, figsize[1] * 80),
        "margin": 40
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the graph
    ig.plot(G, target=ax, **visual_style)

    # Add title
    if title:
        plt.title(title, fontsize=14)

    plt.axis('off')

    # Add legend if variable is used
    if variable and color_mapping:
        from matplotlib.lines import Line2D
        legend_elements = []

        for category in sorted(color_mapping.keys()):
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_mapping[category],
                       markeredgecolor='black',
                       markersize=10,
                       label=str(category))
            )

        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower left', frameon=False,
                     fontsize=12)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network graph saved to: {save_path}")

    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()

    return G


def _apply_networkx_grouping(G, df, variable, color_mapping, pos, ax,
                             vertex_size, vertex_size_seed, edge_width):
    """Apply color grouping for NetworkX graph"""
    node_to_group = {}

    for node_id in G.nodes():
        node_data = df[df['ID'] == node_id]
        if not node_data.empty and variable in node_data.columns:
            group_value = node_data[variable].iloc[0]
            if not pd.isna(group_value):
                node_to_group[node_id] = group_value

    if not node_to_group:
        _draw_networkx_default(G, pos, ax, vertex_size, vertex_size_seed,
                              '#E41A1C', '#377EB8', edge_width)
        return

    # Set node sizes and colors
    node_sizes = [vertex_size_seed if G.nodes[node].get('is_seed', False) else vertex_size
                  for node in G.nodes()]
    node_colors = [color_mapping.get(node_to_group.get(node), '#CCCCCC')
                   for node in G.nodes()]

    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes,
            with_labels=True, arrows=True, ax=ax, edge_color='black',
            width=edge_width)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_mapping[group], markersize=10,
               markeredgecolor='black',
               label=str(group)) for group in sorted(color_mapping.keys())
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize=12)


def _draw_networkx_default(G, pos, ax, vertex_size, vertex_size_seed, seed_color, nonseed_color, edge_width):
    """Draw NetworkX graph with default seed coloring"""
    node_sizes = [vertex_size_seed if G.nodes[node].get('is_seed', False) else vertex_size
                  for node in G.nodes()]
    node_colors = [seed_color if G.nodes[node].get('is_seed', False)
                   else nonseed_color for node in G.nodes()]

    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes,
            with_labels=True, arrows=True, ax=ax, edge_color='black',
            width=edge_width)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=seed_color,
               markersize=10, markeredgecolor='black', label="Seed"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=nonseed_color,
               markersize=8, markeredgecolor='black', label="Non-seed")
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize=12)
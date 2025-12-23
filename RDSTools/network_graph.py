"""
Network graph visualization for RDS data
"""
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # For saving to files, or use 'TkAgg' if you have Tkinter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import igraph as ig
from typing import List, Optional, Union, Literal, Dict, Any


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
# Main Network Graph Function
# ============================================================================

def RDSnetgraph(
        data: pd.DataFrame,
        seed_ids: List[str],
        waves: List[int],
        layout: Literal["Spring", "Circular", "Kamada-Kawai", "Grid", "Star", "Random", "Tree"] = "Spring",
        group_by: Optional[str] = None,
        node_size: Optional[int] = None,
        figsize: tuple = (14, 12),
        show_plot: bool = True,
        save_path: Optional[str] = None
) -> Union[ig.Graph, nx.Graph]:
    """
    Create a network graph visualization for RDS data.

    Parameters
    ----------
    data : pd.DataFrame
        RDS data with required columns: S_ID, WAVE, ID, R_ID
    seed_ids : List[str]
        List of seed IDs to include in the graph
    waves : List[int]
        List of wave numbers to include
    layout : str, default="Spring"
        Graph layout algorithm. Options: "Spring", "Circular", "Kamada-Kawai",
        "Grid", "Star", "Random", "Tree"
        Note: "Tree" requires pygraphviz and uses NetworkX.
        All other layouts use igraph.
    group_by : str, optional
        Column name to use for node coloring/grouping
    node_size : int, optional
        Size of nodes. If None, uses default based on layout
        (700 for Tree with NetworkX, 30 for others with igraph)
    figsize : tuple, default=(14, 12)
        Figure size for matplotlib
    show_plot : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure. If None, figure is not saved

    Returns
    -------
    Union[ig.Graph, nx.Graph]
        The created graph object (igraph for non-Tree layouts, networkx for Tree)

    Examples
    --------
    # Create a simple spring layout (default - uses igraph)
    >>> G = RDSnetgraph(data, seed_ids=['1'], waves=[1, 2])

    >>> # Create tree layout (uses NetworkX, requires pygraphviz)
    >>> G = RDSnetgraph(data, seed_ids=['1'], waves=[1, 2], layout='Tree')

    >>> # Create colored by variable with custom node size
    >>> G = RDSnetgraph(
    ...     data,
    ...     seed_ids=['1', '2'],
    ...     waves=[1, 2, 3],
    ...     layout='Spring',
    ...     group_by='Gender',
    ...     node_size=15
    ... )
    """

    # Set default node size based on layout
    if node_size is None:
        node_size = 700 if layout == "Tree" else 30

    # Filter data
    df = data[data['S_ID'].isin(seed_ids) & data['WAVE'].isin(waves)]

    if df.empty:
        raise ValueError("No data found for the selected seed IDs and waves")

    # Build tree structure
    edges = df.to_dict('records')
    root_nodes = build_tree(edges)

    # Create appropriate graph based on layout
    if layout == "Tree":
        # Tree layout uses NetworkX with pygraphviz
        G = _create_networkx_tree(
            root_nodes, df, seed_ids, waves, group_by,
            node_size, figsize, show_plot, save_path
        )
    else:
        # All other layouts use igraph
        G = _create_igraph_network(
            root_nodes, df, seed_ids, waves, layout,
            group_by, node_size, figsize, show_plot, save_path
        )

    return G


# ============================================================================
# Internal Helper Functions for Graph Creation and Visualization
# ============================================================================

def _create_networkx_tree(root_nodes, df, seed_ids, waves, group_by,
                          node_size, figsize, show_plot, save_path):
    """Create NetworkX tree layout graph (requires pygraphviz)"""
    G = create_networkx_graph(root_nodes, df)

    if len(G.nodes()) == 0:
        raise ValueError("No nodes found for the selected criteria")

    # Create title
    seed_str = ", ".join(seed_ids)
    wave_str = ", ".join([str(w) for w in waves])
    title = f"Network Graph for Seeds: {seed_str} and Waves: {wave_str}"

    if group_by:
        title += f" (Grouped by: {group_by})"

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
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', root=roots[0] if roots else None)

    # Handle coloring
    if group_by and group_by in df.columns:
        _apply_networkx_grouping(G, df, group_by, pos, ax, node_size)
    else:
        _draw_networkx_default(G, pos, ax, node_size)

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return G


def _create_igraph_network(root_nodes, df, seed_ids, waves, layout_type,
                           group_by, node_size, figsize, show_plot, save_path):
    """Create igraph network with specified layout"""
    G = create_igraph_graph(root_nodes, df)

    if G.vcount() == 0:
        raise ValueError("No nodes found for the selected criteria")

    # Apply layout - using only igraph-supported layouts
    layout_map = {
        "Spring": G.layout_fruchterman_reingold,
        "Circular": G.layout_circle,
        "Kamada-Kawai": G.layout_kamada_kawai,
        "Grid": G.layout_grid,
        "Star": G.layout_star,
        "Random": G.layout_random
    }

    layout = layout_map.get(layout_type, G.layout_fruchterman_reingold)()

    # Create title
    seed_str = ", ".join(seed_ids)
    wave_str = ", ".join([str(w) for w in waves])
    title = f"Network Graph for Seeds: {seed_str} and Waves: {wave_str}"

    if group_by:
        title += f" (Grouped by: {group_by})"

    # Configure visual style
    visual_style = {
        "layout": layout,
        "bbox": (800, 800),
        "margin": 40,
        "vertex_label": G.vs["name"],
        "vertex_label_size": 12,
        "edge_arrow_size": 1.5,
        "edge_arrow_width": 1.5,
        "edge_width": 1.5,
        "edge_curved": 0.2,
    }

    # Set node sizes
    vertex_sizes = [node_size * 1.5 if v['is_seed'] else node_size for v in G.vs]
    visual_style["vertex_size"] = vertex_sizes

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Handle coloring
    if group_by and group_by in df.columns:
        _apply_igraph_grouping(G, df, group_by, visual_style, ax)
    else:
        _draw_igraph_default(G, visual_style, ax)

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return G


def _apply_networkx_grouping(G, df, group_var, pos, ax, node_size):
    """Apply color grouping for NetworkX graph"""
    node_to_group = {}

    for node_id in G.nodes():
        node_data = df[df['ID'] == node_id]
        if not node_data.empty and group_var in node_data.columns:
            group_value = node_data[group_var].iloc[0]
            if not pd.isna(group_value):
                node_to_group[node_id] = group_value

    if not node_to_group:
        _draw_networkx_default(G, pos, ax, node_size)
        return

    # Check if continuous variable
    is_continuous = (pd.api.types.is_numeric_dtype(df[group_var]) and
                     len(df[group_var].unique()) >= 10)

    if is_continuous:
        values = list(node_to_group.values())
        vmin, vmax = min(values), max(values)
        cmap = cm.get_cmap('viridis')

        node_colors = []
        for node in G.nodes():
            if node in node_to_group:
                norm_value = (node_to_group[node] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                node_colors.append(cmap(norm_value))
            else:
                node_colors.append((0.8, 0.8, 0.8, 1.0))

        nx.draw(G, pos, node_color=node_colors, node_size=node_size,
                with_labels=True, arrows=True, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=group_var)
    else:
        unique_groups = list(set(node_to_group.values()))
        cmap = cm.get_cmap('tab10', max(8, len(unique_groups)))
        color_map = {group: cmap(i % 10) for i, group in enumerate(unique_groups)}

        node_colors = [color_map.get(node_to_group.get(node), (0.8, 0.8, 0.8, 1.0))
                       for node in G.nodes()]

        nx.draw(G, pos, node_color=node_colors, node_size=node_size,
                with_labels=True, arrows=True, ax=ax)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_map[group], markersize=10,
                   label=str(group)) for group in unique_groups
        ]
        plt.legend(handles=legend_elements, title=group_var, loc='best')


def _draw_networkx_default(G, pos, ax, node_size):
    """Draw NetworkX graph with default seed coloring"""
    node_colors = ['#ff6347' if G.nodes[node].get('is_seed', False)
                   else '#4682b4' for node in G.nodes()]

    nx.draw(G, pos, node_color=node_colors, node_size=node_size,
            with_labels=True, arrows=True, ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#ff6347",
               markersize=10, label="Seed"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#4682b4",
               markersize=8, label="Non-seed")
    ]
    ax.legend(handles=legend_elements, loc='best')


def _apply_igraph_grouping(G, df, group_var, visual_style, ax):
    """Apply color grouping for igraph graph"""
    node_to_group = {}

    for v in G.vs:
        node_data = df[df['ID'] == v["name"]]
        if not node_data.empty and group_var in node_data.columns:
            group_value = node_data[group_var].iloc[0]
            if not pd.isna(group_value):
                node_to_group[v["name"]] = group_value

    if not node_to_group:
        _draw_igraph_default(G, visual_style, ax)
        return

    # Check if continuous
    is_continuous = (pd.api.types.is_numeric_dtype(df[group_var]) and
                     len(df[group_var].unique()) >= 10)

    if is_continuous:
        values = list(node_to_group.values())
        vmin, vmax = min(values), max(values)
        cmap = cm.get_cmap('viridis')

        vertex_colors = []
        for v in G.vs:
            if v["name"] in node_to_group:
                norm_value = (node_to_group[v["name"]] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                rgb_color = cmap(norm_value)
                hex_color = f"#{int(rgb_color[0] * 255):02x}{int(rgb_color[1] * 255):02x}{int(rgb_color[2] * 255):02x}"
                vertex_colors.append(hex_color)
            else:
                vertex_colors.append("#d3d3d3")

        visual_style["vertex_color"] = vertex_colors
        ig.plot(G, **visual_style, target=ax)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=group_var)
    else:
        unique_groups = list(set(node_to_group.values()))
        cmap = cm.get_cmap('tab10', max(8, len(unique_groups)))

        color_map = {}
        for i, group in enumerate(unique_groups):
            rgb = cmap(i % 10)
            color_map[group] = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

        vertex_colors = [color_map.get(node_to_group.get(v["name"]), "#d3d3d3")
                         for v in G.vs]

        visual_style["vertex_color"] = vertex_colors
        ig.plot(G, **visual_style, target=ax)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_map[group], markersize=10,
                   label=str(group)) for group in unique_groups
        ]
        plt.legend(handles=legend_elements, title=group_var, loc='best')


def _draw_igraph_default(G, visual_style, ax):
    """Draw igraph with default seed coloring"""
    vertex_colors = ['#ff6347' if v['is_seed'] else '#4682b4' for v in G.vs]
    visual_style["vertex_color"] = vertex_colors

    ig.plot(G, **visual_style, target=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#ff6347",
               markersize=10, label="Seed"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#4682b4",
               markersize=8, label="Non-seed")
    ]
    plt.legend(handles=legend_elements, loc='best')
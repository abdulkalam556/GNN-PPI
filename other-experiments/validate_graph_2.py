import os
import torch
from torch_geometric.data import Data
import argparse


def validate_graph(graph_path):
    """
    Validates a single graph file.
    
    Args:
        graph_path (str): Path to the graph file.
        
    Returns:
        bool: True if the graph is valid, False otherwise.
        str: Message explaining the validation status.
    """
    try:
        # Load the graph
        graph = torch.load(graph_path)

        # Check if it's a PyTorch Geometric Data object
        if not isinstance(graph, Data):
            return False, f"File at {graph_path} is not a PyTorch Geometric Data object."

        # Check if `x` (node features) exists and is non-empty
        if not hasattr(graph, "x") or graph.x is None or graph.x.size(0) == 0:
            return False, f"Graph at {graph_path} has no valid node features."

        # Check if `edge_index` exists and is non-empty
        if not hasattr(graph, "edge_index") or graph.edge_index is None or graph.edge_index.size(1) == 0:
            return False, f"Graph at {graph_path} has no valid edges."

        # Check if `edge_index` indices are within the valid range
        if graph.edge_index.max() >= graph.x.size(0):
            return False, (
                f"Graph at {graph_path} has invalid edges. "
                f"Max index in edge_index: {graph.edge_index.max()}, Num nodes: {graph.x.size(0)}."
            )

        # Optional: Check if edge_index is symmetric for undirected graphs
        if not is_edge_index_symmetric(graph.edge_index):
            return False, f"Graph at {graph_path} has a non-symmetric edge_index for an undirected graph."

        # Optional: Check if `edge_index` contains duplicate edges
        if has_duplicate_edges(graph.edge_index):
            return False, f"Graph at {graph_path} contains duplicate edges."

        # If all checks pass
        return True, f"Graph at {graph_path} is valid."

    except Exception as e:
        return False, f"Error loading or validating graph at {graph_path}: {str(e)}"


def is_edge_index_symmetric(edge_index):
    """
    Checks if the edge_index is symmetric (for undirected graphs).
    
    Args:
        edge_index (Tensor): The edge_index tensor.
        
    Returns:
        bool: True if symmetric, False otherwise.
    """
    edge_index_flipped = edge_index.flip(0)
    return torch.all(torch.eq(edge_index, edge_index_flipped, out=None))


def has_duplicate_edges(edge_index):
    """
    Checks if `edge_index` contains duplicate edges.
    
    Args:
        edge_index (Tensor): The edge_index tensor.
        
    Returns:
        bool: True if duplicates exist, False otherwise.
    """
    edges = edge_index.t().tolist()
    unique_edges = set(map(tuple, edges))
    return len(edges) != len(unique_edges)


def validate_graphs_in_directory(directory):
    """
    Validates all graph files in a given directory.
    
    Args:
        directory (str): Path to the directory containing graph files.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    graph_files = [f for f in os.listdir(directory) if f.endswith(".pt")]

    if not graph_files:
        print(f"No graph files found in directory: {directory}")
        return

    print(f"Validating graphs in directory: {directory}")
    for graph_file in graph_files:
        graph_path = os.path.join(directory, graph_file)
        is_valid, message = validate_graph(graph_path)
        print(f"Graph: {graph_file} - {message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate graph files in a directory.")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to the directory containing graph files.",
    )
    args = parser.parse_args()
    validate_graphs_in_directory(args.directory)

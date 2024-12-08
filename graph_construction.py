import os
import re
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding, expasy_protein_scale
from graphein.protein.edges.distance import add_distance_threshold

import torch
from torch_geometric.data import Data
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5EncoderModel

# Helper Functions
def validate_graph(graph, sequence_length):
    """
    Validates if a PyTorch Geometric graph is valid.
    """
    
    def is_edge_index_symmetric(edge_index):
        """
        Checks if `edge_index` contains non symmetric edges.
        """
        # Flip the edge index to get reversed edges
        edge_index_flipped = edge_index.flip(0)

        # Combine original and flipped edges
        combined_edges = torch.cat([edge_index, edge_index_flipped], dim=1)

        # Remove duplicate edges
        unique_edges = torch.unique(combined_edges, dim=1)

        # Check if all edges have their reverses
        return unique_edges.size(1) == edge_index.size(1)


    def has_duplicate_edges(edge_index):
        """
        Checks if `edge_index` contains duplicate edges.
        """
        edges = edge_index.t().tolist()
        unique_edges = set(map(tuple, edges))
        return len(edges) != len(unique_edges)

    try:
        if not isinstance(graph, Data):
            return False, "Graph is not a torch_geometric.data.data.Data object."

        if not hasattr(graph, "x") or graph.x is None or graph.x.size(0) == 0:
            return False, "Graph has no valid node features."

        if not hasattr(graph, "edge_index") or graph.edge_index is None or graph.edge_index.size(1) == 0:
            return False, "Graph has no valid edges."
        
        # Validate node feature alignment with sequence length
        if graph.x.size(0) != sequence_length:
            return False, (
                f"Graph node features do not match sequence length. "
                f"Node features: {graph.x.size(0)}, Sequence length: {sequence_length}."
            )

        if graph.edge_index.max() >= graph.x.size(0):
            return False, (
                f"Graph has invalid edges. "
                f"Max index in edge_index: {graph.edge_index.max()}, Num nodes: {graph.x.size(0)}."
            )

        if not is_edge_index_symmetric(graph.edge_index):
            return False, "Graph has a non-symmetric edge_index for an undirected graph."

        if has_duplicate_edges(graph.edge_index):
            return False, "Graph contains duplicate edges."

        return True, "Graph is valid."

    except Exception as e:
        return False, f"Error validating graph: {str(e)}"

# Main Processing Function
def process_pdb_file(pdb_path, graph_type, data_dir):
    """
    Process a single PDB file and return the graph.
    """
    config = ProteinGraphConfig(
        node_metadata_functions=[
            amino_acid_one_hot, meiler_embedding, expasy_protein_scale
        ],
        edge_construction_functions=[
            partial(add_distance_threshold, long_interaction_threshold=0)
        ]
    )

    nx_graph = construct_graph(config=config, path=pdb_path)

    # Map nodes to numeric indices
    node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    #rmove self loops if any
    edge_list = [(node_mapping[u], node_mapping[v]) for u, v in nx_graph.edges() if u != v]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    #ensure symmetry of teh graph
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    # Generate node features based on graph type
    node_features = []
    if graph_type in ["protbert", "prostt5"]:
        # Load the precomputed embedding
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        embedding_dir = os.path.join(data_dir, "embeddings", graph_type)
        embedding_path = os.path.join(embedding_dir, f"{pdb_id}.npy")
        try:
            # Load the saved .npy embedding and convert it to a PyTorch tensor
            embedding = np.load(embedding_path)
            node_features = torch.tensor(embedding[1:-1], dtype=torch.float)
            # Ensure the number of rows matches the number of nodes
            if node_features.size(0) != nx_graph.number_of_nodes():
                raise ValueError(
                    f"Mismatch: Embedding length ({node_features.size(0)}) "
                    f"does not match number of nodes ({nx_graph.number_of_nodes()}) in graph."
                )
        except Exception as e:
            print(f"Error loading embedding for PDB ID {pdb_id} in {graph_type}: {e}")
            raise
    else:
        for _, data in nx_graph.nodes(data=True):
            if graph_type == "onehot":
                node_features.append(torch.tensor(data["amino_acid_one_hot"], dtype=torch.float))
            elif graph_type == "physchem":
                node_features.append(torch.tensor(data["meiler"].values, dtype=torch.float))
            elif graph_type == "expasy":
                node_features.append(torch.tensor(data["expasy"].values, dtype=torch.float))
            elif graph_type == "protbert":
                embedding = protbert_embedding(sequence)
                node_features.append(embedding)
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")

        # Convert node features to a PyTorch tensor
        node_features = torch.stack(node_features)
    
    # Create PyTorch Geometric graph
    graph = Data(x=node_features, edge_index=edge_index)

    # Validate the graph
    is_valid, message = validate_graph(graph, len(node_mapping))
    if not is_valid:
        raise ValueError(f"Graph validation failed: {message}")

    return graph


# Main Function
def main(data_dir, graph_type=None):
    raw_dir = os.path.join(data_dir, "raw")
    graphs_dir = os.path.join(data_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # If a specific graph type is provided, only process that type
    graph_types = [graph_type] if graph_type else ["onehot", "physchem", "expasy", "protbert", "prostt5"]

    for graph_type in graph_types:
        output_dir = os.path.join(graphs_dir, graph_type)
        os.makedirs(output_dir, exist_ok=True)
        
        pdb_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdb")]
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            if pdb_file.endswith(".pdb"):
                pdb_path = os.path.join(raw_dir, pdb_file)
                try:
                    graph = process_pdb_file(pdb_path, graph_type, data_dir)

                    # Save graph
                    output_path = os.path.join(output_dir, f"{os.path.splitext(pdb_file)[0]}.pt")
                    torch.save(graph, output_path)
                    #print(f"Saved {graph_type} graph for {pdb_file} at {output_path}")
                except Exception as e:
                    print(f"Failed to process {pdb_file} for {graph_type}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate residue graphs from PDB files.")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory containing raw PDB files.")
    parser.add_argument("--graph_type", type=str, choices=["onehot", "physchem", "expasy", "protbert", "prostt5"],
                        help="Specify a graph type to process (default: process all graph types).")
    args = parser.parse_args()

    main(args.data_dir, args.graph_type)

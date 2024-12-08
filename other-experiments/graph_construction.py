import os
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import biographs as bg
import networkx as nx
import argparse
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

# Define three-letter to one-letter residue mapping
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def load_embeddings(embedding_file):
    """
    Load and process ProtBERT embeddings.
    Args:
        embedding_file (str): Path to the .npy embedding file.
    Returns:
        np.ndarray: Processed embeddings with shape [num_residues, 1024].
    """
    try:
        embeddings = np.load(embedding_file)
        return embeddings[1:-1]  # Remove [CLS] and [SEP] tokens
    except Exception as e:
        print(f"Error loading embeddings from {embedding_file}: {e}")
        return None


def get_edge_index_from_biographs(pdb_file, cutoff=6.0):
    """
    Create a residue-level graph using biographs and extract edge indices.
    Args:
        pdb_file (str): Path to the PDB file.
        cutoff (float): Distance threshold for edge creation.
    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges].
    """
    try:
        molecule = bg.Pmolecule(pdb_file)
        network = molecule.network(cutoff=cutoff)  # Create graph with the specified cutoff
        adjacency_matrix = nx.adjacency_matrix(network)
        adjacency_matrix = adjacency_matrix.todense()  # Convert to dense matrix

        # Get non-zero indices in the adjacency matrix (COO format)
        rows, cols = np.nonzero(adjacency_matrix)
        edge_index = np.array([rows, cols])  # Convert to a single NumPy array
        edge_index = torch.tensor(edge_index, dtype=torch.long)  # Shape: [2, num_edges]
        return edge_index
    except Exception as e:
        print(f"Error constructing graph for {pdb_file}: {e}")
        return None

def compute_onehot_features(pdb_file):
    """
    Compute one-hot encoded features for residues in a PDB file.
    Args:
        pdb_file (str): Path to the PDB file.
    Returns:
        np.ndarray: One-hot encoded features with shape [num_residues, 20].
    """
    onehot_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    structure = parser.get_structure("protein", pdb_file)
    sequence = []
    for pp in ppb.build_peptides(structure):
        for residue in pp:
            resname = residue.get_resname()
            if resname in three_to_one:  # Map three-letter to one-letter code
                sequence.append(three_to_one[resname])
            else:
                sequence.append("X")  # Default to "X" for unknown residues

    onehot_matrix = np.zeros((len(sequence), len(onehot_table)), dtype=np.float32)
    for i, residue in enumerate(sequence):
        if residue in onehot_table:
            onehot_matrix[i][onehot_table.index(residue)] = 1.0
    return onehot_matrix


def compute_physicochemical_features(pdb_file):
    """
    Compute physicochemical properties for residues in a PDB file.
    Args:
        pdb_file (str): Path to the PDB file.
    Returns:
        np.ndarray: Features with shape [num_residues, num_features].
    """
    physchem_dict = {
        'A':[ 0.62014, -0.18875, -1.2387, -0.083627, -1.3296, -1.3817, -0.44118],
        'C':[0.29007, -0.44041,-0.76847, -1.05, -0.4893, -0.77494, -1.1148],
        'D':[-0.9002, 1.5729, -0.89497, 1.7376, -0.72498, -0.50189, -0.91814],
        'E':[-0.74017, 1.5729, -0.28998, 1.4774, -0.25361, 0.094051, -0.4471],
        'F':[1.1903, -1.1954, 1.1812, -1.1615, 1.1707, 0.8872, 0.02584],
        'G':[ 0.48011, 0.062916, -1.9949, 0.25088, -1.8009, -2.0318, 2.2022],
        'H':[-0.40009, -0.18875, 0.17751, 0.77123, 0.5559, 0.44728, -0.71617],
        'I':[1.3803, -0.84308, 0.57625, -1.1615, 0.10503, -0.018637, -0.21903],
        'K':[-1.5003, 1.5729, 0.75499, 1.1057, 0.44318, 0.95221, -0.27937],
        'L':[1.0602, -0.84308, 0.57625, -1.273, 0.10503, 0.24358, 0.24301],
        'M':[0.64014, -0.59141, 0.59275, -0.97565, 0.46368, 0.46679, -0.51046],
        'N':[-0.78018, 1.0696, -0.38073, 1.2172, -0.42781, -0.35453, -0.46879],
        'P':[0.12003, 0.062916, -0.84272, -0.1208, -0.45855, -0.75977, 3.1323],
        'Q':[-0.85019, 0.16358, 0.22426, 0.8084, 0.04355, 0.24575, 0.20516],
        'R':[-2.5306, 1.5729, 0.89249, 0.8084, 1.181, 1.6067, 0.11866],
        'S':[-0.18004, 0.21392, -1.1892, 0.32522, -1.1656, -1.1282, -0.48056],
        'T':[-0.050011, -0.13842, -0.58422, 0.10221, -0.69424, -0.63625, -0.50017],
        'V':[1.0802, -0.69208, -0.028737, -0.90132, -0.36633, -0.3762, 0.32502],
        'W':[0.81018, -1.6484, 2.0062, -1.0872, 2.3901, 1.8299, 0.032377],
        'Y':[0.26006, -1.0947, 1.2307, -0.78981, 1.2527, 1.1906, -0.18876]
    }
    
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    structure = parser.get_structure("protein", pdb_file)
    sequence = []
    for pp in ppb.build_peptides(structure):
        for residue in pp:
            resname = residue.get_resname()
            if resname in three_to_one:  # Map three-letter to one-letter code
                sequence.append(three_to_one[resname])
            else:
                sequence.append("X")  # Default to "X" for unknown residues

    physchem_matrix = np.zeros((len(sequence), len(next(iter(physchem_dict.values())))), dtype=np.float32)
    for i, residue in enumerate(sequence):
        if residue in physchem_dict:
            physchem_matrix[i] = physchem_dict[residue]
    return physchem_matrix


def construct_graphs(raw_dir, embedding_dir, output_dir, cutoff=6.0, node_feature_type="embedding"):
    """
    Construct residue-level protein graphs with specified node features and save them as PyTorch Geometric Data objects.
    Args:
        raw_dir (str): Directory containing raw PDB files.
        embedding_dir (str): Directory containing embeddings (if using embedding features).
        output_dir (str): Directory to save constructed graphs.
        cutoff (float): Distance threshold for edge creation.
        node_feature_type (str): Type of node features ('onehot', 'physchem', or 'embedding').
    """
    os.makedirs(output_dir, exist_ok=True)

    for pdb_file in tqdm(os.listdir(raw_dir), desc="Constructing graphs"):
        if pdb_file.endswith(".pdb"):
            pdb_id = os.path.splitext(pdb_file)[0]
            pdb_path = os.path.join(raw_dir, pdb_file)
            embedding_path = os.path.join(embedding_dir, f"{pdb_id}.npy") if embedding_dir else None

            # Step 1: Load Node Features
            if node_feature_type == "embedding":
                node_features = load_embeddings(embedding_path)
                if node_features is None:
                    print(f"Skipping {pdb_id} due to missing or corrupted embeddings.")
                    continue
            elif node_feature_type == "onehot":
                node_features = compute_onehot_features(pdb_path)
            elif node_feature_type == "physchem":
                node_features = compute_physicochemical_features(pdb_path)

            # Step 2: Construct graph using biographs
            edge_index = get_edge_index_from_biographs(pdb_path, cutoff=cutoff)
            if edge_index is None:
                print(f"Skipping {pdb_id} due to graph construction error.")
                continue

            # Step 3: Create PyTorch Geometric graph object
            graph = Data(x=torch.tensor(node_features, dtype=torch.float),
                         edge_index=edge_index)

            # Step 4: Save the graph
            output_path = os.path.join(output_dir, f"{pdb_id}.pt")
            torch.save(graph, output_path)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Construct residue-level protein graphs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory for data (e.g., 'data').")
    parser.add_argument("--embedding_type", type=str, required=False, choices=["protbert", "prostt5"],
                        help="Embedding type to compute ('protbert' or 'prostt5').")
    parser.add_argument("--node_feature_type", type=str, required=True, choices=["onehot", "physchem", "embedding"],
                        help="Type of node features to use: 'onehot', 'physchem', or 'embedding'.")
    parser.add_argument("--distance_threshold", type=float, default=6.0,
                        help="Distance threshold (in Å) for edge creation.")
    args = parser.parse_args()

    # Set directories based on arguments
    raw_dir = os.path.join(args.data_dir, "raw/")
    embedding_dir = os.path.join(args.data_dir, "embeddings", args.embedding_type) if args.node_feature_type == "embedding" else None
    
    if args.node_feature_type == "embedding":
        # Include embedding type in the output directory
        output_dir = os.path.join(args.data_dir, "graphs", args.embedding_type)
    else:
        # Use only the node feature type for non-embedding features
        output_dir = os.path.join(args.data_dir, "graphs", args.node_feature_type)


    # Construct graphs
    construct_graphs(raw_dir, embedding_dir, output_dir, args.distance_threshold, args.node_feature_type)



if __name__ == "__main__":
    main()

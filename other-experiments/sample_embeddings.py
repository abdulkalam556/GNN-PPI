import os
import random
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import argparse

def get_sequence_from_pdb(pdb_file):
    """
    Extract the sequence length from a PDB file.
    Args:
        pdb_file (str): Path to the PDB file.
    Returns:
        str: Protein sequence extracted from the PDB file.
    """
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    try:
        structure = parser.get_structure("pdb", pdb_file)
        sequence = ""
        for pp in ppb.build_peptides(structure):
            sequence += str(pp.get_sequence())
        return sequence
    except Exception as e:
        print(f"Error reading PDB file {pdb_file}: {e}")
        return None

def process_samples(raw_dir, embeddings_dir):
    # Get all PDB files
    pdb_files = [file for file in os.listdir(raw_dir) if file.endswith(".pdb")]

    # Randomly select 20 files
    random_files = random.sample(pdb_files, min(20, len(pdb_files)))  # Limit to available files if less than 20

    # Compare sequence lengths and embeddings
    for file_name in random_files:
        pdb_id = os.path.splitext(file_name)[0]
        pdb_file_path = os.path.join(raw_dir, file_name)
        embedding_file_path = os.path.join(embeddings_dir, f"{pdb_id}.npy")

        # Extract sequence from PDB
        sequence = get_sequence_from_pdb(pdb_file_path)
        if sequence is None:
            continue
        sequence_length = len(sequence)

        # Load embeddings
        if os.path.exists(embedding_file_path):
            embeddings = np.load(embedding_file_path)
            embedding_shape = embeddings.shape
        else:
            print(f"Embedding file not found for {pdb_id} in path {embedding_file_path}")
            continue

        # Print comparison
        print(f"PDB ID: {pdb_id}")
        print(f"  Sequence Length (from PDB): {sequence_length}")
        print(f"  Embedding Shape: {embedding_shape} (L={embedding_shape[0]}, d={embedding_shape[1]})")
        if sequence_length != embedding_shape[0]:
            print(f"  WARNING: Sequence length and embedding length mismatch for {pdb_id}")
        print("-" * 30)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Compute protein embeddings using SeqVec or ProtBERT.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory for data (e.g., 'data').")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["protbert", "prostt5"],
                        help="Embedding type to compute ('protbert' or 'prostt5').")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw/")  # Directory with raw PDB files
    embeddings_dir = os.path.join(args.data_dir, "embeddings", args.embedding_type)  # Output directory for embeddings
    
    process_samples(raw_dir, embeddings_dir)


if __name__ == "__main__":
    main()
import os
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5EncoderModel
import argparse
from functools import partial

from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding, expasy_protein_scale
from graphein.protein.edges.distance import add_distance_threshold


# Residue mapping dictionary for 3-letter to single-letter code
ressymbl = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'
}

def load_protein_sequences(raw_dir):
    """
    Load protein sequences from raw PDB files.
    Args:
        raw_dir (str): Directory containing raw PDB files.
    Returns:
        dict: Mapping of PDB ID to protein sequence.
    """
    pdb_sequences = {}
    
    for file in tqdm(os.listdir(raw_dir), desc="Extracting sequences"):
        if file.endswith(".pdb"):
            file_path = os.path.join(raw_dir, file)
            pdb_id = os.path.splitext(file)[0]
            try:
                config = ProteinGraphConfig(
                    node_metadata_functions=[
                        amino_acid_one_hot, meiler_embedding, expasy_protein_scale
                    ],
                    edge_construction_functions=[
                        partial(add_distance_threshold, long_interaction_threshold=0)
                    ]
                )

                nx_graph = construct_graph(config=config, path=file_path)
                # Generate the sequence from residues
                sequence = "".join([ressymbl.get(data["residue_name"], 'X') for _, data in nx_graph.nodes(data=True)])
                pdb_sequences[pdb_id] = sequence
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    return pdb_sequences


def compute_protbert_embeddings(sequences, output_dir, model_name="Rostlab/prot_bert"):
    """
    Compute ProtBERT embeddings for a set of protein sequences.
    Args:
        sequences (dict): Dictionary mapping PDB IDs to protein sequences.
        output_dir (str): Directory to save computed embeddings.
        model_name (str): Hugging Face model name (default: "Rostlab/prot_bert").
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    if torch.cuda.is_available():
        model.to("cuda")

    for pdb_id, sequence in tqdm(sequences.items(), desc="Computing ProtBERT embeddings"):
        try:
            # Replace rare amino acids with X
            sequence = re.sub(r"[UZOB]", "X", sequence)
            
            # Format the sequence
            formatted_sequence = " ".join(list(sequence))  # Add spaces between residues
            
            # Tokenize the sequence
            inputs = tokenizer(formatted_sequence, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {key: val.to("cuda") for key, val in inputs.items()}
            
            # Compute embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [L x 1024]
            
            # Save to file
            output_path = os.path.join(output_dir, f"{pdb_id}.npy")
            np.save(output_path, embeddings.cpu().numpy())
        
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")



def compute_prostt5_embeddings(sequences, output_dir, model_name="Rostlab/ProstT5"):
    """
    Compute ProstT5 embeddings for a set of protein sequences.
    Args:
        sequences (dict): Dictionary mapping PDB IDs to protein sequences.
        output_dir (str): Directory to save computed embeddings.
        model_name (str): Hugging Face model name (default: "Rostlab/ProstT5").
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    
    if torch.cuda.is_available():
        model.to("cuda")

    for pdb_id, sequence in tqdm(sequences.items(), desc="Computing ProstT5 embeddings"):
        try:
            # Preprocess sequence
            formatted_sequence = "<AA2fold> " + " ".join(list(re.sub(r"[UZOB]", "X", sequence)))  # Add prefix
            
            # Tokenize the sequence
            inputs = tokenizer(formatted_sequence, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {key: val.to("cuda") for key, val in inputs.items()}
            
            # Compute embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [L x 1024]
            
            # Save to file
            output_path = os.path.join(output_dir, f"{pdb_id}.npy")
            np.save(output_path, embeddings.cpu().numpy())
        
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Compute protein embeddings using ProtBERT or ProstT5.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory for data (e.g., 'data').")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["protbert", "prostt5"],
                        help="Embedding type to compute ('protbert' or 'prostt5').")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw/")  # Directory with raw PDB files
    output_dir = os.path.join(args.data_dir, "embeddings", args.embedding_type)  # Output directory for embeddings
    
    # Step 1: Extract sequences from raw PDB files
    sequences = load_protein_sequences(raw_dir)
    
    # Step 2: Compute embeddings
    if args.embedding_type == "protbert":
        compute_protbert_embeddings(sequences, output_dir, model_name="Rostlab/prot_bert")
    elif args.embedding_type == "prostt5":
        compute_prostt5_embeddings(sequences, output_dir, model_name="Rostlab/ProstT5")
    else:
        print(f"Unsupported embedding type: {args.embedding_type}")


if __name__ == "__main__":
    main()

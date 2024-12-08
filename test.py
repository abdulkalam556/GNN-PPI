import os
import torch
import yaml
import argparse
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import load_from_disk
from model import ProteinInteractionGNN

# Mapping of graph types to feature dimensions
graph_type_to_features = {
    "onehot": 20,      # One-hot encoding
    "physchem": 7,     # Physicochemical (Meiler 7D features)
    "expasy": 61,      # ExPASy properties
    "protbert": 1024,  # ProtBERT embeddings
    "prostt5": 1024    # ProST5 embeddings
}

class ProteinGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, graph_type):
        """
        Initialize the dataset for dynamic graph loading.

        Args:
            dataset (Dataset): Hugging Face dataset containing graph paths.
            graph_type (str): Type of graph to load (e.g., 'onehot', 'physchem').
        """
        self.dataset = dataset
        self.graph_type = graph_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]

        protein_1_graph = torch.load(row["protein_1_graph"][self.graph_type], weights_only=False)
        protein_2_graph = torch.load(row["protein_2_graph"][self.graph_type], weights_only=False)

        # Check and log invalid edges in protein 1 graph
        if protein_1_graph.edge_index.max() >= protein_1_graph.x.size(0):
            print(f"Invalid edge_index in protein_1_graph for Protein ID {row['protein_1_id']}. "
                  f"Max index: {protein_1_graph.edge_index.max()}, Num nodes: {protein_1_graph.x.size(0)}")
            return None  # Skip this sample

        # Check and log invalid edges in protein 2 graph
        if protein_2_graph.edge_index.max() >= protein_2_graph.x.size(0):
            print(f"Invalid edge_index in protein_2_graph for Protein ID {row['protein_2_id']}. "
                  f"Max index: {protein_2_graph.edge_index.max()}, Num nodes: {protein_2_graph.x.size(0)}")
            return None  # Skip this sample

        # Create Data objects
        protein_1_graph_data = Data(x=protein_1_graph.x, edge_index=protein_1_graph.edge_index)
        protein_2_graph_data = Data(x=protein_2_graph.x, edge_index=protein_2_graph.edge_index)

        return {
            "protein_1_graph": protein_1_graph_data,
            "protein_2_graph": protein_2_graph_data,
            "label": row["label"],
        }

def load_dataset_with_paths(data_dir):
    """
    Load dataset directly if it already contains graph paths.
    Args:
        data_dir (str): Path to the processed dataset directory.
    Returns:
        dict: A dictionary containing train, validation, and test datasets.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(data_dir)
    return dataset


def evaluate_one_epoch(model, criterion, dataloader, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move graphs to device
            protein_1_graph = batch["protein_1_graph"].to(device)
            protein_2_graph = batch["protein_2_graph"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            output = model(protein_1_graph, protein_2_graph)

            # Compute loss
            loss = criterion(output, labels.view(-1, 1).float())
            epoch_loss += loss.item()

            # Compute accuracy
            labels = labels.int()
            predictions = (output.squeeze(1) > 0.5).int()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Collect all predictions and labels for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return epoch_loss / len(dataloader), accuracy, precision, recall, f1


def test(data_dir, graph_type, device, model_name, model_dir, batch_size, dropout):
    # Load dataset with paths
    dataset = load_dataset_with_paths(data_dir)
    
    # Prepare datasets
    test_dataset = ProteinGraphDataset(dataset["test"], graph_type)
    
    # Use PyG's DataLoader
    dataloaders = {
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
    }

    criterion = nn.BCELoss()  # Binary Cross-Entropy for classification
    
    # Get the number of features for the current graph type
    num_features = graph_type_to_features.get(graph_type)

    if num_features is None:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Initialize model
    if model_name == "GCNN":
        model = ProteinInteractionGNN(num_features_pro=num_features, model_type="gcn", dropout=dropout).to(device)
    elif model_name == "AttGNN":
        model = ProteinInteractionGNN(num_features_pro=num_features, model_type="gat", dropout=dropout, heads=1).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    # Test evaluation
    best_model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    loss, accuracy, precision, recall, f1 = evaluate_one_epoch(model, criterion, dataloaders["test"], device)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")

def get_sorted_entries(directory):
    """Returns a list of directory entries sorted by last modified time."""
    entries = [entry for entry in os.scandir(directory) if entry.is_dir() or entry.name.endswith('.yaml')]
    return sorted(entries, key=lambda e: e.stat().st_mtime)

def parse_yaml(file_path):
    """Parses a YAML file and returns its content."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def extract_value(param):
    """Extracts the value from a parameter that might be nested."""
    if isinstance(param, dict) and 'value' in param:
        return param['value']
    return param

if __name__ == "__main__":
        # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process PPI directories and configurations.")
    parser.add_argument('--graph_type', type=str, required=True, help="Type of graph (e.g., onehot).")
    parser.add_argument('--model_type', type=str, required=True, help="Type of model (e.g., GCNN).")
    parser.add_argument('--sweep_id', type=str, required=True, help="Sweep ID for configuration YAMLs.")
    
    args = parser.parse_args()
    
    # Directories and parameters
    data_dir = "/blue/cnt5410/shaik.abdulkalam/PPI/data/processed"
    base_path = "/blue/cnt5410/shaik.abdulkalam/PPI/output/"
    sweep_dir = "/blue/cnt5410/shaik.abdulkalam/PPI/code/wandb/"
    
    graph_type = args.graph_type
    model_type = args.model_type
    sweep_id = args.sweep_id

    # Get sorted folders and YAML files
    base_directory = os.path.join(base_path, graph_type, model_type)
    yaml_dir = os.path.join(sweep_dir, sweep_id)
    
    sorted_folders = get_sorted_entries(base_directory)
    sorted_yaml_files = get_sorted_entries(yaml_dir)

    if len(sorted_folders) != len(sorted_yaml_files):
        print("Mismatch in the number of folders and YAML files. Please check your directories.")
        exit(1)

    # Pair and process
    for folder, yaml_file in zip(sorted_folders, sorted_yaml_files):
        print(f"Processing folder: {folder.path}, with YAML: {yaml_file.path}")
        
        # Extract configuration from YAML
        config = parse_yaml(yaml_file.path)
        batch_size = extract_value(config.get('batch_size', None))
        dropout = extract_value(config.get('dropout', None))

        # Call the test function
        test(
            data_dir=data_dir,
            graph_type=graph_type,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_name=model_type,
            model_dir=folder.path,
            batch_size=batch_size,
            dropout=dropout,
        )
        print(f"Finished processing: {folder.path}")
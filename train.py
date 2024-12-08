import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import json
import wandb
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


def train_one_epoch(model, optimizer, scheduler, criterion, dataloader, device):
    model.train()
    epoch_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move graphs to device
        protein_1_graph = batch["protein_1_graph"].to(device)
        protein_2_graph = batch["protein_2_graph"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()

        # Forward pass
        output = model(protein_1_graph, protein_2_graph)

        # Compute loss
        loss = criterion(output, labels.view(-1, 1).float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        #print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    scheduler.step()
    return epoch_loss / len(dataloader)


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
            
            # print(f"labels: {labels}")
            # print(f"predictions: {predictions}")
            # print(f"batch correct count: {(predictions == labels).sum().item()}")
            # print(f"batch labels count: {labels.size(0)}")
            # print(f"total correct till now: {correct}")
            # print(f"total labels till now: {total}")
            
            # Collect all predictions and labels for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return epoch_loss / len(dataloader), accuracy, precision, recall, f1


def train(data_dir, model_name, graph_type, learning_rate, weight_decay, step_size, gamma, num_epochs, batch_size, dropout, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset with paths
    dataset = load_dataset_with_paths(data_dir)
    
    # Prepare datasets
    train_dataset = ProteinGraphDataset(dataset["train"], graph_type)
    val_dataset = ProteinGraphDataset(dataset["validation"], graph_type)
    test_dataset = ProteinGraphDataset(dataset["test"], graph_type)
    
    # Use PyG's DataLoader
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        "validation": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
    }

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
        
    # Initialize optimizer, criterion, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.BCELoss()  # Binary Cross-Entropy for classification

    # Initialize training history
    history = {
        "train_loss": [], 
        "val_loss": [], 
        "val_accuracy": [], 
        "val_precision":[], 
        "val_recall":[], 
        "val_f1":[],
        "test_loss": [], 
        "test_accuracy": [], 
        "test_precision":[], 
        "test_recall":[], 
        "test_f1":[]
    }

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, optimizer, scheduler, criterion, dataloaders["train"], device)
        print(f"Training Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_one_epoch(model, criterion, dataloaders["validation"], device)
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        
        if wandb.run is not None:
            wandb.log({
                "train_loss": train_loss, 
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            })
        else:
            print(f"WandB not initialized.")

    # Test evaluation
    best_model_path = os.path.join(output_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_one_epoch(model, criterion, dataloaders["validation"], device)
    print(f"final val Loss: {val_loss:.4f} | final val Accuracy: {val_accuracy:.4f} | final val precision: {val_precision:.4f} | final val recall: {val_recall:.4f} | final val f1: {val_f1:.4f}")

    # Update history
    # history["test_loss"] = test_loss
    # history["test_accuracy"] = test_accuracy
    # history["test_precision"].append(test_precision)
    # history["test_recall"].append(test_recall)
    # history["test_f1"].append(test_f1)

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print("Training complete.")
    
    return {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a GNN model for protein-protein interaction prediction.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory containing the dataset.")
    parser.add_argument("--model_name", type=str, required=True, choices=["GCNN", "AttGNN"], help="Model to use.")
    parser.add_argument("--graph_type", type=str, required=True, choices=["onehot", "physchem", "expasy","protbert", "prostt5"], help="Graph type to use.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument("--step_size", type=int, default=20, help="Step size for StepLR scheduler.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for StepLR scheduler.")
    parser.add_argument("--num_epochs", type=int, default=6, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and history.")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_name=args.model_name,
        graph_type=args.graph_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        device=args.device,
        output_dir=args.output_dir,
    )

import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def process_and_split_dataset(data_dir, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Process dataset to include paths for all graph types, split into train/val/test, and save as a Hugging Face Dataset.
    Args:
        data_dir (str): Base directory for data.
        train_size (float): Proportion of training data.
        val_size (float): Proportion of validation data.
        test_size (float): Proportion of test data.
    """
    # Paths
    human_dataset_path = os.path.join(data_dir, "human_dataset.npy")
    graph_dir = os.path.join(data_dir, "graphs")
    output_path = os.path.join(data_dir, "processed")

    # Graph types to include
    graph_types = ["onehot", "physchem", "expasy", "protbert", "prostt5"]

    # Load the human dataset
    human_data = np.load(human_dataset_path, allow_pickle=True)

    dataset = []
    for row in tqdm(human_data, desc="Processing dataset"):
        protein_1_id, protein_2_id, label = row[2], row[5], int(row[6])

        # Store paths for all graph types
        protein_1_graphs = {
            graph_type: os.path.join(graph_dir, graph_type, f"{protein_1_id}.pt")
            for graph_type in graph_types
        }
        protein_2_graphs = {
            graph_type: os.path.join(graph_dir, graph_type, f"{protein_2_id}.pt")
            for graph_type in graph_types
        }

        # Verify that all graph files exist
        if not all(os.path.exists(path) for path in protein_1_graphs.values()) or \
           not all(os.path.exists(path) for path in protein_2_graphs.values()):
            print(f"Skipping pair {protein_1_id}, {protein_2_id} due to missing graphs.")
            continue

        # Add entry to dataset
        dataset.append({
            "protein_1_id": protein_1_id,
            "protein_2_id": protein_2_id,
            "protein_1_graph": protein_1_graphs,
            "protein_2_graph": protein_2_graphs,
            "label": label
        })

    # Create train, validation, and test splits
    train_data, test_data = train_test_split(
        dataset, test_size=1-train_size, random_state=42, stratify=[d["label"] for d in dataset]
    )
    val_data, test_data = train_test_split(
        test_data, test_size=test_size/(test_size + val_size), random_state=42, stratify=[d["label"] for d in test_data]
    )

    # Convert to Hugging Face DatasetDict
    hf_dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })

    # Save dataset
    os.makedirs(output_path, exist_ok=True)
    hf_dataset.save_to_disk(output_path)
    print(f"Dataset with splits saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset with graph paths and splits.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory for data (e.g., 'data').")
    args = parser.parse_args()
    
    process_and_split_dataset(args.data_dir)


if __name__ == "__main__":
    main()

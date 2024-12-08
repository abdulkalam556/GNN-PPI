import wandb
from train import train
import argparse
import torch
import os

# Objective function for training
def objective():
    # Initialize W&B for this run
     with wandb.init() as run:
        config = run.config  # Contains sweep parameters + fixed arguments
        
        os.makedirs(config.output_dir, exist_ok=True)
        graph_type_dir = os.path.join(config.output_dir, config.graph_type)
        os.makedirs(graph_type_dir, exist_ok=True)
        model_name_dir = os.path.join(graph_type_dir, config.model_name)
        os.makedirs(model_name_dir, exist_ok=True)

        # Define training arguments based on W&B sweep configuration
        metrics = train(
            data_dir=config.data_dir,
            model_name=config.model_name,
            graph_type=config.graph_type,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            step_size=config.step_size,
            gamma=config.gamma,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            dropout=config.dropout,
            device=config.device,
            output_dir=os.path.join(model_name_dir, f"{run.id}"),
        )

        # Log metrics to W&B
        wandb.log(metrics)

# Main function to set up W&B Sweep and agent
def main(args):
    # Log into W&B
    with open("wandb_key.txt", "r") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    # Define the sweep configuration
    sweep_configuration = {
        "method": "bayes",  # Using Bayesian Optimization
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
            "step_size": {"distribution": "int_uniform", "min": 10, "max": 50},
            "gamma": {"distribution": "uniform", "min": 0.1, "max": 0.9},
            "batch_size": {"values": [8, 16, 32, 64]},
            "dropout": {"distribution": "uniform", "min": 0.1, "max": 0.5},
            "num_epochs": {"value": 7}  # Fixed number of epochs
        },
    }
    
    # Add fixed parameters to the sweep config
    sweep_configuration["parameters"].update({
        "data_dir": {"value": args.data_dir},
        "model_name": {"value": args.model_name},
        "graph_type": {"value": args.graph_type},
        "output_dir": {"value": args.output_dir},
        "device": {"value": "cuda" if torch.cuda.is_available() else "cpu"},
    })

    # Start the W&B Sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"PPI_{args.graph_type}_{args.model_name}")

    # Start the sweep agent with the specified objective function
    wandb.agent(sweep_id, function=objective, count=10)

# Execute main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a WandB sweep for GNN training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory containing the dataset.")
    parser.add_argument("--model_name", type=str, required=True, choices=["GCNN", "AttGNN"], help="Model to use.")
    parser.add_argument("--graph_type", type=str, required=True, choices=["onehot", "physchem", "expasy", "protbert", "prostt5"], help="Graph type to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and history.")
    args = parser.parse_args()
    
    main(args)

# -*- coding: utf-8 -*-
"""
fl_server.py

Defines the Flower server logic for federated fine-tuning, including the aggregation strategy (FedAvg).
Sets up and runs the Flower simulation.
"""

import flwr as fl
import torch
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitRes, Parameters, Scalar, EvaluateRes, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import logging
import numpy as np
from config import DATA_DIR
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming model.py, data_utils.py, fl_client.py are accessible
from .model import BitNetGainCellLLM, BitNetGainCellConfig
from .data_utils import load_full_texts_csv, preprocess_and_tokenize, partition_data
from .fl_client import client_fn, get_lora_config, get_peft_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Custom FedAvg Strategy (Optional, for potential modifications) ---
# We can use the default FedAvg, but a custom strategy allows for more control,
# e.g., logging, custom aggregation, or handling LoRA parameters specifically.

class FedAvgLora(FedAvg):
    """Custom FedAvg strategy to handle LoRA parameters and logging."""
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate LoRA parameter updates using weighted average."""
        logging.info(f"Server: Aggregating fit results for round {server_round}")
        
        # Call the parent FedAvg aggregate_fit method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            logging.info(f"Server: Aggregation successful for round {server_round}")
            # Convert aggregated NumPy arrays back to Parameters object
            # aggregated_parameters_obj = fl.common.ndarrays_to_parameters(aggregated_parameters)
        else:
            logging.warning(f"Server: Aggregation failed for round {server_round}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses."""
        logging.info(f"Server: Aggregating evaluation results for round {server_round}")
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        if loss_aggregated is not None:
            logging.info(f"Server: Round {server_round} aggregated evaluation loss: {loss_aggregated:.4f}")
        else:
             logging.warning(f"Server: Evaluation aggregation failed for round {server_round}")

        # CSV saving
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, 'aggregate_server_metrics.csv')
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["round", "loss"])
            writer.writerow([server_round, loss_aggregated])

        return loss_aggregated, metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        
        # Add standard FL parameters like learning rate, epochs for this round
        # These can be fixed or vary per round
        config["learning_rate"] = 1e-4 # Example LR
        config["local_epochs"] = 1      # Example local epochs
        config["current_round"] = server_round

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

# --- Main Simulation Function ---
def run_simulation(num_rounds=5, num_clients=16, batch_size=8, data_dir= DATA_DIR):
    """
    Sets up and runs the Flower simulation.

    Args:
        num_rounds (int): Number of federated learning rounds.
        num_clients (int): Total number of clients available for simulation.
        batch_size (int): Batch size for client dataloaders.
        data_dir (str, optional): Directory containing MIMIC data. Defaults to None, uses DEFAULT_DATA_DIR.
    """
    logging.info("--- Starting Federated Learning Simulation --- ")
    
    # 1. Load and Prepare Data
    if data_dir is None:
        data_dir = "/home/ubuntu/federated_llm/data" # Use default if not provided
    
    mimic_df = load_full_texts_csv(csv_path=data_dir)
    if mimic_df is None or mimic_df.empty:
        logging.error("Failed to load data. Aborting simulation.")
        return

    # --- test set ---
    trainval_df, test_df = train_test_split(mimic_df, test_size=0.1, random_state=42, shuffle=True)
    logging.info(f"Train/Val set: {len(trainval_df)}, Test set: {len(test_df)}")

    # --- train/val tokenizing ---
    tokenized_data = preprocess_and_tokenize(trainval_df)
    client_datasets = partition_data(tokenized_data, num_clients=num_clients, alpha=0.5)
    logging.info(f"Data loaded, tokenized, and partitioned for {num_clients} clients.")

    # --- test set tokenizing ---
    test_tokenized = preprocess_and_tokenize(test_df)
    # test set csv saving (for prompt test)
    test_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test_set_for_prompt.csv'), index=False)

    # or tokenized test set saving (for model evaluation)
    import pickle
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test_tokenized.pkl'), 'wb') as f:
        pickle.dump(test_tokenized, f)

    # 2. Define Model Configuration
    # Use a smaller config for faster simulation if needed, otherwise use defaults
    model_config = BitNetGainCellConfig(
        block_size=4096, # official config
        vocab_size=128256, # official config
        n_layer=30,     # official config
        n_head=20,      # official config
        n_embd=2560,    # official config
        dropout=0.0,    # official config
        bias=True,      # official config
        window_size=64, # custom
        stride=32       # custom
    )
    logging.info(f"Using model config: {model_config}")

    # 3. Define Federated Learning Strategy (FedAvg for LoRA)
    # We need initial parameters, but LoRA parameters are created *inside* the client.
    # For FedAvg with PEFT, the server often doesn't need initial PEFT params.
    # It sends global model params (which are frozen) and aggregates PEFT params.
    # However, Flower's FedAvg expects *some* initial parameters.
    # Let's provide dummy parameters initially, or modify the strategy.
    # A better approach: Initialize the strategy *without* initial parameters, 
    # and let the first round's parameters come from the clients.
    # Or, initialize a dummy PEFT model on the server just to get parameter shapes.
    
    # Dummy model to get initial parameter structure (only LoRA weights)
    temp_model = BitNetGainCellLLM(model_config)
    lora_conf = get_lora_config() # Defined in fl_client.py
    temp_peft_model = get_peft_model(temp_model, lora_conf)
    initial_lora_params = [val.cpu().numpy() for name, val in temp_peft_model.named_parameters() if "lora_" in name]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_lora_params)
    logging.info(f"Initialized strategy with {len(initial_lora_params)} dummy LoRA parameters.")
    del temp_model, temp_peft_model # Free memory

    strategy = FedAvgLora(
        fraction_fit=1.0,  # Sample 100% of clients for training each round
        fraction_evaluate=1.0, # Sample 100% for evaluation (can reduce if needed)
        min_fit_clients=num_clients, # Minimum clients for training
        min_evaluate_clients=num_clients, # Minimum clients for evaluation
        min_available_clients=num_clients, # Wait for all clients to be available
        initial_parameters=initial_parameters,
        # evaluate_fn=get_evaluate_fn(model_config, device), # Optional: Server-side evaluation
        # on_fit_config_fn=fit_config, # Optional: Function to configure client training each round
    )
    logging.info("Federated learning strategy initialized (FedAvgLora).")

    # 4. Define Client Resources
    # Determine device and resources per client for simulation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Simulation device: {device}")
    client_resources = None
    if device.type == "cuda":
        # Assign GPU resources if available. Adjust num_gpus based on your system.
        # Example: Allow each client to use 1 GPU if available
        # Flower simulation handles resource allocation.
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
             gpu_per_client = 1 # Assign 1 GPU per client if possible
             # Ensure we don't request more GPUs than available per client instance
             # This depends on how many clients run concurrently. Flower handles this.
             client_resources = {"num_cpus": 2, "num_gpus": gpu_per_client}
             logging.info(f"Assigning {gpu_per_client} GPU per client.")
        else:
             client_resources = {"num_cpus": 2, "num_gpus": 0}
             logging.warning("No GPUs detected by PyTorch. Running on CPU.")
    else:
        client_resources = {"num_cpus": 2, "num_gpus": 0}
        logging.info("Running on CPU.")

    # 5. Start Simulation
    logging.info(f"Starting Flower simulation with {num_clients} clients for {num_rounds} rounds.")
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, model_config, client_datasets, batch_size, device),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": "ray.util.accelerate.accelerate_torch_on_actor", # If using accelerate with Ray
        }
    )

    # --- Simulation Finished ---
    logging.info("--- Simulation Finished --- ")
    logging.info(f"History (losses_distributed): {history.losses_distributed}")

    # Extract final LoRA parameters (example: assume last parameters from history)
    # In practice, the final parameters should be obtained from the strategy or server object
    final_lora_params = strategy.parameters  # Modify as needed

    # test set evaluation
    evaluate_on_testset(model_config, final_lora_params, test_tokenized, batch_size, device)

    # Save final model
    base_model = BitNetGainCellLLM(model_config).to(device)
    lora_config = get_lora_config()
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.eval()
    # Apply LoRA parameters
    lora_param_names = [name for name, _ in model.named_parameters() if "lora_" in name]
    params_dict = zip(lora_param_names, final_lora_params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=False)
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'finetuned_model')
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    # Save tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")  # Use actual tokenizer name
    tokenizer.save_pretrained(save_dir)
    logging.info(f"Final fine-tuned model saved to {save_dir}")

    return history

def evaluate_on_testset(model_config, lora_params, test_tokenized, batch_size, device):
    """
    Evaluate the model on the test set after final federated learning.
    model_config: model configuration
    lora_params: final LoRA parameters (aggregated by the server)
    test_tokenized: tokenized test set
    batch_size: batch size
    device: device for evaluation
    """
    from torch.utils.data import DataLoader
    from fl_client import get_peft_model, get_lora_config
    import torch
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    # 1. Create model and apply LoRA
    base_model = BitNetGainCellLLM(model_config).to(device)
    lora_config = get_lora_config()
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.eval()

    # 2. Apply LoRA parameters
    lora_param_names = [name for name, _ in model.named_parameters() if "lora_" in name]
    params_dict = zip(lora_param_names, lora_params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=False)

    # 3. Prepare DataLoader
    test_loader = DataLoader(test_tokenized, batch_size=batch_size)

    # 4. Evaluation loop
    total_loss = 0
    num_examples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(idx=batch["input_ids"], targets=batch["input_ids"])
            logits = outputs[0]
            loss = outputs[1]
            if loss is not None:
                total_loss += loss.item() * batch["input_ids"].size(0)
                num_examples += batch["input_ids"].size(0)
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy().flatten()
                labels = batch["input_ids"].detach().cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels)
    avg_loss = total_loss / num_examples if num_examples > 0 else 0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
    perplexity = float(np.exp(avg_loss)) if avg_loss < 20 else float('inf')

    print(f"Test set evaluation results: Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}, Perplexity={perplexity:.4f}")
    return avg_loss, acc, f1, perplexity

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Run the simulation with specified parameters
    # Adjust parameters as needed
    run_simulation(
        num_rounds=20,      # Keep low for testing
        num_clients=16,     # Use fewer clients for faster testing
        batch_size=4,      # Smaller batch size for testing
        # data_dir="/path/to/your/mimic/csvs" # Optional: Specify data directory if not default
    )


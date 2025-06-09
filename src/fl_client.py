import flwr as fl
import torch
import torch.nn as nn
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model import BitNetGainCellLLM, BitNetGainCellConfig
from data_utils import load_mimic_data, preprocess_and_tokenize, partition_data
from peft import get_peft_model, LoraConfig, TaskType
import torch.cuda.amp as amp

logging.basicConfig(level=logging.INFO)

def get_lora_config() -> LoraConfig:
    """Return LoRA config"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
    )

class FlowerClient(fl.client.NumPyClient):
    """Optimized Flower client"""
    def __init__(
        self,
        cid: str,
        model: BitNetGainCellLLM,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scaler for mixed precision training
        self.scaler = amp.GradScaler()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.trainloader) * 5,  # 5 epochs
            eta_min=1e-6
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Extract LoRA parameters"""
        return [val.cpu().numpy() for name, val in self.model.named_parameters() if "lora_" in name]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set LoRA parameters"""
        lora_params = [name for name, _ in self.model.named_parameters() if "lora_" in name]
        params_dict = zip(lora_params, parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Union[int, float]]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Optimized local training"""
        self.set_parameters(parameters)
        self.model.train()
        
        # Training setup
        epochs = config.get("local_epochs", 1)
        batch_acc = []
        batch_losses = []
        num_examples = 0
        
        for _ in range(epochs):
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Mixed precision training
                with amp.autocast():
                    outputs = self.model(idx=batch["input_ids"], targets=batch["input_ids"])
                    logits = outputs[0]
                    loss = outputs[1]
                
                if loss is not None:
                    # Gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    # Metric calculation
                    batch_size = batch["input_ids"].size(0)
                    num_examples += batch_size
                    batch_losses.append(loss.item())
                    
                    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy().flatten()
                    labels = batch["input_ids"].detach().cpu().numpy().flatten()
                    batch_acc.append(accuracy_score(labels, preds))

        # Average metric calculation
        metrics = {
            "loss": np.mean(batch_losses),
            "accuracy": np.mean(batch_acc),
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
        return self.get_parameters({}), num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Union[int, float]]
    ) -> Tuple[float, int, Dict]:
        """Optimized evaluation"""
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0
        num_examples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.valloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with amp.autocast():
                    outputs = self.model(idx=batch["input_ids"], targets=batch["input_ids"])
                    logits = outputs[0]
                    loss = outputs[1]
                
                if loss is not None:
                    batch_size = batch["input_ids"].size(0)
                    total_loss += loss.item() * batch_size
                    num_examples += batch_size
                    
                    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy().flatten()
                    labels = batch["input_ids"].detach().cpu().numpy().flatten()
                    all_preds.extend(preds)
                    all_labels.extend(labels)

        # Metric calculation
        avg_loss = total_loss / num_examples if num_examples > 0 else float('inf')
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "perplexity": float(np.exp(avg_loss)) if avg_loss < 20 else float('inf')
        }
        
        return avg_loss, num_examples, metrics

def client_fn(
    cid: str,
    model_config: BitNetGainCellConfig,
    client_datasets: List,
    batch_size: int,
    device: torch.device
) -> FlowerClient:
    """Client factory function"""
    # Dataset split
    train_size = int(0.8 * len(client_datasets[int(cid)]))
    val_size = len(client_datasets[int(cid)]) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        client_datasets[int(cid)],
        [train_size, val_size]
    )
    
    # Create dataloader
    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = BitNetGainCellLLM(model_config).to(device)
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Return client
    return FlowerClient(cid, model, trainloader, valloader, device)

# Example execution
if __name__ == '__main__':
    logging.info("--- Running FL Client Example (Conceptual) ---")
    
    # Setup
    config = BitNetGainCellConfig(
        block_size=4096,
        vocab_size=128256,
        n_layer=30,
        n_head=20,
        n_embd=2560,
        dropout=0.0,
        bias=True,
        window_size=64,
        stride=32,
        use_flash_attention=True,
        use_checkpoint=True,
        mixed_precision=True,
        gradient_checkpointing=True
    )
    
    # Load and preprocess data
    mimic_df = load_mimic_data(data_dir="data", use_cache=True)
    tokenized_data = preprocess_and_tokenize(
        mimic_df,
        tokenizer_name="microsoft/bitnet-b1.58-2B-4T-bf16",
        max_length=config.block_size
    )
    client_datasets = partition_data(tokenized_data, num_clients=10, alpha=0.5)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create client and test
    client = client_fn("0", config, client_datasets, batch_size=4, device=device)
    logging.info("Client instance created.")
    
    # Get parameters/test setup
    initial_params = client.get_parameters({})
    logging.info(f"Got {len(initial_params)} initial LoRA parameters.")
    
    # Training test
    fit_config = {"local_epochs": 1, "learning_rate": 1e-4}
    updated_params, num_examples_train, metrics_train = client.fit(initial_params, fit_config)
    logging.info(f"Fit completed. Trained on {num_examples_train} examples. Got {len(updated_params)} updated parameters.")
    
    # Evaluation test
    eval_loss, num_examples_eval, metrics_eval = client.evaluate(updated_params, {})
    logging.info(f"Evaluate completed. Loss: {eval_loss:.4f} on {num_examples_eval} examples.")
    
    logging.info("--- FL Client Example Finished ---")


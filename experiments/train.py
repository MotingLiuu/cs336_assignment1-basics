from cs336_basics import model as mt_model
from cs336_basics import BPETokenizer
import wandb
import datetime
import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import typing as npt
from einops import rearrange, einsum
import os
from os import PathLike
from pathlib import Path
from typing import BinaryIO, IO
from jaxtyping import Float, Int
from collections.abc import Iterable, Callable
import logging
import math
import argparse

current_dir = Path(__file__).parent
run_name = f"train_model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" + "training"

os.environ["WANDB_MODE"] = "offline"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=current_dir / f"train_model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
)



def get_args():
    parser = argparse.ArgumentParser(description="Train a model with AdamW optimizer.")
    
    parser.add_argument("--log_step", type=int, default=100, help="Number of steps between logging.")
    parser.add_argument("--checkpoint_save_step", type=int, default=1000, help="Number of steps between saving checkpoints.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_batches", type=int, default=1000, help="Number of batches to train on.")
    parser.add_argument("--valid_batch_size", type=int, default=32, help="Batch size for validation.")
    parser.add_argument("--checkpoint_path", type=str, default=None , help="Path to save the checkpoint.")
    parser.add_argument("--checkpoint_folder", type=str, default="./checkpoints", help="Path to save the model and optimizer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")

    data_group = parser.add_argument_group("Data arguments")
    data_group.add_argument("--train_data", type=str, required=True, help="Path to the training data.")
    data_group.add_argument("--valid_data", type=str, required=True, help="Path to the validation data.")

    model_group = parser.add_argument_group("Model arguments")
    model_group.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    model_group.add_argument("--d_ff", type=int, default=2048, help="Dimension of the feed-forward network.")
    model_group.add_argument("--n_layers", type=int, default=6, help="Number of layers in the model.")
    model_group.add_argument("--n_heads", type=int, default=8, help="Number of attention heads.")
    model_group.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary.")
    model_group.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length.")
    model_group.add_argument("--theta", type=float, default=10000.0, help="Theta value for Rope.")
    

    optimizer_group = parser.add_argument_group("Optimizer arguments")
    optimizer_group.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    optimizer_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    optimizer_group.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="Betas for the AdamW optimizer.")
    optimizer_group.add_argument("--eps", type=float, default=1e-8, help="Epsilon for the AdamW optimizer.")
    optimizer_group.add_argument("--optimizer_type", type=str, default="adamw", choices=["adamw", "sgd"], help="Type of optimizer to use.")
    optimizer_group.add_argument("--num_iters", type=int, default=1000, help="Number of optimization iterations.")
    
    args = parser.parse_args()
    return args
    
def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: str | PathLike | BinaryIO | IO[bytes],
    valid_data: str | PathLike | BinaryIO | IO[bytes],
    num_batches: int = 1,
    batch_size: int = 1,
    valid_batch_size: int = 1,
    log_step: int = 100,
    checkpoint_save_step: int = 1000,
    checkpoint_folder: str = "./checkpoints",
    curr_iter: int = 0,
    device: str = "cpu",
):
    wandb.init(
        project="cs336-assignment1",
        name=run_name,
        config={
            "num_batches": num_batches,
            "batch_size": batch_size,
            "log_step": log_step,
            "curr_iter": curr_iter,
            "device": device,
            "model_parameters": {
                "d_model": model.d_model,
                "d_ff": model.d_ff,
                "n_layers": model.num_layers,
                "n_heads": model.num_heads,
                "vocab_size": model.vocab_size,
                "max_seq_len": model.max_seq_len,
                "theta": model.theta
            },
            "optimizer_parameters": {
                "lr": optimizer.defaults['lr'],
                "weight_decay": optimizer.defaults['weight_decay'],
                "betas": optimizer.defaults['betas'],
                "eps": optimizer.defaults['eps'],
                "optimizer_type": type(optimizer).__name__
            }
        }
        )
    train_data = current_dir / train_data
    valid_data = current_dir / valid_data
    mmap_train_data = np.load(train_data, mmap_mode="r")
    mmap_valid_data = np.load(valid_data, mmap_mode="r")
    model.to(device)
    model.train()
    for iter in range(curr_iter + 1, num_batches + 1):
        batch_tensor, target_tensor = model.data_loading(
            dataset_encoded=mmap_train_data,
            batch_size=batch_size,
            context_length=model.max_seq_len,
            device=device,
        )
        optimizer.zero_grad()
        logits = model(batch_tensor)
        loss = mt_model.cross_entropy_loss(
            logits=logits,
            targets=target_tensor
        )
        loss.backward()
        optimizer.step()
        
        if iter % log_step == 0:
            model.eval()
            with torch.no_grad():
                valid_batch_tensor, valid_target_tensor = model.data_loading(
                    dataset_encoded=mmap_valid_data,
                    batch_size=valid_batch_size,
                    context_length=model.max_seq_len,
                    device=device,
                )
                valid_logits = model(valid_batch_tensor)
                valid_loss = mt_model.cross_entropy_loss(
                    logits=valid_logits,
                    targets=valid_target_tensor
                )
            logger.info(f"Iteration {iter}, Loss: {loss.item()}, Valid Loss: {valid_loss.item()}")
            wandb.log({"train_loss": loss.item(), "valid_loss": valid_loss.item(), "iteration": iter})
            model.train()
            
        if iter % checkpoint_save_step == 0:
            checkpoint_path =  current_dir / checkpoint_folder
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_path / f"checkpoint_iter_{iter}.pth"
            mt_model.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iter,
                out=checkpoint_path
            )

if __name__ == "__main__":
    args = get_args()
    model = mt_model.Transformer(
        d_model=args.d_model,
        num_heads=args.n_heads,
        d_ff=args.d_ff,
        num_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        theta=args.theta
    )
    
    if args.optimizer_type == "adamw":
        optimizer = mt_model.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    elif args.optimizer_type == "sgd":
        optimizer = mt_model.SGD(
            model.parameters(),
            lr=args.lr,
        )
        
    current_iter = 0
    if args.checkpoint_path:
        current_iter = mt_model.load_checkpoint(
            src=args.checkpoint_path,
            model=model,
            optimizer=optimizer,
        )

    train(
        model=model,
        optimizer=optimizer,
        train_data=args.train_data,
        valid_data=args.valid_data,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        log_step=args.log_step,
        checkpoint_save_step=args.checkpoint_save_step,
        checkpoint_folder=args.checkpoint_folder,
        curr_iter=current_iter,  
        device=args.device
    )
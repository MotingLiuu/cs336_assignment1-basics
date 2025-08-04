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
import yaml
import math
import argparse

current_dir = Path(__file__).parent

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
    parser.add_argument("--checkpoint_path", type=str, default=None , help="Path to save the checkpoint.")
    parser.add_argument("--checkpoint_folder", type=str, default="./checkpoints", help="Path to save the model and optimizer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")

    data_group = parser.add_argument_group("Data arguments")
    data_group.add_argument("--train_data", type=str, help="Path to the training data.")
    data_group.add_argument("--valid_data", type=str, help="Path to the validation data.")

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
    optimizer_group.add_argument("--beta1", type=float, default=0.9, help="Beta1 for the AdamW optimizer.")
    optimizer_group.add_argument("--beta2", type=float, default=0.999, help="Beta2 for the AdamW optimizer.")
    optimizer_group.add_argument("--eps", type=float, default=1e-8, help="Epsilon for the AdamW optimizer.")
    optimizer_group.add_argument("--optimizer_type", type=str, default="adamw", choices=["adamw", "sgd"], help="Type of optimizer to use.")
    

    scheduer_group = parser.add_argument_group("Scheduler arguments")
    scheduer_group.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "linear"], help="Type of learning rate scheduler to use.")
    scheduer_group.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the learning rate scheduler.")
    scheduer_group.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for the scheduler.")
    scheduer_group.add_argument("--max_lr", type=float, default=1e-3, help="Maximum learning rate for the scheduler.")
    scheduer_group.add_argument("--cosine_annealing_steps", type=int, default=1000, help="Number of steps for the learning rate decay.")

    args = parser.parse_args()
    return args
    
def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: str | PathLike | BinaryIO | IO[bytes],
    valid_data: str | PathLike | BinaryIO | IO[bytes],
    scheduler: mt_model.LRScheduler | None = None,
    num_batches: int = 1,
    batch_size: int = 1,
    log_step: int = 100,
    checkpoint_save_step: int = 1000,
    checkpoint_folder: str = "./checkpoints",
    curr_iter: int = 0,
    device: str = "cpu",
):
    wandb.init(
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
            },
            "scheduler_parameters": {
                "lr_scheduler": type(scheduler).__name__ if scheduler else None,
                "warmup_steps": scheduler.warmup_steps if scheduler else None,
                "min_lr": scheduler.min_lr if scheduler else None,
                "max_lr": scheduler.max_lr if scheduler else None,
                "cosine_annealing_steps": scheduler.cosine_annealing_steps if scheduler else None
            }
        }
        )
    run_id = wandb.run.id
    checkpoint_folder_with_id = Path(checkpoint_folder) / f"{run_id}"
    mmap_train_data = np.load(train_data, mmap_mode="r")
    mmap_valid_data = np.load(valid_data, mmap_mode="r")
    model.to(device)
    model.train()
    for iter in range(curr_iter + 1, num_batches + 1):
        batch_tensor, target_tensor = mt_model.data_loading(
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
        if scheduler is not None:
            scheduler.step(iter)
        optimizer.step()
        
        if iter % log_step == 0:
            model.eval()
            with torch.no_grad():
                valid_batch_tensor, valid_target_tensor = mt_model.data_loading(
                    dataset_encoded=mmap_valid_data,
                    batch_size=batch_size,
                    context_length=model.max_seq_len,
                    device=device,
                )
                valid_logits = model(valid_batch_tensor)
                valid_loss = mt_model.cross_entropy_loss(
                    logits=valid_logits,
                    targets=valid_target_tensor
                )
            logger.info(f"Iteration {iter}, Loss: {loss.item()}, Valid Loss: {valid_loss.item()}")
            wandb.log({
                "train_loss": loss.item(), 
                "valid_loss": valid_loss.item(), 
                "iteration": iter,
                "learning_rate": optimizer.param_groups[0]['lr'],
                })
            model.train()
            
        if iter % checkpoint_save_step == 0:
            checkpoint_path =  current_dir / checkpoint_folder_with_id
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
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    elif args.optimizer_type == "sgd":
        optimizer = mt_model.SGD(
            model.parameters(),
            lr=args.lr,
        )
        
    if args.lr_scheduler == "cosine":
        scheduler = mt_model.CosineLRScheduler(
            optimizer=optimizer,
            warmup_steps=args.warmup_steps,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            cosine_annealing_steps=args.cosine_annealing_steps
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
        scheduler=scheduler,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        log_step=args.log_step,
        checkpoint_save_step=args.checkpoint_save_step,
        checkpoint_folder=args.checkpoint_folder,
        curr_iter=current_iter,  
        device=args.device
    )
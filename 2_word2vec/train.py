import math
import os
import random
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import TypeAlias

import datasets
import lightning as L
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from IPython.display import HTML, display
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from model import W2V_CBOW
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import StripAccents
from tokenizers.trainers import BpeTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

W2V: TypeAlias = W2V_CBOW


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def CBOW_collate_fn(
    batch: list[str], chunk_size: int, neighborhood_size: int
) -> dict[str, list[torch.Tensor]]:
    batch_input, batch_output = [], []
    batch_encoding = tokenizer.encode_batch(batch)
    for encoding in batch_encoding:
        if len(encoding.ids) < chunk_size:
            continue
        seq_len = len(encoding.ids)
        n_possible_chunks = seq_len // chunk_size
        for i in range(0, n_possible_chunks * chunk_size, chunk_size):
            chunk = encoding.ids[i : i + chunk_size]
            output = chunk.pop(neighborhood_size)
            batch_input.append(chunk)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input)
    batch_output = torch.tensor(batch_output)

    return {"features": batch_input, "labels": batch_output}


def initialize_model_weights(model: W2V_CBOW, initialize_mode: str = "equal") -> None:
    if initialize_mode == "equal":
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            else:
                param.data.normal_(std=1.0 / math.sqrt(param.shape[1]))
    else:
        raise ValueError(f"Unsupport initialize mode: {initialize_mode}")


def test_step(model: W2V, test_dl: DataLoader, loss_fn: nn.CrossEntropyLoss):
    model.eval()
    size = len(test_dl.dataset)
    num_batches = len(test_dl)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in test_dl:
            y_hat = model(X)
            test_loss += loss_fn(y_hat, y).item()
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    logger.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def gen_epoch_str(epoch_idx: int) -> str:
    ret_str = f"Epoch: {epoch_idx} "
    str_size = len(ret_str)
    term_size = shutil.get_terminal_size()
    fill_size = term_size - str_size - 1
    return ret_str + fill_size * "_"


def train(
    model: W2V_CBOW, train_dl: DataLoader, test_dl: DataLoader, hyperparams: dict
) -> None:
    assert hyperparams is not None, "Must supply a dictionary"
    metrics_logging_path = Path("./runs").absolute()
    metrics_logger = SummaryWriter(log_dir=metrics_logging_path)
    ds_size = len(train_dl.dataset)
    lr = hyperparams["lr"]
    chkpt_inter = hyperparams["checkpoint_interval"]
    n_epochs = hyperparams["n_training_epochs"]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    chkpt_idx = 0
    run_id = str(datetime.now()).replace(" ", "_")
    chkpt_dir = Path(f"./checkpoints/{run_id}")
    if not chkpt_dir.exists():
        chkpt_dir.mkdir(parents=True)
    for epoch_idx in range(n_epochs):
        logger.info(f"\nEpoch: {epoch_idx}\n")
        for batch_idx, (X, y) in enumerate(train_dl):
            print(X)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % hyperparams["logging_interval"] == 0:
                loss, current = loss.item(), batch_idx * ds_size + len(X)
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{ds_size:>5d}]")

            if batch_idx % chkpt_inter == 0:
                logger.info("Saving checkpoint!")
                torch.save(model.state_dict(), f"{run_id}/checkpoint_{chkpt_idx}.pth")
                chkpt_idx += 1
        test_step(model, test_dl, loss_fn)


if __name__ == "__main__":
    logger.info("__/\\\______________/\\\____/\\\\\\\\\______/\\\________/\\\_")
    logger.info(" _\/\\\_____________\/\\\__/\\\///////\\\___\/\\\_______\/\\\_ ")
    logger.info("  _\/\\\_____________\/\\\_\///______\//\\\__\//\\\______/\\\__ ")
    logger.info(" v _\//\\\____/\\\____/\\\____________/\\\/____\//\\\____/\\\___")
    logger.info("    __\//\\\__/\\\\\__/\\\__________/\\\//_______\//\\\__/\\\____")
    logger.info("     ___\//\\\/\\\/\\\/\\\________/\\\//___________\//\\\/\\\_____")
    logger.info("      ____\//\\\\\\//\\\\\_______/\\\/_______________\//\\\\\______")
    logger.info("       _____\//\\\__\//\\\_______/\\\\\\\\\\\\\\\______\//\\\_______")
    logger.info("        ______\///____\///_______\///////////////________\///________")

    logger.info(
        "\n=====================================================================\n"
    )

    MODE = "CBOW"
    hyperparams = {
        "seed": 577,
        "test_set_proportion": 0.05,
        "batch_size": 4,
        "n_dataloader_workers": 8,
        "neighborhood_size": 2,
        "embedding_dim": 128,
        "validation_interval": 10,
        "checkpoint_interval": 5,
        "logging_interval": 10,
        "n_training_epochs": 3,
        "lr": 1e-3,
    }
    logger.info("Training configuration:\n")
    logger.info(pprint(hyperparams))
    chunk_size = hyperparams["neighborhood_size"] * 2 + 1

    # Set random number generator seed
    logger.info(f"Setting seeds with value: {hyperparams['seed']}")
    set_seeds(hyperparams["seed"])

    # Prepare dataset
    dataset_id = "wikimedia/wikipedia"
    logger.info(f"Retrieving dataset: {dataset_id}")
    ds = load_dataset(dataset_id, "20231101.en")
    logger.info("Producing train/test splits")
    ds = ds["train"].train_test_split(test_size=hyperparams["test_set_proportion"])

    # Load pre-trained tokenizer
    tokenizer_path = Path("models/tokenizer.json")
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    with open(tokenizer_path, "r") as f:
        tokenizer_json = f.read()
    tokenizer = Tokenizer.from_str(tokenizer_json)
    logger.info("Tokenizer loaded!")

    # Define PyTorch DataLoaders
    bs = hyperparams["batch_size"]
    train_dataset = ds["train"]
    test_dataset = ds["test"]
    logger.info("Preparing collation function")
    collate_fn = partial(
        CBOW_collate_fn,
        chunk_size=chunk_size,
        neighborhood_size=hyperparams["neighborhood_size"],
    )
    logger.info("Creating training dataloader")
    train_dl = DataLoader(
        train_dataset["text"],
        batch_size=bs,
        collate_fn=collate_fn,
        num_workers=hyperparams["n_dataloader_workers"],
    )
    logger.info("Creating testing dataloader")
    test_dl = DataLoader(
        test_dataset["text"],
        batch_size=bs,
        collate_fn=collate_fn,
        num_workers=hyperparams["n_dataloader_workers"],
    )
    loop_eval_dl = DataLoader(
        train_dataset[0:100]["text"],
        batch_size=bs,
        collate_fn=collate_fn,
        num_workers=hyperparams["n_dataloader_workers"],
    )

    # Train
    logger.info("Insantiating model")
    model = W2V_CBOW(
        tokenizer=tokenizer,
        embedding_dim=hyperparams["embedding_dim"],
        neighborhood_size=hyperparams["neighborhood_size"],
    )
    logger.info("Initializing model weights")
    initialize_model_weights(model)
    logger.info("Beginning training loop...")
    train(
        model=model,
        train_dl=loop_eval_dl,
        test_dl=loop_eval_dl,
        hyperparams=hyperparams,
    )

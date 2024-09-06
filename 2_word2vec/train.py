import datetime
import math
import os
import random
import shutil
import time
import toml
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pprint
from time import perf_counter
from timeit import default_timer as timer
from typing import TypeAlias, Callable, TypeVar

from rich.console import Console
from rich.table import Table

import lightning as L
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import tokenizers
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset, Dataset
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
from tqdm import tqdm

W2V: TypeAlias = W2V_CBOW


class TermColors:
    GOOD = '\033[92m'
    BAD = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# noinspection SpellCheckingInspection
def cbow_collate_fn(
        batch: list[dict], chunk_size: int, neighborhood_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_input, batch_output = [], []
    for sample in batch:
        encoding = tokenizer.encode(sample["text"])
        if len(encoding.ids) < chunk_size:
            continue
        seq_len = len(encoding.ids)
        n_possible_chunks = seq_len // chunk_size
        for i in range(0, n_possible_chunks * chunk_size, chunk_size):
            chunk = encoding.ids[i: i + chunk_size]
            output = chunk.pop(neighborhood_size)
            batch_input.append(chunk)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input)
    batch_output = torch.tensor(batch_output)

    return batch_input, batch_output


def initialize_model_weights(model: W2V_CBOW, initialize_mode: str = "equal") -> None:
    if initialize_mode == "equal":
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            else:
                param.data.normal_(std=1.0 / math.sqrt(param.shape[1]))
    else:
        raise ValueError(f"Unsupported initialize mode: {initialize_mode}")


def test_step(model: W2V, test_dl: DataLoader, loss_fn: nn.CrossEntropyLoss, device, h_params: dict):
    model.eval()
    size = len(test_dl)
    num_batches = len(test_dl)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dl:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            test_loss += loss_fn(y_hat, y).item()
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= (size * h_params["batch_size"])
    logger.info(
        f"Test Metrics: \n Accuracy: {correct:>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def gen_epoch_str(epoch_idx: int) -> str:
    ret_str = f"Epoch: {epoch_idx} "
    str_size = len(ret_str)
    term_size = shutil.get_terminal_size().columns
    fill_size = term_size - str_size - 1
    return ret_str + fill_size * "_"


def load_tokenizer(path: Path) -> tokenizers.Tokenizer:
    logger.info(f"Loading tokenizer from {path}")
    with open(path, "r") as f:
        tokenizer_json = f.read()
    tokenizer = Tokenizer.from_str(tokenizer_json)
    logger.info("Tokenizer loaded!")
    return tokenizer


def prepare_dataset(
        path: str | Path, name: str = None, test_size: float = 0.1
) -> DatasetDict:
    logger.info(f"Retrieving dataset: {path}")
    if name:
        ds = load_dataset(path=path, name=name)
    else:
        ds = load_dataset(path)
    logger.info("Producing train/test splits")
    ds = ds["train"].train_test_split(test_size=test_size)
    return ds


def build_dataloaders(
        train_ds: Dataset | list,
        test_ds: Dataset | list,
        debug_ds: Dataset | list,
        batch_size: int,
        collate_fn: Callable,
        n_workers: int = 1,
        debug: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    if debug:
        logger.info("Building train, test, and debug dataloaders")
    else:
        logger.info("Building train and test dataloaders")
    logger.info(f"Batch size: {batch_size}, Number of workers: {n_workers}")
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=n_workers,
    )
    logger.info("Train dataloader constructed")
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=n_workers,
    )
    logger.info("Test dataloader constructed")
    if debug:
        debug_dl = DataLoader(
            debug_ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=n_workers,
        )
        logger.info("Debug dataloader constructed")
    else:
        debug_dl = None

    return train_dl, test_dl, debug_dl


def train(
        model: W2V_CBOW,
        optimizer,
        loss_fn,
        train_dl: DataLoader,
        test_dl: DataLoader,
        h_params: dict,
        training_cfg: dict,
) -> None:
    run_id = training_cfg["run_id"]
    should_resume = training_cfg["resume"]
    if should_resume:
        import os

        mw_path = Path(f"./checkpoints/{run_id}/")
        mw_path = sorted(mw_path.glob("checkpoint_epoch_*.pth"), key=os.path.getctime)[
            -1
        ]
        logger.info(f"Resuming training, loading model at {mw_path}")
        checkpoint = torch.load(mw_path)
        model.load_state_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        logger.info("Initializing model weights")
        initialize_model_weights(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)

    metrics_logging_path = Path(f"./runs/{run_id}").absolute()
    logger.info(f"Logging metrics to {metrics_logging_path}")
    if not metrics_logging_path.exists():
        logger.warning("Metrics logging directory not found, creating...")
        metrics_logging_path.mkdir(parents=True)
    metrics_logger = SummaryWriter(log_dir=str(metrics_logging_path))

    log_inter = math.floor(len(train_dl) * training_cfg["logging_interval"])
    chkpt_inter = math.floor(len(train_dl) * training_cfg["ckpt_interval"])
    n_epochs = training_cfg["n_epochs"]

    model.train()
    prev_loss = 0
    chkpt_idx = 0
    chkpt_dir = Path(f"./checkpoints/{run_id}")
    if not chkpt_dir.exists():
        chkpt_dir.mkdir(parents=True)
    for epoch_idx in range(n_epochs):
        logger.info(f"Epoch: {epoch_idx}")
        print(term_size * "_")
        for batch_idx, (X, y) in enumerate(train_dl):
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % log_inter == 0:
                metrics_logger.add_scalar("training_loss", loss.item())
                loss = loss.item()
                batch_idx_str_len = len(str(batch_idx))
                train_dl_str_len = len(str(len(train_dl)))
                batch_str_delta = train_dl_str_len - batch_idx_str_len
                if batch_idx == 0:
                    logger.info(
                        f"{TermColors.UNDERLINE}Loss{TermColors.ENDC}: {loss:<7f} [{batch_str_delta * '0'}{batch_idx}/{TermColors.BOLD}{len(train_dl)}{TermColors.ENDC}]")
                else:
                    if prev_loss > loss:
                        logger.info(
                            f"{TermColors.UNDERLINE}Loss{TermColors.ENDC}: {TermColors.GOOD}{loss:<7f}{TermColors.ENDC} [{batch_str_delta * '0'}{batch_idx}/{TermColors.BOLD}{len(train_dl)}{TermColors.ENDC}]")
                    else:
                        logger.info(
                            f"{TermColors.UNDERLINE}Loss{TermColors.ENDC}: {TermColors.BAD}{loss:<7f}{TermColors.ENDC} [{batch_str_delta * '0'}{batch_idx}/{TermColors.BOLD}{len(train_dl)}{TermColors.ENDC}]")
                    prev_loss = loss
            if batch_idx % chkpt_inter == 0:
                logger.info(
                    f"{TermColors.BOLD}Saving checkpoint to: {chkpt_dir / f'checkpoint_{chkpt_idx}.pth'}{TermColors.ENDC}")
                torch.save(
                    {
                        "epoch": epoch_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    chkpt_dir / f"checkpoint_{chkpt_idx}.pth",
                )
                chkpt_idx += 1

        logger.info(f"Saving epoch checkpoint to: {f'checkpoint_epoch_{chkpt_idx}.pth'}")
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            chkpt_dir / f"checkpoint_epoch_{chkpt_idx}.pth",
        )

        logger.info("Running validation")
        test_step(model, test_dl, loss_fn, device=device, h_params=h_params)


if __name__ == "__main__":
    ##############################################################################
    term_size = shutil.get_terminal_size().columns
    print("\n" + term_size * "=" + "\n")
    print("__/\\\______________/\\\____/\\\\\\\\\______/\\\________/\\\_")
    print(" _\/\\\_____________\/\\\__/\\\///////\\\___\/\\\_______\/\\\_ ")
    print("  _\/\\\_____________\/\\\_\///______\//\\\__\//\\\______/\\\__ ")
    print(" v _\//\\\____/\\\____/\\\____________/\\\/____\//\\\____/\\\___")
    print("    __\//\\\__/\\\\\__/\\\__________/\\\//_______\//\\\__/\\\____")
    print("     ___\//\\\/\\\/\\\/\\\________/\\\//___________\//\\\/\\\_____")
    print("      ____\//\\\\\\//\\\\\_______/\\\/_______________\//\\\\\______")
    print("       _____\//\\\__\//\\\_______/\\\\\\\\\\\\\\\______\//\\\_______")
    print("        ______\///____\///_______\///////////////________\///________")
    print("\n" + term_size * "=" + "\n")
    ##############################################################################
    console = Console()

    # Load training configuration
    training_cfg_path = Path("./training_config.toml")
    logger.info(f"Loading training configuration from {training_cfg_path}")
    assert training_cfg_path.exists(), f"Cannot find training config! Supplied path: {training_cfg_path}"

    cfg = toml.load(training_cfg_path)
    h_params = cfg["Model"]
    train_cfg = cfg["Train"]

    model_table = Table(title="Model Configurations")
    model_table.add_column("Key", no_wrap=True)
    model_table.add_column("Value")
    model_table.add_row("Batch Size", str(h_params["batch_size"]), style="orange_red1")
    model_table.add_row("Window Size", str(h_params["window_size"]), style="orange_red1")
    model_table.add_row("Embedding Dimension", str(h_params["embedding_dim"]), style="orange_red1")
    console.print(model_table)

    training_table = Table(title="Training Configurations")
    training_table.add_column("Key", no_wrap=True)
    training_table.add_column("Value")
    training_table.add_row("Mode", train_cfg["mode"], style="dodger_blue1")
    training_table.add_row("Run Id", train_cfg["run_id"], style="dodger_blue1")
    training_table.add_row("Dataset Path", train_cfg["dataset_path"], style="dodger_blue1")
    training_table.add_row("Dataset Name", train_cfg["dataset_name"], style="dodger_blue1")
    training_table.add_row("Tokenizer Path", train_cfg["tokenizer_path"], style="dodger_blue1")
    training_table.add_row("Number of Debug Samples", str(train_cfg["n_debug_samples"]), style="orange_red1")
    training_table.add_row("Seed", str(train_cfg["seed"]), style="orange_red1")
    training_table.add_row("Test Size", str(train_cfg["test_size"]), style="orange_red1")
    training_table.add_row("Number of Data Loader Workers", str(train_cfg["n_dl_workers"]), style="orange_red1")
    training_table.add_row("Evaluation Interval", str(train_cfg["val_interval"]), style="orange_red1")
    training_table.add_row("Checkpoint Interval", str(train_cfg["ckpt_interval"]), style="orange_red1")
    training_table.add_row("Logging Interval", str(train_cfg["logging_interval"]), style="orange_red1")
    training_table.add_row("Number of Epochs", str(train_cfg["n_epochs"]), style="orange_red1")
    training_table.add_row("Learning Rate", str(train_cfg["lr"]), style="orange_red1")
    training_table.add_row("Resume Training", str(train_cfg["resume"]), style="purple")
    console.print(training_table)

    mode = train_cfg["mode"]
    window_size = h_params["window_size"]
    chunk_size = window_size * 2 + 1

    # Set random number generator seed
    seed = train_cfg['seed']
    logger.info(f"Setting seeds with value: {seed}")
    set_seeds(seed)

    # Prepare dataset
    # dataset_path = "wikimedia/wikipedia"
    dataset_path = train_cfg["dataset_path"]
    # dataset_name = "20231101.en"
    dataset_name = train_cfg["dataset_name"]
    dataset_test_size = train_cfg["test_size"]
    ds = prepare_dataset(dataset_path, dataset_name, dataset_test_size)

    # Load pre-trained tokenizer
    tokenizer_path = Path(train_cfg["tokenizer_path"])
    tokenizer = load_tokenizer(tokenizer_path)

    # Define PyTorch DataLoaders
    batch_size = h_params["batch_size"]
    n_workers = train_cfg["n_dl_workers"]
    train_ds = ds["train"]
    test_ds = ds["test"]
    logger.info("Preparing collation function")
    collate_fn = partial(
        cbow_collate_fn,
        chunk_size=chunk_size,
        neighborhood_size=window_size,
    )
    n_debug_samples = train_cfg["n_debug_samples"]
    debug_ds = train_ds[:n_debug_samples]
    train_dl, test_dl, debug_dl = build_dataloaders(
        train_ds, test_ds, debug_ds, batch_size, collate_fn, n_workers, False
    )

    # Train
    logger.info("Instantiating model")
    model = W2V_CBOW(
        tokenizer=tokenizer,
        embedding_dim=h_params["embedding_dim"],
        neighborhood_size=window_size,
    )
    lr = train_cfg["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    logger.info("Beginning training loop...")
    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dl=train_dl,
        test_dl=test_dl,
        h_params=h_params,
        training_cfg=train_cfg,
    )

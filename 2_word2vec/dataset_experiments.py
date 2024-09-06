import datasets
from pathlib import Path
import toml
from train import cbow_collate_fn, build_dataloaders, load_tokenizer, prepare_dataset
from functools import partial
import time
import torch

if __name__ == '__main__':
    training_cfg_path = Path("./training_config.toml")
    assert training_cfg_path.exists(), f"Cannot find training config! Supplied path: {training_cfg_path}"
    cfg = toml.load(training_cfg_path)
    h_params = cfg["Model"]
    train_cfg = cfg["Train"]
    misc_cfg = cfg["Misc"]

    ds = prepare_dataset("roneneldan/TinyStories", "default", 0.2)
    tokenizer_path = Path(train_cfg["tokenizer_path"])
    tokenizer = load_tokenizer(tokenizer_path)

    batch_size = h_params["batch_size"]
    n_workers = train_cfg["n_dl_workers"]
    train_ds = ds["train"]
    test_ds = ds["test"]
    window_size = h_params["window_size"]
    chunk_size = window_size * 2 + 1
    collate_fn = partial(
        cbow_collate_fn,
        chunk_size=chunk_size,
        neighborhood_size=window_size,
        tokenizer=tokenizer,
    )
    n_debug_samples = train_cfg["n_debug_samples"]
    debug_ds = train_ds[:n_debug_samples]
    train_dl, test_dl, debug_dl = build_dataloaders(
        train_ds, test_ds, debug_ds, batch_size, collate_fn, n_workers, False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    for (X, y) in test_dl:
        X = X.to(device)
        y = y.to(device)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")

import datasets
from pathlib import Path
import toml
from train import cbow_collate_fn, build_dataloaders, load_tokenizer
from functools import partial
import time

if __name__ == '__main__':
    training_cfg_path = Path("./training_config.toml")
    assert training_cfg_path.exists(), f"Cannot find training config! Supplied path: {training_cfg_path}"
    cfg = toml.load(training_cfg_path)
    h_params = cfg["Model"]
    train_cfg = cfg["Train"]
    misc_cfg = cfg["Misc"]

    ds = datasets.load_dataset("roneneldan/TinyStories")
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
    )
    n_debug_samples = train_cfg["n_debug_samples"]
    debug_ds = train_ds[:n_debug_samples]
    train_dl, test_dl, debug_dl = build_dataloaders(
        train_ds, test_ds, debug_ds, batch_size, collate_fn, n_workers, False
    )

    start_time = time.time()
    # exp
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")

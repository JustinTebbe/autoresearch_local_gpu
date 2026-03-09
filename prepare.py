"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle
import platform
import multiprocessing
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _detect_local_gpu_profile():
    profile = os.environ.get("AUTORESEARCH_PROFILE", "auto").strip().lower()
    if profile in {"local", "local-gpu", "windows"}:
        return True
    if profile in {"baseline", "h100"}:
        return False
    if platform.system() != "Windows" or not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return True
    return props.total_memory <= 24 * 1024**3


LOCAL_GPU_PROFILE = _detect_local_gpu_profile()
DEFAULT_MAX_SEQ_LEN = 512 if LOCAL_GPU_PROFILE else 2048
DEFAULT_EVAL_TOKENS = 4 * 524288 if LOCAL_GPU_PROFILE else 40 * 524288
DEFAULT_VOCAB_SIZE = 4096 if LOCAL_GPU_PROFILE else 8192


def _detect_dataset_name():
    dataset = os.environ.get("AUTORESEARCH_DATASET", "auto").strip().lower()
    if dataset == "auto":
        return "tinystories" if LOCAL_GPU_PROFILE else "climbmix"
    aliases = {
        "climbmix": "climbmix",
        "climbmix-400b": "climbmix",
        "tinystories": "tinystories",
        "tinystories-gpt4-clean": "tinystories",
    }
    if dataset not in aliases:
        raise ValueError(f"Unsupported AUTORESEARCH_DATASET={dataset!r}")
    return aliases[dataset]


MAX_SEQ_LEN = _env_int("AUTORESEARCH_MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN)
TIME_BUDGET = _env_int("AUTORESEARCH_TIME_BUDGET", 300)
EVAL_TOKENS = _env_int("AUTORESEARCH_EVAL_TOKENS", DEFAULT_EVAL_TOKENS)
DATASET_NAME = _detect_dataset_name()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

if platform.system() == "Windows":
    CACHE_ROOT = os.environ.get("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local"))
else:
    CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache")

CACHE_DIR = os.path.join(CACHE_ROOT, "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data", DATASET_NAME)
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer", DATASET_NAME)
VOCAB_SIZE = _env_int("AUTORESEARCH_VOCAB_SIZE", DEFAULT_VOCAB_SIZE)

CLIMBMIX_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
CLIMBMIX_MAX_SHARD = 6542
CLIMBMIX_VAL_SHARD = CLIMBMIX_MAX_SHARD
CLIMBMIX_VAL_FILENAME = f"shard_{CLIMBMIX_VAL_SHARD:05d}.parquet"

TINYSTORIES_BASE_URL = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main"
TINYSTORIES_FILENAME = "tinystories_gpt4_clean.parquet"
TINYSTORIES_VAL_ROWS = 10_000
TINYSTORIES_TEST_ROWS = 10_000

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{CLIMBMIX_BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            temp_path = filepath + ".tmp"
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_file(filename, url):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            temp_path = filepath + ".tmp"
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download the selected dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if DATASET_NAME == "climbmix":
        num_train = min(num_shards, CLIMBMIX_MAX_SHARD)
        ids = list(range(num_train))
        if CLIMBMIX_VAL_SHARD not in ids:
            ids.append(CLIMBMIX_VAL_SHARD)

        existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
        if existing == len(ids):
            print(f"Data ({DATASET_NAME}): all {len(ids)} shards already downloaded at {DATA_DIR}")
            return

        needed = len(ids) - existing
        print(f"Data ({DATASET_NAME}): downloading {needed} shards ({existing} already exist)...")

        workers = max(1, min(download_workers, needed))
        with Pool(processes=workers) as pool:
            results = pool.map(download_single_shard, ids)

        ok = sum(1 for r in results if r)
        print(f"Data ({DATASET_NAME}): {ok}/{len(ids)} shards ready at {DATA_DIR}")
        return

    if DATASET_NAME == "tinystories":
        print(f"Data ({DATASET_NAME}): downloading 1 parquet shard...")
        ok = download_file(TINYSTORIES_FILENAME, f"{TINYSTORIES_BASE_URL}/{TINYSTORIES_FILENAME}")
        if not ok:
            print(f"Data ({DATASET_NAME}): failed to download {TINYSTORIES_FILENAME}")
            sys.exit(1)
        print(f"Data ({DATASET_NAME}): ready at {DATA_DIR}")
        return

    raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def _iter_climbmix_text(split):
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, CLIMBMIX_VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    elif split == "val":
        parquet_paths = [val_path]
    else:
        raise ValueError(f"Unsupported split for climbmix: {split}")

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            for text in rg.column("text").to_pylist():
                yield text


def _iter_tinystories_text(split):
    parquet_path = os.path.join(DATA_DIR, TINYSTORIES_FILENAME)
    assert os.path.exists(parquet_path), "TinyStories parquet not found. Run prepare.py first."
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    train_end = max(total_rows - TINYSTORIES_VAL_ROWS - TINYSTORIES_TEST_ROWS, 0)
    val_end = min(train_end + TINYSTORIES_VAL_ROWS, total_rows)
    if split == "train":
        start_idx, end_idx = 0, train_end
    elif split == "val":
        start_idx, end_idx = train_end, val_end
    else:
        raise ValueError(f"Unsupported split for tinystories: {split}")

    row_offset = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=["text"])
        texts = rg.column("text").to_pylist()
        rg_start = row_offset
        rg_end = row_offset + len(texts)
        if rg_end <= start_idx:
            row_offset = rg_end
            continue
        if rg_start >= end_idx:
            break
        local_start = max(0, start_idx - rg_start)
        local_end = min(len(texts), end_idx - rg_start)
        for text in texts[local_start:local_end]:
            yield text
        row_offset = rg_end


def iter_split_text(split):
    if DATASET_NAME == "climbmix":
        yield from _iter_climbmix_text(split)
        return
    if DATASET_NAME == "tinystories":
        yield from _iter_tinystories_text(split)
        return
    raise ValueError(f"Unsupported dataset: {DATASET_NAME}")


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from the training split."""
    nchars = 0
    for text in iter_split_text("train"):
        doc = text[:doc_cap] if len(text) > doc_cap else text
        nchars += len(doc)
        yield doc
        if nchars >= max_chars:
            return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    if not list_parquet_files():
        print("Tokenizer: no parquet files found. Download data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from the configured dataset."""
    epoch = 1
    while True:
        batch = []
        for text in iter_split_text(split):
            batch.append(text)
            if len(batch) == tokenizer_batch_size:
                yield batch, epoch
                batch = []
        if batch:
            yield batch, epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000, device=None):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    batch_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    inputs = batch_buffer[:B * T].view(B, T)
    targets = batch_buffer[B * T:].view(B, T)
    if device.type == "cuda":
        cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
        cpu_inputs = cpu_buffer[:B * T].view(B, T)
        cpu_targets = cpu_buffer[B * T:].view(B, T)
    else:
        cpu_buffer = None

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        if cpu_buffer is None:
            inputs.copy_(row_buffer[:, :-1])
            targets.copy_(row_buffer[:, 1:])
        else:
            cpu_inputs.copy_(row_buffer[:, :-1])
            cpu_targets.copy_(row_buffer[:, 1:])
            batch_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    device = next(model.parameters()).device
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", device=device)
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of climbmix training shards to download (-1 = all). Ignored for TinyStories.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    args = parser.parse_args()

    num_shards = CLIMBMIX_MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Dataset: {DATASET_NAME}")
    if LOCAL_GPU_PROFILE:
        print("Runtime profile: local-gpu")
        print(f"  MAX_SEQ_LEN={MAX_SEQ_LEN}, EVAL_TOKENS={EVAL_TOKENS}, VOCAB_SIZE={VOCAB_SIZE}")
    else:
        print("Runtime profile: baseline")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")

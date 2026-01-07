import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset

IGNORE_INDEX = -100


class ShiftedCausalLMDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        seq_len: int = 128,
        add_eos: bool = True,
        streaming: bool = False,
        cache_path: str | None = None,  # path to save/load pre-tokenized dataset
        max_chunk_len: int = 10000,     # max string length per chunk
    ):
        """
        dataset_name: "tinystories" or "openwebtext"
        split: "train", "validation", "test"
        seq_len: context length for LM
        add_eos: append EOS token
        streaming: if True, skip caching
        cache_path: path to load/save pre-tokenized dataset
        max_chunk_len: max chars per text chunk to avoid tokenizer overflow
        """

        self.seq_len = seq_len
        self.add_eos = add_eos
        self.cache_path = cache_path
        self.max_chunk_len = max_chunk_len

        # -------------------------
        # 1️⃣ Build tokenizer
        # -------------------------
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # -------------------------
        # 2️⃣ Load cached dataset if exists
        # -------------------------
        if self.cache_path is not None and os.path.exists(self.cache_path):
            print(f"Loading tokenized dataset from {self.cache_path}")
            self.dataset = torch.load(self.cache_path)
            return

        # -------------------------
        # 3️⃣ Load HF dataset
        # -------------------------
        dataset_name = dataset_name.lower()
        if dataset_name == "tinystories":
            hf_name = "roneneldan/TinyStories"
            text_field = "text"
        elif dataset_name == "openwebtext":
            hf_name = "openwebtext"
            text_field = "text"
        else:
            raise ValueError(
                f"Unsupported dataset {dataset_name}, must be 'tinystories' or 'openwebtext'"
            )

        self.dataset = load_dataset(hf_name, split=split, streaming=streaming)

        if not streaming:
            # -------------------------
            # 4️⃣ Pre-tokenize dataset
            # -------------------------
            self.dataset = self.dataset.map(
                lambda batch: self._tokenize(batch, text_field),
                batched=True,
                remove_columns=self.dataset.column_names,
            )
            self.dataset.set_format(type="python")

            # -------------------------
            # 5️⃣ Save to cache if requested
            # -------------------------
            if self.cache_path is not None:
                print(f"Saving tokenized dataset to {self.cache_path}")
                torch.save(self.dataset, self.cache_path)

    # -------------------------
    # Helper: tokenize + chunk long texts
    # -------------------------
    def _tokenize(self, batch, text_field):
        texts = batch[text_field]

        if self.add_eos:
            texts = [t + self.tokenizer.eos_token for t in texts]

        # Split very long texts into smaller chunks to avoid tokenizer overflow
        all_chunks = []
        for t in texts:
            for i in range(0, len(t), self.max_chunk_len):
                all_chunks.append(t[i:i + self.max_chunk_len])

        encodings = self.tokenizer(
            all_chunks,
            truncation=False,
            add_special_tokens=False,
            max_length=None,  # disable internal max_length
        )
        return encodings

    # -------------------------
    # Dataset interface
    # -------------------------
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.dataset[idx]["input_ids"]

        # -------------------------
        # 1️⃣ Slice sequence for input/label
        # -------------------------
        if len(tokens) >= self.seq_len + 1:
            start = torch.randint(0, len(tokens) - self.seq_len, (1,)).item()
            window = tokens[start:start + self.seq_len + 1]
        else:
            pad_len = self.seq_len + 1 - len(tokens)
            window = tokens + [self.pad_token_id] * pad_len

        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        labels = torch.tensor(window[1:], dtype=torch.long)

        # -------------------------
        # 2️⃣ Mask padding labels
        # -------------------------
        labels[labels == self.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    # -------------------------
    # 3️⃣ Manual save function
    # -------------------------
    def save_to_cache(self, path: str):
        """Serialize the pre-tokenized dataset to disk for fast reload."""
        if hasattr(self, "dataset"):
            print(f"Saving dataset to {path}")
            torch.save(self.dataset, path)
        else:
            raise RuntimeError("No dataset loaded to save")


def build_dataset(
    name: str,
    seq_len: int,
    split: str = "train",
    streaming: bool = False,
):
    name = name.lower()

    if name == "tinystories":
        return ShiftedCausalLMDataset(
            dataset_name="TinyStories",
            split=split,
            seq_len=seq_len,
            streaming=streaming,
        )

    elif name == "openwebtext":
        return ShiftedCausalLMDataset(
            dataset_name="openwebtext",
            split=split,
            seq_len=seq_len,
            streaming=streaming,
        )

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            "Supported: ['tinystories', 'openwebtext']"
        )

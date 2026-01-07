import os
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
        cache_dir: str = "data",  # directory for cached datasets
        max_chunk_len: int = 10000,
    ):
        self.seq_len = seq_len
        self.add_eos = add_eos
        self.max_chunk_len = max_chunk_len

        # -------------------------
        # 1️⃣ Build tokenizer
        # -------------------------
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # -------------------------
        # 2️⃣ Construct cache path in data/
        # -------------------------
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = dataset_name.lower()
        self.cache_path = os.path.join(cache_dir, f"{safe_name}_{split}_tokenized.pt")

        # -------------------------
        # 3️⃣ Load cached dataset if exists
        # -------------------------
        if os.path.exists(self.cache_path):
            print(f"Loading tokenized dataset from {self.cache_path}")
            self.dataset_data = torch.load(self.cache_path)  # list of dicts with 'input_ids'
            return

        # -------------------------
        # 4️⃣ Load HF dataset
        # -------------------------
        if safe_name == "tinystories":
            hf_name = "roneneldan/TinyStories"
            text_field = "text"
        elif safe_name == "openwebtext":
            hf_name = "openwebtext"
            text_field = "text"
        else:
            raise ValueError(f"Unsupported dataset {dataset_name}")

        dataset = load_dataset(hf_name, split=split, streaming=streaming)

        if streaming:
            raise ValueError("Streaming mode not supported with caching")

        # -------------------------
        # 5️⃣ Pre-tokenize and flatten dataset
        # -------------------------
        tokenized_list = []
        for item in dataset:
            texts = [item[text_field]]
            if self.add_eos:
                texts = [t + self.tokenizer.eos_token for t in texts]

            for t in texts:
                # Chunk very long texts
                for i in range(0, len(t), self.max_chunk_len):
                    chunk = t[i:i + self.max_chunk_len]
                    encodings = self.tokenizer(
                        chunk,
                        truncation=False,
                        add_special_tokens=False,
                        max_length=None,
                    )
                    tokenized_list.append({"input_ids": encodings["input_ids"]})

        # Save for fast reload
        print(f"Saving tokenized dataset to {self.cache_path}")
        torch.save(tokenized_list, self.cache_path)

        self.dataset_data = tokenized_list

    # -------------------------
    # Dataset interface
    # -------------------------
    def __len__(self):
        return len(self.dataset_data)

    def __getitem__(self, idx):
        tokens = self.dataset_data[idx]["input_ids"]

        # -------------------------
        # Slice sequence for input/label
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
        # Mask padding labels
        # -------------------------
        labels[labels == self.pad_token_id] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels}


# -------------------------
# Factory function
# -------------------------
def build_dataset(name: str, seq_len: int, split: str = "train"):
    name = name.lower()
    if name in ["tinystories", "openwebtext"]:
        return ShiftedCausalLMDataset(
            dataset_name=name,
            split=split,
            seq_len=seq_len,
            cache_dir="data",  # automatically store in data/
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: ['tinystories', 'openwebtext']")

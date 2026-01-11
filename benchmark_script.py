import argparse
from datetime import datetime
from functools import partial
import os
from pathlib import Path
import timeit
import torch
import torch.cuda.nvtx as nvtx
from tqdm import tqdm

from custom_modules import (
    TransformerLM,
    AdamW,
    cross_entropy,
)

from train_script import read_json_to_dict

def generate_batch(vocab_size: int, batch_size: int, seq_len: int, device) -> torch.Tensor:
    data = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    labels = data[:, 1:]
    new_labels = torch.randint(low=0, high=vocab_size, size=(batch_size, 1), device=device)
    labels = torch.cat((labels, new_labels), dim=1)
    return data, labels

def train_step(data: torch.Tensor,
               labels: torch.Tensor,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer | None,
               use_cuda: bool) -> torch.Tensor:
    with nvtx.range("forward pass"):
        outputs = model(data)
    with nvtx.range("computing loss"):
        loss = cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            labels.reshape(-1),
        )
    with nvtx.range("backwards pass"):
        loss.backward()
    if optimizer:
        with nvtx.range("optimizer step"):
            optimizer.step()
    if use_cuda:
        torch.cuda.synchronize()

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device = {device}")

    parser = argparse.ArgumentParser(description="A script that benchmarks the LLM.")
    parser.add_argument("--config", type=str, help="Path to config file", default="")
    args = parser.parse_args()
    hyperparams = read_json_to_dict(args.config)
    batch_size = hyperparams["batch_size"]
    vocab_size = hyperparams["vocab_size"]
    context_length = hyperparams["context_length"]
    d_model = hyperparams["d_model"]
    num_layers = hyperparams["num_layers"]
    num_heads = hyperparams["num_heads"]
    d_ff = hyperparams["d_ff"]
    rope_theta = hyperparams["rope_theta"]

    model = TransformerLM(vocab_size=vocab_size, 
                          context_length=context_length,
                          d_model=d_model,
                          num_layers=num_layers,
                          num_heads=num_heads,
                          d_ff=d_ff,
                          rope_theta=rope_theta,
                          device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    test_sample, test_label = generate_batch(vocab_size=vocab_size,
                                             batch_size=batch_size,
                                             seq_len=context_length,
                                             device=device)
    optimizer = AdamW(params=model.parameters())
    ## Warmup phase
    with nvtx.range("warmup"):
        for warmup_idx in range(hyperparams["num_warmups"]):
            model(test_sample)
        if use_cuda:
            torch.cuda.synchronize()

    ## Measurement phase
    setup_stmt = "from __main__ import train_step"
    main_stmt = partial(train_step,
                        data=test_sample,
                        labels=test_label,
                        model=model,
                        optimizer=optimizer,
                        use_cuda=use_cuda)
    with nvtx.range("benchmark"):
        execution_time = timeit.repeat(stmt=main_stmt,
                                       setup=setup_stmt,
                                       repeat=hyperparams["num_measurements"],
                                       number=1)
    print(f"Execution times: {execution_time}")
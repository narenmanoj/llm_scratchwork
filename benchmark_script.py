import argparse
from datetime import datetime
import os
from pathlib import Path
import timeit
import torch
from tqdm import tqdm

from custom_modules import (
    TransformerLM,
    AdamW,
    cross_entropy,
)

from train_script import read_json_to_dict

def generate_batch(vocab_size: int, batch_size: int, seq_len: int) -> torch.Tensor:
    data = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    return data

def forward_pass(data: torch.Tensor, model: torch.nn.Module, use_cuda: bool) -> torch.Tensor:
    if use_cuda:
        torch.cuda.synchronize()
    model(data)

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device = {device}")

    parser = argparse.ArgumentParser(description="A script that benchmarks the LLM.")
    parser.add_argument("--config", type=str, help="Path to config file", default="")
    args = parser.parse_args()
    hyperparams = read_json_to_dict(args.config)
    batch_size = hyperparams["batch_size"]
    vocab_size = hyperparams["vocab_size"], 
    context_length = hyperparams["context_length"],
    d_model = hyperparams["d_model"],
    num_layers = hyperparams["num_layers"],
    num_heads = hyperparams["num_heads"],
    d_ff = hyperparams["d_ff"],
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

    test_sample = generate_batch(vocab_size=vocab_size,
                                 batch_size=batch_size,
                                 seq_len=context_length)
    
    ## Warmup phase
    for warmup_idx in range(hyperparams["num_warmups"]):
        model(test_sample)

    ## Measurement phase
    time_stmt = "forward_pass(data=test_sample, model=model, use_cuda=use_cuda)"
    execution_time = timeit.repeat(stmt=time_stmt,
                                   setup="pass",
                                   repeat=hyperparams["num_measurements"],
                                   number=1)
    print(f"Execution times: {execution_time}")
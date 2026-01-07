import argparse
from dataset_prepare import build_dataset   # wherever you placed it
from GPTDatasetV1 import GPTDatasetV1
from datetime import datetime
import json
import os
from pathlib import Path
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm




from custom_modules import (
    TransformerLM,
    AdamW,
    cross_entropy,
)

EPOCH_KEY = "epoch"
MODEL_STATE_KEY = "model_state_dict"
OPTIMIZER_STATE_KEY = "optimizer_state_dict"
LOSS_KEY = "loss"

def read_json_to_dict(filename):
    """Reads a JSON file and returns a Python dictionary."""
    try:
        with open(filename, "r") as file_in:
            data_dict = json.load(file_in)
            return data_dict
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        current_directory = os.getcwd()
        print("Current Working Directory:", current_directory)
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{filename}'. Check file format.")
        return None


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

        total_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)
    model.train()
    return total_loss / count



def train_one_epoch(epoch_index, num_epochs, tb_writer, loss_fn, optimizer, model, dataloader, logdir, device, val_dataloader=None, print_every=100):
    running_loss = 0.
    last_loss = 0.

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=True)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in pbar:
        inputs = data["input_ids"].to(device)
        labels = data["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # (B, S, V)

        loss = loss_fn(
            outputs.reshape(-1, outputs.size(-1)),
            labels.reshape(-1),
        )

        loss.backward()
        optimizer.step()


        # Gather data and report
        running_loss += loss.item()
        if i % print_every == print_every - 1:
            last_loss = running_loss / print_every # loss per batch
            tqdm.write(f"  batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.
            # -------------------------
            # Validation step
            # -------------------------
            if val_dataloader is not None:
                val_loss = validate(model, val_dataloader, loss_fn, device)
                tqdm.write(f"  batch {i + 1} validation loss: {val_loss:.4f}")
                tb_writer.add_scalar("Loss/validation", val_loss, tb_x)
    torch.save({
        EPOCH_KEY: epoch_index,
        MODEL_STATE_KEY: model.state_dict(),
        OPTIMIZER_STATE_KEY: optimizer.state_dict(),
        LOSS_KEY: last_loss,
    }, f"{logdir}/{epoch_index}_checkpoint.tar")
    return last_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device = {device}")

    parser = argparse.ArgumentParser(description="A script that trains the LLM.")
    parser.add_argument("--load_checkpoint", type=str, help="The directory and epoch index from which to load. e.g. directory/5", default="")
    parser.add_argument("--config", type=str, help="Path to config file", default="")

    args = parser.parse_args()
    assert len(args.load_checkpoint) > 0 or len(args.config) > 0, "we need at least a checkpoint directory or a config file"
    assert len(args.load_checkpoint) == 0 or len(args.config) == 0, "only one of the checkpoint directory or the config file can be active"

    if len(args.config) > 0:
        hyperparams = read_json_to_dict(args.config)
        dataset_name = hyperparams["dataset_name"].split(".")[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logdir = f"runs/{dataset_name}/{timestamp}"
        config_filename = f"{logdir}/config.json"
        os.makedirs(os.path.dirname(config_filename), exist_ok=True)
        with open(config_filename, "w") as json_file:
            json.dump(hyperparams, json_file, indent=4)
    else:
        logdir = args.load_checkpoint.rsplit("/", 1)[0]
        epoch_index = args.load_checkpoint.rsplit("/", 1)[1]
        hyperparams = read_json_to_dict(Path(f"{logdir}/config.json"))

    tokenizer = tiktoken.get_encoding("gpt2")

    vocab_size = tokenizer.n_vocab

    dataset = build_dataset(
        name=hyperparams["dataset_name"],   # e.g. "tinystories" or "openwebtext"
        seq_len=hyperparams["context_length"],
        split="train",
    )

    # Validation dataset
    val_dataset = build_dataset(
        name=hyperparams["dataset_name"],
        seq_len=hyperparams["context_length"],
        split="validation",  # validation split
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hyperparams.get("batch_size", 8),
        shuffle=False,  # no shuffle for validation
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=hyperparams.get("batch_size", 8),
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    
    model = TransformerLM(vocab_size=vocab_size, 
                        context_length=hyperparams["context_length"],
                        d_model=hyperparams["d_model"],
                        num_layers=hyperparams["num_layers"],
                        num_heads=hyperparams["num_heads"],
                        d_ff=hyperparams["d_ff"],
                        rope_theta=hyperparams["rope_theta"],
                        device=device)
    optimizer = AdamW(model.parameters())
    loss_fn = cross_entropy
    current_epoch = 0
    if len(args.load_checkpoint) > 0:
        checkpoint_file = f"{logdir}/{epoch_index}_checkpoint.tar"
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint[MODEL_STATE_KEY])
        optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_KEY])
        current_epoch = checkpoint[EPOCH_KEY] + 1
        loaded_loss = checkpoint[LOSS_KEY]
        print(f"Resuming from beginning of epoch {current_epoch}")
        print(f"Current loss = {loaded_loss}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    tb_writer = SummaryWriter(logdir)

    for epoch_it in tqdm(range(current_epoch, hyperparams["num_epochs"], 1)):
        train_one_epoch(epoch_index=epoch_it,
                        num_epochs=hyperparams["num_epochs"],
                        tb_writer=tb_writer,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        model=model,
                        dataloader=dataloader,
                        device=device,
                        logdir=logdir)
    test_messages = [
        "test text",
    ]
    max_tokens = 20
    tok_msg = torch.tensor(tokenizer.encode_batch(test_messages, allowed_special={"<|endoftext|>"}), device=device)
    print(f"token ids = {tok_msg}")
    result_toks = model.decode(tok_msg, temperature=1, max_tokens=max_tokens)
    result = tokenizer.decode_batch(result_toks.tolist())

    print(result)
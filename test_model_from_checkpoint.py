import argparse
from custom_modules import TransformerLM
from datetime import datetime
import json
import os
from pathlib import Path
import tiktoken
import torch
from train_script import (
    MODEL_STATE_KEY,
    EPOCH_KEY,
    LOSS_KEY,
    read_json_to_dict,
)

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

    model = TransformerLM(vocab_size=vocab_size, 
                          context_length=hyperparams["context_length"],
                          d_model=hyperparams["d_model"],
                          num_layers=hyperparams["num_layers"],
                          num_heads=hyperparams["num_heads"],
                          d_ff=hyperparams["d_ff"],
                          rope_theta=hyperparams["rope_theta"],
                          use_triton=hyperparams["triton"],
                          device=device)
    current_epoch = 0
    if len(args.load_checkpoint) > 0:
        checkpoint_file = f"{logdir}/{epoch_index}_checkpoint.tar"
        checkpoint = torch.load(checkpoint_file)
        model_state_dict = checkpoint[MODEL_STATE_KEY]
        model_state_dict_keymap = {k: k.replace("_orig_mod.", "") for k in model_state_dict.keys()}
        model_state_dict_clean = {model_state_dict_keymap[k]: v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict_clean)
        current_epoch = checkpoint[EPOCH_KEY] + 1
        loaded_loss = checkpoint[LOSS_KEY]
        print(f"Loading checkpoint from beginning of epoch {current_epoch}")
        print(f"Current loss = {loaded_loss}")
    test_messages = [
        "Once upon a time, there was",
        "A little boy and his mother",
        "The girl and her dog",
        "A witch and a wizard happened upon",
        "The kangaroo was so sorry. He thought it had done something wrong.",
        "Are you kidding me",
        "So why does she hate her student",
        "The supervisor said I can fire you right now and then the supervisor walked over to him and then the supervisor told him I might even do it",
        "My friend asked the moose why he was on the road",
    ]
    max_tokens = 200
    for msg in test_messages:
        tok_msg = torch.tensor(tokenizer.encode_batch([msg], allowed_special={"<|endoftext|>"}), device=device)
        print(f"token ids = {tok_msg}")
        result_toks = model.decode(tok_msg, temperature=1, max_tokens=max_tokens)
        result = tokenizer.decode_batch(result_toks.tolist())
        for res in result:
            print(res)

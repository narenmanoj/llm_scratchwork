from GPTDatasetV1 import GPTDatasetV1
from datetime import datetime
import tiktoken
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from custom_modules import (
    TransformerLM,
    AdamW,
    cross_entropy,
)

def train_one_epoch(epoch_index, tb_writer, loss_fn, optimizer, model, dataloader):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
    hyperparams = {
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 1e4,
        "context_length": 256,
        "num_epochs": 1,
    }

    vocab_size = tokenizer.n_vocab

    dataloader = GPTDatasetV1.create_dataloader(raw_text,
                                                batch_size=8,
                                                shuffle=False,
                                                stride=1,
                                                max_length=hyperparams["context_length"],
                                                device=device)
    
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter('runs/the_verdict_{}'.format(timestamp))

    print(next(model.parameters()).device)

    for epoch_it in range(hyperparams["num_epochs"]):
        train_one_epoch(epoch_index=0,
                        tb_writer=tb_writer,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        model=model,
                        dataloader=dataloader)
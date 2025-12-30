import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        tokenized = tokenizer.encode(txt)
        tsr = torch.tensor
        self.Xy_pairs = [(tsr(tokenized[i: i + max_length]), tsr(tokenized[i + 1: i + max_length + 1])) 
                         for i in range(0, len(tokenized) - max_length, stride)]

    def __len__(self):
        return len(self.Xy_pairs)

    def __getitem__(self, index):
        return self.Xy_pairs[index]
    
    def create_dataloader(txt, 
                          batch_size=4, 
                          max_length=256, 
                          stride=128, 
                          shuffle=True, 
                          drop_last=True, 
                          num_workers=0):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt=txt, tokenizer=tokenizer, max_length=max_length, stride=stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return dataloader
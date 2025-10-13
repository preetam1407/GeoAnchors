from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiViewDataset(Dataset):
    def __init__(self, input_file, tokenizer, transform=None, num_views=6):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.transform = transform
        self.num_views = num_views

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        qa, img_path_dict = self.data[idx]
        img_path = list(img_path_dict.values()) 

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        imgs = [self.transform(read_image(p).float()) for p in img_path]
        imgs = torch.stack(imgs, dim=0)  # [T, C, H, W]

        return q_text, imgs, a_text, img_path
    
    def collate_fn(self, batch):
        q_texts, imgs, a_texts, img_paths = zip(*batch)  # <-- plural names
        imgs = torch.stack(imgs, dim=0)  # [B, T, C, H, W]

        encodings = self.tokenizer(list(q_texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
        labels_tok = self.tokenizer(list(a_texts), padding=True, truncation=True, max_length=128,  return_tensors="pt")

        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        labels = labels_tok.input_ids
        # ignore pad in loss
        labels[labels == self.tokenizer.pad_token_id] = -100  
        labels = labels.to(device)

        return input_ids, attention_mask, imgs.to(device), labels
    
    def test_collate_fn(self, batch):
        q_texts, imgs, a_texts, img_paths = zip(*batch)
        imgs = torch.stack(imgs, dim=0)

        encodings = self.tokenizer(list(q_texts), padding=True, truncation=True, return_tensors="pt")
        labels_tok = self.tokenizer(list(a_texts), padding=True, truncation=True, return_tensors="pt")

        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        labels = labels_tok.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels = labels.to(device)

        return list(q_texts), input_ids, attention_mask, imgs.to(device), labels, img_paths

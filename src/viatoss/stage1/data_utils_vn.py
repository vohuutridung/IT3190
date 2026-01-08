# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

class VietnameseABSADataset(Dataset):
    def __init__(self, tokenizer, data_path, max_len=256):
        """
        Args:
            tokenizer: Tokenizer cá»§a ViT5
            data_path: ÄÆ°á»ng dáº«n tá»›i file train.txt hoáº·c validation.txt
            max_len: Äá»™ dÃ i tá»‘i Ä‘a cá»§a chuá»—i token
        """
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_len = max_len
        
        self.inputs = []
        self.targets = []
        
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_text = self.inputs[index]
        target_text = self.targets[index]

        # Tokenize Input (CÃ¢u gá»‘c)
        source = self.tokenizer(
            source_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize Output (CÃ¢u Ä‘Ã­ch - Label)
        target = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tráº£ vá» dictionary dáº¡ng tensor
        return {
            "source_ids": source.input_ids.squeeze(),
            "source_mask": source.attention_mask.squeeze(),
            "target_ids": target.input_ids.squeeze(),
            "target_mask": target.attention_mask.squeeze()
        }

    def _build_examples(self):
        """
        Äá»c file txt, tÃ¡ch dá»¯ liá»‡u báº±ng dáº¥u '####'
        """
        print(f"Dang doc du lieu tu: {self.data_path}...")
        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('####')
                if len(parts) == 2:
                    src = parts[0].strip()
                    tgt = parts[1].strip()
                    
                    self.inputs.append(src)
                    self.targets.append(tgt)
                    count += 1
        
        print(f"-> Da load xong {count} mau du lieu.")
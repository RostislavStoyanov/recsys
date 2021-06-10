import torch
import torch.utils.data as td

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, text, features, targets, tokenizer, max_len):
        self.features = features
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.encodings = self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=False,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        self.input_ids = self.encodings['input_ids']
        self.attention_masks = self.encodings['attention_mask']
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,item):
        input_ids = self.input_ids[item]
        att_mask = self.attention_masks[item]
        target = self.targets[item]
        
        return {
            'input_ids': input_ids.flatten(),
            'attention_mask': att_mask.flatten(),
            'labels': torch.tensor(target, dtype=torch.float),
            'features': torch.tensor(self.features[item], dtype=torch.float)
        }

import torch
import torch.utils.data as td

class WD_Dataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,item):
        target = self.targets[item]
        
        return {
            'labels': torch.tensor(target, dtype=torch.float),
            'features': torch.tensor(self.features[item], dtype=torch.float)
        }

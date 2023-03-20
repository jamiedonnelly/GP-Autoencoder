import random
import torch 
import os
from torch.utils.data import Dataset, DataLoader 
import src.preprocessing as prep

class ClimateData(Dataset):
    
    def __init__(self,root,transform=None):
        super(ClimateData,self).__init__()
        self.files = sorted([os.path.join(root,f) for f in os.listdir(root)])  
        self.transform = transform
        
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Unpack input/output
        x, y = torch.load(self.files[index])
        
        if self.transform:
            for transform in self.transform.transforms:
                if isinstance(transform,prep.Scale):
                    # Deterministically apply scaling transforms
                    x = transform(x.unsqueeze(0)).squeeze(0)
                    y = transform(y.unsqueeze(0)).squeeze(0)
                else:
                    # Randomly perform non-scaling transforms
                    p = random.random()
                    if p > 0.8:
                        x = transform(x.unsqueeze(0)).squeeze(0)
                        y = transform(y.unsqueeze(0)).squeeze(0)
                        break
        return (x,y)
    
    def __len__(self):
        return len(self.files)
 
if __name__=="__main__":
    
    root = './train'
    dataset = ClimateData(root)
    
    print(f"Dataset length: {len(dataset)}\n")
    print(f"Shape of data at index 10: {dataset[10][0].shape}")
    
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)       
    
    for ix, (input,output) in enumerate(dataloader):
        print(f"{ix} === {input.shape} === {output.shape}")
    

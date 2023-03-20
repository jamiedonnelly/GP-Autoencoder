from typing import Dict
import torch 
import random


# Calculate normalising parameters

def calculate_params(trainloader,shape) -> Dict:
    params = {}
    mean = torch.zeros(shape)
    std  = torch.zeros(shape)

    # Batch-expectation
    for ix, (x,y) in enumerate(trainloader):
        mean += torch.mean(x,dim=0)
        std  += torch.std(x,dim=0)

    # Expectation over batches
    params['mean'] = mean / (ix+1)
    params['std']  = std / (ix+1)

    return params  


# Transforms 

class RandomCropResized(torch.nn.Module):
    def __init__(self):
        super(RandomCropResized,self).__init__()
        pass
        
    def crop(self,x):
        h, w = x.shape[-2:]
        new_h = random.randint(50,int((2*h)/5))
        new_w = int(new_h*2)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        return x[..., top:top + new_h, left:left + new_w]
    
    def resize(self,x,size):
        return torch.nn.functional.interpolate(x, size=size, mode="nearest")
    
    def forward(self,x):
        cropped = self.crop(x)
        return self.resize(cropped,x.shape[-2:])

class FlipChannels(torch.nn.Module):
    def __init__(self):
        super(FlipChannels,self).__init__()
        pass
    
    def forward(self,x):
        return x.flip(dims=(1,))

class GaussianNoise(torch.nn.Module):
    def __init__(self):
        super(GaussianNoise,self).__init__()
        pass

    def forward(self,x):
        return x + torch.randn(x.shape)

class Mask(torch.nn.Module):
    def __init__(self):
        super(Mask,self).__init__()

    def forward(self,x):
        mask = torch.randint_like(x,0,2)
        return x*mask

class Scale(torch.nn.Module):
    def __init__(self, params: dict, epsilon: float =0.01):
        super(Scale,self).__init__()
        self.params = params 
        assert ('mean' in params.keys()) & ('std' in params.keys()), "Param dict doesn't have keys `mean` and `std`"
        self.epsilon = epsilon 

    def forward(self,x):
        return (x-self.params['mean']) / (self.params['std'] + self.epsilon)
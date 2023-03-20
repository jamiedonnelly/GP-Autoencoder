import torch
from torch import nn
import math
from typing import Tuple
from src.models import model_dict
import src.parsing as parsing
import src.build as build

class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential):
        super(Autoencoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._init()

    def _init(self):
        layers = [l for l in self.encoder]
        layers.extend([l for l in self.decoder])
        for layer in layers:
            # weight init
            try:
                weight = layer.weight
                nn.init.kaiming_uniform_(weight,math.sqrt(5,0))
            except:
                pass
            # bias init
            try:
                bias = layer.bias
                nn.init.uniform_(bias)
            except:
                pass

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z)


class vAutoencoder(Autoencoder):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential):
        super(vAutoencoder,self).__init__(encoder,decoder)
    
    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        mu, logvar = z[:,:int(z.shape[1]/2)], z[:,int(z.shape[1]/2):]
        return self.decoder(vAutoencoder.reparameterize(mu,logvar)), mu, logvar

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        L = torch.linalg.cholesky(torch.diag(torch.exp(logvar)))
        epsilon = torch.randn_like(mu)
        return ((L @ epsilon) + mu)
    
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        return torch.vstack([vAutoencoder._reparameterize(mu[i],logvar[i]) for i in range(mu.shape[0])])
        #return vmap(vAutoencoder._reparameterize,randomness='different')(mu, logvar)


# Declare functions
def configure_model(arch,input_shape,latent_dim) -> Autoencoder:
    
    # Initialise base encoder
    base = model_dict[arch]()

    # Skip 'downsample' layers in Resnet architecture
    if 'res' in arch:
        encoder = parsing._parse_torchvision_model(base,['downsample'])
    else:
        encoder = parsing._parse_torchvision_model(base)

    # Modify encoder
        # Modify input layer 
    input_params = parsing._extract_params(encoder[0])
    input_params['in_channels'] = input_shape[0]
    encoder[0] = nn.Conv2d(**input_params) 

        # Modify output layer 
    output_layer_params = parsing._extract_params(encoder[-1])
    output_layer_params['out_features'] = latent_dim
    encoder[-1] = nn.Linear(**output_layer_params)
    
    # Create decoder 
    decoder = build.build_decoder(encoder,input_shape)

    # Check for resnet 
    if 'res' in arch:
        decoder.append(nn.ConvTranspose2d(input_shape[0],input_shape[0],5,padding=2))

    # Create autoencoder
    auto = Autoencoder(encoder,decoder)
    return auto 
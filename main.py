# Misc. Imports
import argparse
import os
from copy import copy

# Backend Imports
import numpy as np 
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision 
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Local Imports
#from plotting import plot_samples
from src.log import get_logger, logger
from src.dataset import ClimateData
from src.models import model_dict
import src.parsing as parsing
import src.autoencoder as autoencoder
import src.build as build
import src.preprocessing as prep


# Parse arguments
parser = argparse.ArgumentParser(description='Autoencoder for mulit-channel hydrodynamic data')
parser.add_argument('--arch','-a',type=str,choices=model_dict.keys(),help='Choice of architecture based on \
    existing torchvision convolutional architectures, e.g., VGG11, VGG19, ResNet50, etc...')
parser.add_argument('--latent_dim',default=4,type=int,help='Number of dimensions for the latent variables produced\
    by the encoding subnetwork. (default: 4)')
parser.add_argument('--epochs',default=10,type=int,help='Number of epochs to train the model for')
parser.add_argument('--batch_size','-b',default=32,type=int,help='mini-batch size (default: 32)')
parser.add_argument('--lr','--learning-rate',default=1e-3,type=float,help='Initial learning rate')
parser.add_argument('--plot-freq', '-p', default=500, type=int,
    metavar='N', help='plot frequency in batches (default: 500) i.e., plot sample of predictions\
    every 500 batches.')
parser.add_argument('--save_freq',default=25,type=int,help='Number of epochs to save model after during training (default = 25), i.e., save model every 25 epochs.')
parser.add_argument('--model_path',default='./',help='Path to save model to duringtraining')

# Declare functions
def configure_model(input_shape) -> autoencoder.Autoencoder:
    
    # Initialise base encoder
    base = model_dict[args.arch]()

    # Skip 'downsample' layers in Resnet architecture
    if 'res' in args.arch:
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
    output_layer_params['out_features'] = args.latent_dim
    encoder[-1] = nn.Linear(**output_layer_params)
    
    # Create decoder 
    decoder = build.build_decoder(encoder,input_shape)

    # Check for resnet 
    if 'res' in args.arch:
        decoder.append(nn.ConvTranspose2d(input_shape[0],input_shape[0],5,padding=2))

    # Create autoencoder
    auto = autoencoder.Autoencoder(encoder,decoder)
    return auto 

def load_data(root: str = './train', batch_size: int=32):
    # Full data
    data = ClimateData(root)
    N = len(data)
    input_shape = data[0][0].shape

    # Partition 
    # Partition into train/test set
    train_data, test_data = random_split(data, [int(0.90*N),N-int(0.90*N)])
    
    Ntr = len(train_data)
    
    # Partition train into train/val set
    train_data, val_data = random_split(train_data, [int(0.8*Ntr),Ntr-int(0.8*Ntr)])
    
    # Save indices
    train_indices, val_indices, test_indices = torch.tensor(train_data.indices),\
    torch.tensor(val_data.indices), torch.tensor(test_data.indices)
    torch.save((train_indices,val_indices,test_indices),f'./data/{args.arch}_z_{args.latent_dim}_indices.pt')

    # Copy underlying data - ensures that different subsets dont reference same underlying object
    train_data.dataset = copy(data)
    val_data.dataset = copy(data)

    # Pre-processing params for training data 
    no_transform_trainloader = DataLoader(train_data, batch_size)

    # Calculate Parameters 
    params = prep.calculate_params(no_transform_trainloader, data[0][0].shape)

    # Define transformations 
    train_tf = transforms.Compose([prep.Scale(params,epsilon=0.01), prep.FlipChannels(),\
                                   prep.RandomCropResized(), prep.Mask(), prep.GaussianNoise()])

    # Define train and validation dataset
    train_data.dataset.transform = train_tf
    # No transformation for validation and test dataset
    
    # Define DistributedSampler for training 
    trainsampler = DistributedSampler(train_data,
                   num_replicas=args.world_size,
                   rank=args.rank)

    # Define loaders 
    train_loader = DataLoader(train_data, batch_size, sampler=trainsampler)
    val_loader   = DataLoader(val_data, batch_size)
    test_loader  = DataLoader(test_data,batch_size)

    return input_shape, params, train_loader, val_loader, test_loader


def train(model, trainloader, valloader, params):
    
    #logger(args,f"{args.rank} init completed")
    # Define loss function and send to GPU 
    criterion = nn.MSELoss().cuda(args.gpu)
    # Define optimiser 
    optim = torch.optim.Adam(model.parameters(),args.lr)

    # Wrap model in DDP
    logger(args,"Wrapping model...")
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    print(f"{args.gpu}")
    model = DDP(model, device_ids=[args.gpu])
    logger(args,"Model successfully wrapped")
    # Training loop
    logger(args,"Beginning Training...")
    for epoch in range(args.epochs):
        model.train()
        logger(args,f"Epoch: {epoch}...")
        for ix, (x,y) in enumerate(trainloader):
            # Zero-out gradients
            optim.zero_grad()

            # Sent inputs to GPU
            x = x.to(args.gpu)
            
            # Forward pass
            outputs = model.forward(x)

            # Backward pass
            loss = criterion(outputs,x)
            loss.backward()
            optim.step()

            # Logging
            if ix % args.plot_freq == 0:
                logger(args,f"\tEpoch: {epoch} == Batch: {ix} == MSE: {loss.item():.4f}")
    
    # Checkpoint save
        if epoch % args.save_freq == 0:
            if args.rank==0:
                validate(model.module.to(args.gpu), valloader, params)
                dist.barrier()
            else:
                dist.barrier()
            torch.save(model.module.state_dict(),
            os.path.join(args.model_path, f"{args.arch}_z_{args.latent_dim}_epoch_{epoch}.pth"))

    logger(args,"Training finished...")

    torch.save(model.module.cpu().state_dict(),
    os.path.join(args.model_path, f"{args.arch}_z_{args.latent_dim}.pth"))

    return model

def validate(model, testloader, params, directory=False):
        test_criterion = nn.MSELoss()
        logger(args,"Testing model...")
        preds, true = [], []
        model.eval()
        with torch.no_grad():
            for ix, (x,y) in enumerate(testloader):
                z_score = (x-params['mean'])/(params['std']+0.01)
                z_score = z_score.to(args.gpu)
                out = model.forward(z_score).cpu()
                out = (out*(params['std']+0.01))+params['mean']
                preds.append(out)
                true.append(x)
        preds, true = torch.vstack([t for t in preds]).float().cpu(), torch.vstack([t for t in true]).float().cpu()
        #logger(args,"Testing RMSE values:...")
        logger(args,f"Temp RMSE: {test_criterion(preds[:,0],true[:,0]).item():.4f}")
        logger(args,f"Pressure RMSE: {test_criterion(preds[:,1],true[:,1]).item():.4f}")
        if directory:
             torch.save((preds,true),os.path.join(directory,f'preds_{args.arch}_z_{args.latent_dim}.pt'))
             logger(args,"Testing finished and predictions saved...")

def main():
    global args 
    args = parser.parse_args()

    # Configure logger
    args.log = get_logger('./logs',f"{args.arch}_{args.latent_dim}.log")

    # Init distributed process 
    args.world_size = int(os.environ['SLURM_NTASKS'])
    args.rank = int(os.environ['SLURM_PROCID'])
    args.gpu = int(os.environ['SLURM_LOCALID'])

    # Init Process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    logger(args,"Process group initialised...")
    # Load data 
    input_shape, params, trainloader, valloader, testloader = load_data('./train')
    logger(args,f"Data loaded...")     

    # Configure base model 
    auto = configure_model(input_shape)
    logger(args,f"Model loaded...")    

    # Training loop
    #   Model is wrapped in DDP within `train`
    trained_model = train(auto, trainloader, valloader, params)

    # Test 
    if args.rank==0:
        validate(trained_model.module.to(args.gpu), testloader, params, './testing')

if __name__=="__main__":
    main()

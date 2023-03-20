import torch 
import numpy as np 
import matplotlib.pyplot as plt
import os

def plot_samples(input_shape,pred,sample,loss,epoch,batch,dir) -> None:
    true_images = [sample[i][0].detach().numpy().reshape(input_shape[-2],input_shape[-1]) for i in range(3)]
    pred_images = [pred[i][0].detach().numpy().reshape(input_shape[-2],input_shape[-1]) for i in range(3)]
    fig, axes = plt.subplots(3,2,figsize=(6,6),dpi=200,tight_layout=True,gridspec_kw={'wspace': -0.20, 'hspace': 0.08})
    for ix, ax in enumerate(axes):
        minv, maxv = np.min(true_images[ix]), np.max(true_images[ix])
        pred_plot = ax[0].imshow(pred_images[ix])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.colorbar(pred_plot,ax=ax[0])
        pred_plot.set_clim(minv,maxv)
        true_plot = ax[1].imshow(true_images[ix])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.colorbar(true_plot,ax=ax[1])
        true_plot.set_clim(minv,maxv)
    plt.suptitle(f"=== Epoch: {epoch} === Batch: {batch} === Loss: {loss:.4f}")
    plt.savefig(os.path.join(dir,f"epoch_{epoch}_batch_{batch}_loss_{loss:.4f}.png"))

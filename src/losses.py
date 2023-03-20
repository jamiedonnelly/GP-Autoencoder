import torch 

class KLPrior(torch.nn.modules.loss._Loss):
    def __init__(self, alpha: float = 0.1):
        super(KLPrior,self).__init__()
        self.alpha = alpha
        pass

    def __call__(self, mu: torch.Tensor, logvar: torch.Tensor):
        e = (torch.exp(logvar)**2)  +  mu - logvar - 1
        return self.alpha * torch.mean(torch.sum(e,dim=1),dim=0)
    

class VAELoss(torch.nn.modules.loss._Loss):
    def __init__(self, alpha: float = 0.1):
        super(VAELoss,self).__init__()
        self.recon_error = torch.nn.MSELoss()
        self.kl_prior_loss = KLPrior(alpha)

    def __call__(self, pred, target, mu, logvar):
        return self.recon_error(pred, target) + self.kl_prior_loss(mu, logvar)
    

""" Sb Model for Testing """

import torch 
from utils.loss_criterions import *
import numpy as np 

#define gen, disc, device 

class SBModel_test(nn.Module):
    """ Initializes the SBModel class, setting up parameters, loss names, model names, visual names, optimizers, and other necessary configurations """
    def __init__(self):
        super(SBModel_test,self).__init__()
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','SB']
        self.model_names = ['G']
        self.visual_names = ['real']
        self.T = 5
        for NFE in range(self.T):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        self.optimizers = []
        self.tau = 0.1 
        self.device = device
        self.lambda_GAN = 1.0   
        self.lambda_NCE = 1.0 
        self.lambda_SB = 1.0
        self.nce_idt = True
        self.nce_layers = [0,4,8,12,16] 
        self.num_patches = 256
        self.netG = gen
        self.netF = netF 
        self.ngf = 64
        self.criterionNCE = criterionNCE(self.nce_layers)
        self.criterionGAN = GANLoss().to(device)
        self.lr = 0.00001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.current_epoch = 0
        self.total_epochs = 180
        
        # optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        
        
    def data_dependent_initialize(self, dataA,dataB): 
        bs = 1
        self.set_input(dataA,dataB)
        self.real_A = self.real_A[:bs]
        self.real_B = self.real_B[:bs]
        self.forward()   
    
    def set_input(self, dataA, dataB):
        """ Responsible for unpacking input data from the dataloader and performing any necessary preprocessing steps """
        self.real_A = dataA.to(device)
        self.real_B = dataB.to(device)
    
    
    def forward(self):
        '''Forward function'''
        tau = 0.01
        T = 5
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = torch.tensor(times).float().cuda()
        self.times = times
        bs =  1
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[bs]).cuda()).long()
        self.time_idx = time_idx
        self.timestep     = times[time_idx]
        with torch.no_grad():
            self.netG.eval()
            for t in range(T):
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom)
                    scale = (delta * (1 - delta / denom))
                Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.ngf]).to(self.real_A.device)
                Xt_1     = self.netG(Xt, time_idx, z)

                self.Xt_1 = Xt_1
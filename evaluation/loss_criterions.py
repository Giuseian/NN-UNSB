import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def criterionNCE(nce_layers):
    criterionNCE = []
    for nce_layer in nce_layers:
        criterionNCE.append(nn.CrossEntropyLoss(reduction='none').to(device))
    return criterionNCE

def criterionGAN():
    return nn.MSELoss().to(device)
import torch
from torch import nn
from torchvision.models import mobilenet_v2


def load_checkpoint (checkpoint_path, model, device):
    """ loading model's weights """
    model.load_state_dict (torch.load (checkpoint_path, map_location = device) ['state_dict'])


def get_model (checkpoint_path, device):    
    
    model = mobilenet_v2 (pretrained = False)
    model.classifier = nn.Sequential(
    nn.Linear (1280, 512),
    nn.ReLU (),
    nn.Dropout (0.5),
    nn.Linear(512, 14)
    )
    model.to(device)
    load_checkpoint (checkpoint_path, model, device)
    
    return model
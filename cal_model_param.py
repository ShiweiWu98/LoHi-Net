import os
from typing import Optional, Dict
from omegaconf import OmegaConf, DictConfig
from utils.loader import *
from ptflops import get_model_complexity_info
import torch

if __name__ == "__main__":
    config_path = r'path/to/config.yaml'
    cfg: DictConfig = OmegaConf.load(config_path)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model = instantiate_from_config(cfg.model.params.net_desmoke)
    
    device = 'cuda:0'
    x = torch.randn((2, 3, 224, 224)).to(device)
    model= model.to(device)
    model.eval()
    _,y= model(x)
    print(y.shape)
    print(y)
    input_shape = (3, 224, 224)
    macs, params = get_model_complexity_info(
        model, input_shape, as_strings=True, print_per_layer_stat=False)
    print("MACs:", macs)
    print("Parameters:", params)

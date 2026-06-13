from typing import Mapping, Any, Tuple, Callable, Dict, Literal
import importlib
from omegaconf import OmegaConf, DictConfig

def get_obj_from_str(string: str, reload: bool = False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def build_callbacks(callbacks_cfg_list):
    """
    Build Lightning callbacks from a list config.

    Example item:
      - class_path: pytorch_lightning.callbacks.ModelCheckpoint
        init_args: ...
    """
    if not callbacks_cfg_list:
        return []

    callbacks = []
    for item in callbacks_cfg_list:
        cls_path = item["class_path"]
        init_args = item.get("init_args", {}) or {}
        callbacks.append(
            instantiate_from_config({"target": cls_path, "params": init_args})
        )
    return callbacks

def build_logger(logger_cfg_list):
    """
    Build Lightning logger(s) from a list config.

    Example item:
      - class_path: pytorch_lightning.loggers.CSVLogger
        init_args:
          save_dir: ...
          name: ...
    """
    if not logger_cfg_list:
        return True  # Lightning: True = enable default logger

    loggers = []
    for item in logger_cfg_list:
        cls_path = item["class_path"]
        init_args = item.get("init_args", {}) or {}

        # Use your factory function to instantiate from "target/params" style config
        loggers.append(
            instantiate_from_config({"target": cls_path, "params": init_args})
        )

    return loggers[0] if len(loggers) == 1 else loggers
def build_model(model_cfg: DictConfig):
    """
    Build the model.
    Example item:
      target: 
        models.archs.backbone
      params: 
        backbone_args
    """
    model = instantiate_from_config(model_cfg)
    return model

def build_datamodule(data_cfg: DictConfig):
    """
    Build the datamodule.
    Example item:
      target: 
        data.dataset.DataModule
      params:
        dataset_args
    """
    datamodule = instantiate_from_config(data_cfg)
    return datamodule

def build_optimizer(model_params, optimizer_cfg):
    cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)
    target = cfg.pop('target')
    params = cfg.pop('params', {})
    
    if 'betas' in params and isinstance(params['betas'], list):
        params['betas'] = tuple(params['betas'])
    
    return get_obj_from_str(target)(model_params, **params)

def build_scheduler(optimizer, scheduler_cfg):
    """
    Build a torch.optim.lr_scheduler from a target/params config, attaching
    the given optimizer at construction time.
    Example config:
        target: torch.optim.lr_scheduler.CosineAnnealingLR
        params:
          T_max: 100
          eta_min: 1.0e-6
        interval: step      # Lightning metadata, ignored by this function
        frequency: 1        # Lightning metadata, ignored by this function
    """
    if scheduler_cfg is None:
        return None

    # 兼容 DictConfig 和原生 dict 两种输入
    if OmegaConf.is_config(scheduler_cfg):
        cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)
    else:
        cfg = dict(scheduler_cfg)

    if not cfg:
        return None

    if "target" not in cfg:
        raise KeyError("Expected key `target` in scheduler config.")

    target = cfg["target"]
    params = dict(cfg.get("params") or {})

    params["optimizer"] = optimizer

    return get_obj_from_str(target)(**params)
import os
import sys
import builtins
import logging
import threading
from functools import wraps
from typing import Any, Dict, Optional, List
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.loader import *

log = logging.getLogger(__name__)
_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(log_path: str) -> None:
    _log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    _lock = threading.Lock()
    _orig = builtins.print

    @wraps(_orig)
    def _tee_print(*args, **kwargs):
        with _lock:
            _orig(*args, **kwargs)
            if kwargs.get("file") is None:
                _orig(*args, **{**kwargs, "file": _log_file, "flush": True})

    builtins.print = _tee_print

    # fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(logging.Formatter(_LOG_FMT))
    # logging.getLogger(__name__).addHandler(fh)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

    if not checkpoints:
        return None

    checkpoints = sorted(
        checkpoints,
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )

    latest_ckpt = os.path.join(checkpoint_dir, checkpoints[0])
    log.info(f"Found latest checkpoint: {latest_ckpt}")
    return latest_ckpt


def find_checkpoint_callback(callbacks: List) -> Optional[ModelCheckpoint]:
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    return None


def main(config_path: str = "config.yaml", resume_from: Optional[str] = None, overrides: Optional[Dict] = None):
    cfg: DictConfig = OmegaConf.load(config_path)

    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value, merge=False)

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_dir = cfg.checkpoint_base_dir
    os.makedirs(cfg_dir, exist_ok=True)

    setup_logging(os.path.join(cfg_dir, "train.log"))

    if overrides:
        log.info("Applying configuration overrides:")
        for key, value in overrides.items():
            log.info(f"  {key} = {value}")

    saved_cfg_path = os.path.join(cfg_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    seed = cfg.get("seed_everything", 42)
    pl.seed_everything(seed, workers=True)

    # print() inside build_model() is now captured by _tee_print
    model = build_model(cfg.model)
    datamodule = build_datamodule(cfg.data)

    trainer_cfg = cfg.get("trainer", {})
    logger = build_logger(trainer_cfg.get("logger", None))
    callbacks = build_callbacks(trainer_cfg.get("callbacks", None))

    ckpt_path = None
    if resume_from == "auto":
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_dir = callback.dirpath
                if checkpoint_dir:
                    ckpt_path = find_latest_checkpoint(checkpoint_dir)
                    if ckpt_path:
                        break

        if not ckpt_path:
            log.info("No checkpoint found for auto-resume. Starting from scratch.")
    elif resume_from is not None:
        if os.path.exists(resume_from):
            ckpt_path = resume_from
            log.info(f"Resuming from checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
    try:
        trainer = pl.Trainer(
            max_epochs=trainer_cfg.max_epochs,
            check_val_every_n_epoch=trainer_cfg.check_val_every_n_epoch,
            accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
            logger=logger,
            callbacks=callbacks,
            devices=trainer_cfg.devices,
        )
    except:
        trainer = pl.Trainer(
            max_epochs=trainer_cfg.max_epochs,
            check_val_every_n_epoch=trainer_cfg.check_val_every_n_epoch,
            accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
            logger=logger,
            callbacks=callbacks,
            gpus=trainer_cfg.devices,
        )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if trainer_cfg.get('test_after_training', True):
        cb = find_checkpoint_callback(callbacks)
        ckpt = (cb.best_model_path or None) if cb else None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)

    parser = argparse.ArgumentParser(description="Train Model with Configurable Options")
    parser.add_argument("--config", type=str, default="configs/coarse_to_fine.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])

    args = parser.parse_args()

    overrides = {}

    if args.lr is not None:
        overrides["model.learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["data.batch_size"] = args.batch_size
    if args.max_epochs is not None:
        overrides["trainer.max_epochs"] = args.max_epochs
    if args.devices is not None:
        overrides["trainer.devices"] = args.devices
        log.info(f"Setting devices to: {args.devices}")
    if args.seed is not None:
        overrides["seed_everything"] = args.seed

    for override_str in args.override:
        if "=" not in override_str:
            log.warning(f"Invalid override format '{override_str}', skipping.")
            continue

        key, value = override_str.split("=", 1)
        key = key.strip()
        value = value.strip()

        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        overrides[key] = value

    main(
        config_path=args.config,
        resume_from=args.resume,
        overrides=overrides if overrides else None
    )
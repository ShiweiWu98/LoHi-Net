import os
from typing import Optional, Dict
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from utils.loader import *
import time

def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Time duration in seconds
    
    Returns:
        Formatted time string (e.g., "1h 23m 45.67s" or "123.45s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"

def predict(config_path: str, checkpoint_path: str, overrides: Optional[Dict] = None):
    """
    Load model from checkpoint and run prediction.
    
    Args:
        config_path: Path to the config YAML file
        checkpoint_path: Path to checkpoint file for prediction
        overrides: Dictionary of config overrides
    
    Returns:
        predictions: List of prediction results from all batches
    """
    # Load config from YAML
    cfg: DictConfig = OmegaConf.load(config_path)
    
    # Apply overrides if provided
    if overrides:
        print("Applying configuration overrides:")
        for key, value in overrides.items():
            print(f"  {key} = {value}")
            OmegaConf.update(cfg, key, value, merge=False)
    
    # Resolve interpolations
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Using config from: {config_path}")
    
    # Set seed for reproducibility
    seed = cfg.get("seed_everything", 42)
    pl.seed_everything(seed, workers=True)
    
    # Build model and datamodule
    print("Building model...")
    model = build_model(cfg.model)
    
    print("Building datamodule...")
    datamodule = build_datamodule(cfg.data)
    
    # Build trainer for prediction
    trainer_cfg = cfg.get("trainer", {})
    print("Creating trainer...")
    trainer = pl.Trainer(
        logger=False,  # Disable logging for prediction
        devices=trainer_cfg.get("devices", 1),
        accelerator="auto",
    )
    
    # Run prediction
    print("Starting prediction...")
    trainer.test(
        model=model, 
        datamodule=datamodule, 
        ckpt_path=checkpoint_path
    )
    predictions = trainer.predict(
        model=model, 
        datamodule=datamodule, 
        ckpt_path=checkpoint_path
    )
    
    print(f"Prediction completed! Generated {len(predictions)} batches of predictions.")
    
    return predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Model Prediction")
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--predict_save_dir",
        type=str,
        required=True,
        help="Directory to save predictions"
    )    
    # Optional arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for prediction (overrides config)"
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=None,
        help="GPU devices to use. Examples: --devices 0 or --devices 0 1 2"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    
    # Generic configuration override
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override any config parameter. Format: key=value. "
             "Example: --override data.num_workers=8"
    )
    
    args = parser.parse_args()
    
    # Build overrides dictionary
    overrides = {}
    if args.predict_save_dir is not None:
        overrides["model.params.predict_save_dir"] = args.predict_save_dir

    if args.batch_size is not None:
        overrides["data.params.batch_size"] = args.batch_size
    
    if args.devices is not None:
        overrides["trainer.devices"] = args.devices
        print(f"Setting devices to: {args.devices}")
    
    if args.seed is not None:
        overrides["seed_everything"] = args.seed
    
    # Parse generic overrides
    for override_str in args.override:
        if "=" not in override_str:
            print(f"Warning: Invalid override format '{override_str}', skipping.")
            continue
        
        key, value = override_str.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Try to convert to appropriate type
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        
        overrides[key] = value
    
    # Run prediction
    main_start_time = time.time()
    predictions = predict(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        overrides=overrides if overrides else None
    )
    main_elapsed = time.time() - main_start_time
    print(f"\n[Timer] Total script execution time: {format_time(main_elapsed)}")

from .unpaired_base import *
from utils.scheduler import *
from utils.loader import *
from typing import Dict, List
from .losses.loss import FFT_Loss, MultiL1Loss, SSIM_Loss
from torch import optim
from torch.nn import functional as F
from .waveelt_utils import _as_wavelet, get_filter_tensors, DWT, IDWT
from torch import nn

class Desmoke_System(Base_Unpaired_Trainer):
    """
    Training system for LoHi-Net with unpaired data
    """
    def __init__(self,
                 net_desmoke: Dict,
                 loss_name: List[str],
                 loss_weights: List[float],
                 predict_save_dir: str,
                 optimizer_cfg: Dict,
                 scheduler_cfg: Dict = None,
                 pwcnet_ckpt: str = None,
                 ):
        super().__init__(predict_save_dir, pwcnet_ckpt)
        self.save_hyperparameters()
        
        # Build network
        self.net_desmoke = instantiate_from_config(net_desmoke)

        print(self.net_desmoke)
        
        # Loss functions
        self.criterions_fft = FFT_Loss()
        self.criterions_l1 = MultiL1Loss()
        # self.criterions_ssim = SSIM_Loss()

        # Loss weights
        assert len(loss_name) == len(loss_weights)
        self.weight_dict = {key: value for key, value in zip(loss_name, loss_weights)}
        
        # Wavelet transform setup
        wt_type = 'db3'
        self.wavelet = _as_wavelet(wt_type)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(wt_type, flip=True)

        self.dec_lo = nn.Parameter(dec_lo, requires_grad=False)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=False)
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=False)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=False)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wt_type, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wt_type, level=1)
        
    def forward(self, smoky):
        """
        Forward pass
        Returns: [out_3, out_4, out_5]
            - out_3: 1/4 resolution output (wavelet domain)
            - out_4: 1/2 resolution output (wavelet domain)
            - out_5: full resolution output
        """
        out_3, out_4, out_5 = self.net_desmoke(smoky)
        return [out_3, out_4, out_5]
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optim_cfg = self.hparams.optimizer_cfg.copy()

        optimizer = optim.AdamW(
            self.net_desmoke.parameters(),
            lr=optim_cfg.lr,
        )

        scheduler_cfg = self.hparams.scheduler_cfg.copy()
        scheduler_type = getattr(scheduler_cfg, "lr_scheduler", "cosine")  
        
        if scheduler_type == "cosine" or scheduler_type == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=scheduler_cfg.lr_min,
                last_epoch=-1
            )
            print('total iters:', self.trainer.estimated_stepping_batches)

        elif scheduler_type == "constant" or scheduler_type == "ConstantLR":
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: 1.0
            )

        elif scheduler_type == "MultiStepRestartLR":
            scheduler = MultiStepRestartLR(
                optimizer,
                milestones=scheduler_cfg.milestones,
                gamma=scheduler_cfg.gamma
            )

        elif scheduler_type == "CosineAnnealingWarmupRestarts":
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=self.trainer.estimated_stepping_batches,
                max_lr=scheduler_cfg.lr,
                min_lr=scheduler_cfg.lr_min,
                warmup_steps=scheduler_cfg.warmup_epochs * self.trainer.num_training_batches,
                gamma=1.0
            )
        else: 
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        """
        Training step
        """

        smoky, real_clear = batch['lq_files'], batch['gt_files']
        pred_clear = self(smoky)  # [out_3, out_4, out_5]
        

        ya_4, yh_4, yv_4, yd_4 = torch.chunk(pred_clear[1], 4, dim=1)
        pred_clear_4 = self.waverec([ya_4, (yh_4, yv_4, yd_4)], None)  # 1/2 -> full resolution
        
        # Align predictions with ground truth at the same scale
        aligned_pred_4, gt_4, _ = self._aligned(pred_clear_4, real_clear)
        aligned_pred_5, gt_5, _ = self._aligned(pred_clear[-1], real_clear)

        aligned_pred = [aligned_pred_4, aligned_pred_5]
        gt = [gt_4, gt_5]
        
        # Compute loss
        loss = 0.
        loss_dict = dict()
        
        # FFT Loss
        if 'fft_loss' in self.weight_dict:
            fft_loss = self.criterions_fft(aligned_pred, gt)
            loss_dict.update({'loss/fft_loss': fft_loss})
            loss = loss + self.weight_dict['fft_loss'] * fft_loss
        
        # L1 Loss
        if 'l1_loss' in self.weight_dict or 'l1' in self.weight_dict:
            l1_loss = self.criterions_l1(aligned_pred, gt)
            loss_dict.update({'loss/l1_loss': l1_loss})
            loss = loss + self.weight_dict['l1_loss'] * l1_loss
        
        # SSIM Loss (optional)
        # if 'ssim_loss' in self.weight_dict:
        #     ssim_loss = self.criterions_ssim(aligned_pred, gt)
        #     loss_dict.update({'loss/ssim_loss': ssim_loss})
        #     loss = loss + self.weight_dict['ssim_loss'] * ssim_loss    
        
        loss_dict.update({'loss/total_loss': loss})
        
        # Get learning rate
        try:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        except (AttributeError, IndexError):
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            
        log_dict = {'learning_rate': lr}
        log_dict.update(loss_dict)

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True
        )
        
        return loss
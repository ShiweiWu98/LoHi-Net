import pytorch_lightning as PL
import torch
from torch import Tensor
import os.path as osp
from PIL import Image
import os
import torchmetrics
from .metrics import Reference_Metric,Reference_Std
from .pwcnet.pwcnet import PWCNet

class Base_Unpaired_Trainer(PL.LightningModule):
    def __init__(self,predict_save_dir: str=None,pwcnet_ckpt: str=None):
        super().__init__()
        
        self.save_hyperparameters({'predict_save_dir': predict_save_dir})
        
        self.pwcnet = PWCNet(load_pretrained=True,weights_path=pwcnet_ckpt)
        
        for param in self.pwcnet.parameters():
            param.requires_grad = False
            
        self.valid_metrics = torchmetrics.MetricCollection(
            {
                "ssim" : Reference_Metric('ssim',data_range=(0.0,1.0)),
                "psnr": Reference_Metric('psnr',data_range=(0.0,1.0)),
                'ciede': Reference_Metric('ciede',data_range=(0.0,1.0)),
            },
            prefix = 'valid/'
        )
        
        self.test_metrics = torchmetrics.MetricCollection(
            {
                "ssim" : Reference_Metric('ssim',data_range=(0.0,1.0)),
                "psnr": Reference_Metric('psnr',data_range=(0.0,1.0)),
                'ciede': Reference_Metric('ciede',data_range=(0.0,1.0)),
                "ssim_std" : Reference_Std('ssim',data_range=(0.0,1.0)),
                "psnr_std": Reference_Std('psnr',data_range=(0.0,1.0)),
                'ciede_std': Reference_Std('ciede',data_range=(0.0,1.0)),
            },
            prefix = 'test/'
        )

    def inference(self, x):
        pred = self.forward(x)
        return pred[-1] if isinstance(pred, list) else pred
    
    def _aligned(self,pred, gt):
        pred, mask, _ = self.pwcnet(pred.contiguous(), gt.contiguous())
        pred = pred * mask
        gt = gt * mask
        return pred, gt, mask
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        smoky, real_clear = batch['lq_files'], batch['gt_files'] 
        pred_clear = self.inference(smoky)
        aligned_pred, gt, _ = self._aligned(pred_clear, real_clear)
        self.valid_metrics.update(aligned_pred, gt)
        
    def on_validation_epoch_end(self,):
        self.log_dict(self.valid_metrics.compute(),sync_dist=False, prog_bar=True, on_epoch=True)
        self.valid_metrics.reset()    
        
    def test_step(self, batch, batch_idx):
        smoky, real_clear = batch['lq_files'], batch['gt_files'] 
        pred_clear = self.inference(smoky)
        aligned_pred, gt, _ = self._aligned(pred_clear, real_clear)
        self.test_metrics.update(aligned_pred, gt)
        
    def on_test_epoch_end(self,):
        self.log_dict(self.test_metrics.compute(),sync_dist=False, on_epoch=True)
        self.test_metrics.reset()      
              
    def predict_step(self, batch, batch_idx):
        smoky, smoky_path = batch['lq_files'], batch['lq_paths'] 
        pred_clear = self.inference(smoky)
        for idx in range(smoky.shape[0]):
            img_name = osp.basename(smoky_path[idx])
            seq_name = osp.basename(osp.dirname(smoky_path[idx]))
            save_name = osp.join(seq_name, img_name)
            self._save_image(pred_clear[idx],save_name)
    
    def _save_image(self, img_tensor, name):
        narr = img_tensor.detach().mul_(255).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
        pil_img = Image.fromarray(narr)
        path = self.hparams.predict_save_dir
        save_path = osp.join(path,name)
        os.makedirs(osp.dirname(save_path),exist_ok=True)
        pil_img.save(save_path)


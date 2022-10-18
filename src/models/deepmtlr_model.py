from typing import Any, List
import pandas as pd

import torch
from torch import nn
from torchmtlr import MTLR


#Number of clin_var
n_clin_var = 12

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CyclicLR

import torch.nn as nn

from torchmtlr import mtlr_neg_log_likelihood, mtlr_survival, mtlr_risk
import numpy as np
from scipy.spatial import cKDTree

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold

from src.models.components.net import UNETR, ViT


class DEEP_MTLR(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
       
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()


        if self.hparams['model'] == 'UNETR':
            self.model = UNETR( hparams = self.hparams,
                                in_channels= 2,
                                out_channels= 1,
                                img_size = (80,80,48),
                                feature_size = 16,
                                hidden_size = 768,
                                mlp_dim = 3072,
                                num_heads = 12,
                                pos_embed = "conv",
                                norm_name = "instance",
                                conv_block = True,
                                res_block = True,
                                dropout_rate = 0.0,
                                spatial_dims = 3,
                            )
        elif self.hparams['model'] == 'ViT':
            self.model = ViT( hparams = self.hparams,
                                in_channels=2,
                                img_size=(80,80,48),
                                patch_size=3,
                                hidden_size=768,
                                mlp_dim=3072,
                                num_heads=12,
                                pos_embed="conv",
                                classification=False,
                                dropout_rate=0.0,
                                spatial_dims=3,
                            )
        
        else:
            print('Please select the correct model architecture name.')


        
            
        self.loss_mask = Dice_and_FocalLoss()


    def forward(self, x: torch.Tensor):
        return self.model(x)

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

            nn.init.constant_(m.bias, -1.5214691)



    def step(self, batch: Any):
        (sample, clin_var), y, labels = batch
        pred_mask, logits = self.forward((sample,clin_var))
        
        loss_mask = self.loss_mask(pred_mask, sample['target_mask'])
        loss_mtlr = mtlr_neg_log_likelihood(logits, y.float(), self.model, self.hparams['C1'], average=True)
      
        
        loss = (1-self.hparams['loss_gamma'])*loss_mtlr + self.hparams['loss_gamma']*loss_mask
        

        return loss, logits, y, labels , pred_mask, sample['target_mask']

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, labels, pred_mask, target_mask = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, y, labels, pred_mask, target_mask = self.step(batch)



        return {"loss": loss, "preds": preds, "y": y, "labels": labels, "pred_mask": pred_mask, "target_mask": target_mask}

    def validation_epoch_end(self, outputs: List[Any]):
        loss        = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob   = torch.cat([x["preds"] for x in outputs]).cpu() 
        y           = torch.cat([x["y"] for x in outputs]).cpu()
        pred_mask   = torch.cat([x["pred_mask"] for x in outputs])
        target_mask   = torch.cat([x["target_mask"] for x in outputs])
        true_time   = torch.cat([x["labels"]["time"] for x in outputs]).cpu()
        true_event  = torch.cat([x["labels"]["event"] for x in outputs]).cpu()
        pred_risk = mtlr_risk(pred_prob).detach().numpy()  
        ci_event  = concordance_index(true_time, -pred_risk, event_observed=true_event)

        log = {"val/loss": loss,
               "val/ci": ci_event,
               "val/dice": dice(pred_mask, target_mask),
               }  
        self.log_dict(log)

        logits  = torch.cat([x["preds"] for x in outputs]).cpu()
        pred_risk = mtlr_risk(logits).detach().numpy()


        PatientID  = [x['labels']['PatientID'] for x in outputs] 
        PatientID = sum(PatientID, []) #inefficient way to flatten a list


        results = pd.DataFrame({'PatientID':PatientID, 'pred_risk':pred_risk})
        results.to_csv('Predictions.csv')
        


        return {"loss": loss, "CI": ci_event}



    def test_step(self, batch: Any, batch_idx: int):
        
        loss, preds, y, labels, pred_mask, target_mask = self.step(batch)
        return {"loss": loss, "preds": preds, "y": y, "labels": labels, "pred_mask": pred_mask, "target_mask": target_mask}



    def test_epoch_end(self, outputs: List[Any]):
        loss        = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob   = torch.cat([x["preds"] for x in outputs]).cpu() 
        y           = torch.cat([x["y"] for x in outputs]).cpu()
        pred_mask   = torch.cat([x["pred_mask"] for x in outputs])
        target_mask   = torch.cat([x["target_mask"] for x in outputs])

        true_time   = torch.cat([x["labels"]["time"] for x in outputs]).cpu()
        true_event  = torch.cat([x["labels"]["event"] for x in outputs]).cpu()


        pred_risk = mtlr_risk(pred_prob).detach().numpy()  

        ci_event  = concordance_index(true_time, -pred_risk, event_observed=true_event)

        log = {"val/loss": loss,
               "val/ci": ci_event,
               "val/dice": dice(pred_mask, target_mask),
            
               }
        
        self.log_dict(log)

        
        logits  = torch.cat([x["preds"] for x in outputs]).cpu()
        pred_risk = mtlr_risk(logits).detach().numpy()


        PatientID  = [x['labels']['PatientID'] for x in outputs] 
        PatientID = sum(PatientID, []) #inefficient way to flatten a list


        results = pd.DataFrame({'PatientID':PatientID, 'pred_risk':pred_risk})
        results.to_csv('Predictions.csv')
        


        return {"loss": loss, "CI": ci_event}
   

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = make_optimizer(AdamW, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        

        scheduler = {
            "scheduler": MultiStepLR(optimizer, milestones=[self.hparams.step], gamma=0.1),
          
            "monitor": "val/loss",
        }
        return [optimizer] , [scheduler]
    
    

def make_optimizer(opt_cls, model, **kwargs):
    """Creates a PyTorch optimizer for MTLR training."""
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},
    ], **kwargs)
    return optimizer



def make_data():
    """Load and preprocess the data."""
    try:
        X = pd.read_csv('/home/numansaeed/Desktop/TMSS/data/edited/hecktor2021_patient_info_training.csv')
        y = pd.read_csv('/home/numansaeed/Desktop/TMSS/data/edited/hecktor2021_patient_endpoint_training.csv')
        df = pd.merge(X, y, on="PatientID")
    except:
        df = path.index

    clinical_data = df
    # if split == "training":
    clinical_data = clinical_data.rename(columns={"Progression": "event", "Progression free survival": "time", "TNM group":"Stage_group", "Gender (1=M,0=F)":"Gender"})
    # Convert time to months

    clinical_data["Age"] = scale(clinical_data["Age"])

    # binarize T stage as T1/2 = 0, T3/4 = 1
    clinical_data["T-stage"] = clinical_data["T-stage"].map(
        lambda x: "T1/2" if x in ["T1", "T2"] else("Tx" if x == "Tx" else "T3/4"), na_action="ignore")

    # use more fine-grained grouping for N stage
    clinical_data["N-stage"] = clinical_data["N-stage"].str.slice(0, 2)

    clinical_data["Stage_group"] = clinical_data["Stage_group"].map(
        lambda x: "I/II" if x in ["I", "II"] else "III/IV", na_action="ignore")

    clinical_data = pd.get_dummies(clinical_data,
                                   columns=["Gender",
                                            "N-stage",
                                            "M-stage",],
                                   drop_first=True)

    cols_to_drop = [
        "PatientID",
        "Tobacco",
        "Alcohol",
        "Performance status",
        "HPV status (0=-, 1=+)",
        "Estimated weight (kg) for SUV",

    ]


    clinical_data = clinical_data.drop(cols_to_drop, axis=1)


    clinical_data = pd.get_dummies(clinical_data,
                                   columns=["T-stage",
                                            "Stage_group",])

    return clinical_data



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss
    

def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()


def hausdorff_distance(image0, image1):
    """Code copied from 
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """
    a_points = np.transpose(np.nonzero(image0.cpu()))
    b_points = np.transpose(np.nonzero(image1.cpu()))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))
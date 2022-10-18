from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.datamodules.transforms import *

from src.datamodules.components.hecktor_dataset import HecktorDataset
import pandas as pd

class HECKTORDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = self.hparams["data_dir"]
      
        self.batch_size = self.hparams["batch_size"]
        self.num_workers = self.hparams["num_workers"]
        self.pin_memory = self.hparams["pin_memory"]
        self.Fold = self.hparams["Fold"]

        self.test_transforms = transforms.Compose(
            [ 
             NormalizeIntensity(),
              ToTensor()]
        )

        self.train_transforms = transforms.Compose([
            NormalizeIntensity(),
            ToTensor(),
        ])



        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def setup(self, stage: Optional[str] = None):

        self.dataset = HecktorDataset(self.hparams["root_dir"],
                                      self.hparams["data_dir"],
                                      self.hparams["patch_sz"],
                                      self.hparams["time_bins_data"],
                                      transform=self.train_transforms,
                                      cache_dir=self.hparams["cache_dir"],
                                      num_workers=self.hparams["num_workers"])
        


        X = pd.read_csv('/home/numansaeed/Desktop/TMSS/data/edited/hecktor2021_patient_info_training.csv')
        y = pd.read_csv('/home/numansaeed/Desktop/TMSS/data/edited/hecktor2021_patient_endpoint_training.csv')
        df = pd.merge(X, y, on="PatientID")


        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5820222)

        train_idx = {}
        test_idx = {}

        key = 1
        for i,j in kf.split(df,df['Progression']):
            train_idx[key] = i
            test_idx[key] = j

            key += 1

        
        print(test_idx[self.Fold])
        train_dataset, val_dataset = Subset(self.dataset, train_idx[self.Fold]), Subset(self.dataset, test_idx[self.Fold])
        val_dataset.dataset.transform = self.test_transforms
        

        self.data_train = train_dataset
        self.data_val = val_dataset
       

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )
    

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )
    ### since test set ground truth is unavailable, testing is done on validation set
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )

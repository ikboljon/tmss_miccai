import os
from traceback import print_tb
from typing import Callable, Optional, Tuple
import sys

import SimpleITK as sitk
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import nibabel as nib
import pathlib
from einops import rearrange

from joblib import Parallel, delayed

from sklearn.preprocessing import scale

from torchmtlr.utils import make_time_bins, encode_survival


def find_centroid(mask: sitk.Image):

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)

    return np.asarray(centroid_idx, dtype=np.float64)

def get_paths_to_patient_files(path_to_imgs, PatientID, append_mask=True):
        path_to_imgs = pathlib.Path(path_to_imgs)

        patients = [p for p in PatientID if os.path.isdir(path_to_imgs / p)]
        paths = []
        for p in patients:
            path_to_ct = path_to_imgs / p / (p + '_ct.nii.gz')
            path_to_pt = path_to_imgs / p / (p + '_pt.nii.gz')

            if append_mask:
                path_to_mask = path_to_imgs / p / (p + '_ct_gtvt.nii.gz')
                paths.append((path_to_ct, path_to_pt, path_to_mask))
            else:
                paths.append((path_to_ct, path_to_pt))
        return paths

class HecktorDataset(Dataset):

    def __init__(self,
                 root_directory:str, 
                 clinical_data_path:str, 
                 patch_size:int =50,
                 time_bins:int = 14,
                 cache_dir:str = "data_cropped/data_cache/",
                 transform: Optional[Callable] = None,
                 num_workers: int = 1
    ):
        print(cache_dir)
        self.num_of_seqs = 2 #CT PT
        
        self.root_directory = root_directory
        self.patch_size = patch_size

        self.transforms = transform
        self.num_workers = num_workers

        self.clinical_data = self.make_data(clinical_data_path)
        self.time_bins = make_time_bins(times=self.clinical_data["time"], num_bins=time_bins, event = self.clinical_data["event"])
        self.y = encode_survival(self.clinical_data["time"].values, self.clinical_data["event"].values, self.time_bins) # single event

        self.cache_path = get_paths_to_patient_files(cache_dir, self.clinical_data['PatientID'])


    def make_data(self, path):

        try:
            X = pd.read_csv(path + '/edited/hecktor2021_patient_info_training.csv')
            y = pd.read_csv(path + '/edited/hecktor2021_patient_endpoint_training.csv')
            df = pd.merge(X, y, on="PatientID")
        except:
            df = path

        clinical_data = df
        clinical_data = clinical_data.rename(columns={"Progression": "event", "Progression free survival": "time", "TNM group":"Stage_group", "Gender (1=M,0=F)":"Gender"})

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
            #"PatientID",
            "Tobacco",
            "Alcohol",
            "Performance status",
            "HPV status (0=-, 1=+)",
            "Estimated weight (kg) for SUV",
            "CenterID",

        ]

        clinical_data = clinical_data.drop(cols_to_drop, axis=1)


        clinical_data = pd.get_dummies(clinical_data,
                                    columns=["T-stage",
                                                "Stage_group",])
        
        return clinical_data


    

    def _prepare_data(self):
     

        Parallel(n_jobs=self.num_workers)(
            delayed(self._preprocess_subject)(subject_id)
            for subject_id in self.clinical_data["PatientID"]
        )

    def _preprocess_subject(self, subject_id: str):

        print(self.root_directory)
        print(subject_id)
        
        path = os.path.join(self.root_directory, "data/hecktor_nii/"
                            "{}",f"{subject_id}"+"{}"+".nii")

        image = sitk.ReadImage(path.format("images", "_ct"))
        mask = sitk.ReadImage(path.format("masks", "_gtvt"))

        #crop the image to (patch_size)^3 patch around the tumor center
        tumour_center = find_centroid(mask)
        size = np.ceil(self.patch_size / np.asarray(image.GetSpacing())).astype(np.int) + 1
        min_coords = np.floor(tumour_center - size / 2).astype(np.int64)
        max_coords = np.floor(tumour_center + size / 2).astype(np.int64)
        min_x, min_y, min_z = min_coords
        max_x, max_y, max_z = max_coords
        image = image[min_x:max_x, min_y:max_y, min_z:max_z]

        # resample to isotropic 1 mm spacing
        reference_image = sitk.Image([self.patch_size]*3, sitk.sitkFloat32)
        reference_image.SetOrigin(image.GetOrigin())
        image = sitk.Resample(image, reference_image)

        # window image intensities to [-500, 1000] HU range
        image = sitk.Clamp(image, sitk.sitkFloat32, -500, 1000)

        sitk.WriteImage(image, os.path.join(self.cache_path, f"{subject_id}.nii"), True)


    def __getitem__(self, idx: int):
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and int
            The input-target pair.
        """
        
        try:      # training data
            # clin_var_data = self.clinical_data.drop(["target_binary", 'time', 'event', 'Study ID'], axis=1) # single event
            clin_var_data = self.clinical_data.drop(['PatientID','time', 'event'], axis=1)
        except:   # test data
            clin_var_data = self.clinical_data.drop(['PatientID'], axis=1)


        clin_var = clin_var_data.iloc[idx].to_numpy(dtype='float32')
        
        target = self.y[idx]
        
        labels = self.clinical_data.iloc[idx].to_dict()
 
        
        subject_id = self.clinical_data.iloc[idx]["PatientID"]
        # path = self.cache_path, f"{subject_id}_ct.nii.gz")
#         print('hi:', path)
        
        # image = sitk.ReadImage(path)
        # if self.transform is not None:
        #     image = self.transform(image)
        
        
        sample = dict()
        
        id_ = self.cache_path[idx][0].parent.stem

        sample['id'] = id_
        img = [self.read_data(self.cache_path[idx][i]) for i in range(self.num_of_seqs)]
        img = np.stack(img, axis=-1)
        #img = rearrange(img,'h w d c -> c h w d')
        sample['input'] = img #np.expand_dims(img, axis=0)
        
        mask = self.read_data(self.cache_path[idx][-1])
        mask = np.expand_dims(mask, axis=3)
        #mask = rearrange(mask,'h w d c->c h w d')
        sample['target_mask'] = mask
        
        if self.transforms:
            sample = self.transforms(sample)
    
        return (sample, clin_var), target, labels
    
    

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.clinical_data)
    
    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        if return_numpy:
            return nib.load(str(path_to_nifti)).get_fdata()
        return nib.load(str(path_to_nifti))

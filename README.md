<div align="center">

# TMSS: An End-to-End Transformer-based Multimodal Network for Segmentation and Survival Prediction

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](Coming)
[![Conference](https://img.shields.io/badge/Conference-MICCAI-informational)](https://conferences.miccai.org/2022/en/)
[![Dataset](https://img.shields.io/badge/Dataset-HECKTOR-blue)](https://www.aicrowd.com/challenges/miccai-2021-hecktor)
 
_Numan Saeed, Ikboljon Sobirov, Roba Al Majzoub, and Mohammad Yaqub_
  
_Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE_ 
  
_{numan.saeed, ikboljon.sobirov, roba.majzoub, mohammad.yaqub}@mbzuai.ac.ae_

</div>

## 📌&nbsp;&nbsp;Abstract

When oncologists estimate cancer patient survival, they rely on multimodal data. Even though some multimodal deep learning methods have been proposed in the literature, the majority rely on having two or more independent networks that share knowledge at a later stage in the overall model. On the other hand, oncologists do not do this in their analysis but rather fuse the information in their brain from multiple sources such as medical images and patient history. This work proposes a deep learning method that mimics oncologists' analytical behavior when quantifying cancer and estimating patient survival. We propose TMSS, an end-to-end **T**ransformer based **M**ultimodal network for **S**egmentation and **S**urvival predication that leverages the superiority of transformers that lies in their abilities to handle different modalities. The model was trained and validated for segmentation and prognosis tasks on the training dataset from the HEad & NeCK TumOR segmentation and the outcome prediction in PET/CT images challenge (HECKTOR). We show that the proposed prognostic model significantly outperforms state-of-the-art methods with a concordance index of **0.763** while achieving a comparable dice score of **0.772** to a standalone segmentation model. TMSS implementation code will be publicly available soon.


## 📌&nbsp;&nbsp;Architecture
![alt text](https://github.com/ikboljon/tmss_miccai/blob/master/TMSS_updated.png?raw=true)
**Figure 1.** An illustration of the proposed TMSS architecture and the multimodal training strategy. TMSS linearly projects EHR and multimodal images into a feature vector and feeds it into a Transformer encoder. The CNN decoder is fed with the input images, skip connection outputs at different layers, and the final layer output to perform the segmentation, whereas the prognostic end utilizes the output of the last layer of the encoder to predict the risk score.

## 📌&nbsp;&nbsp;Checkpoints
The checkpoint to the model is available at [![GoogleDrive](https://img.shields.io/badge/GoogleDrive-Checkpoint-blue)](https://drive.google.com/file/d/1FeFlXNvIrMYjrDgT6jiCqEGfV6ld6Vug/view?usp=sharing)



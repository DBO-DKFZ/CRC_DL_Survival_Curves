"""
This script includes all necessary classes for the Slide-Level_Pipeline for the binary classification task "5-year survival yes/no": "SlideDataSet", "SlideDataModule" and "SlideModel"
The SlideModel receives features of all tiles of a histological whole slide image and predicts one score

1) SlideDataSet: provides for each patient
- patient: the patient id 
- features: pre-saved features of a specified feature extractor of all tiles of specified tissue types (tumor, stroma, lymphocytes and/or mucus)
- event: the event indicator (death yes/no) 
- time: the time from the beginning of an observation period to an event or end of study or loss of contact or withdrawal from the study
- ipcw: inverse-probability-of-censoring weighting 

2) SlideModel:consists of a 
- tile-to-slide aggregation: attention-based fusion of tile feature vectors to one slide feature vector
- survival network: predicts a score between 0 and 1 
- loss function: binary cross entropy loss (torch package)
- optimizer: AdamW
- schedular: cosine schedule with warm up 
- performance metrics:  c-index (lifelines package)

"""

import os
from os import path
import torch
import numpy as np
import pandas as pd
import random
import math

import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F


from lifelines.utils import concordance_index
from torch.utils.data import DataLoader, Dataset
from fastcore.foundation import L
import transformers 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##############################
# Functions
##############################
def check_tissue(frame, tissue_type, n_min):
    type_tiles, type_percent = [], []
    for i, row in frame.iterrows():
        n_tum = len(np.where(np.asarray(row.types)==tissue_type)[0])
        type_tiles.append(n_tum)
        type_percent.append(n_tum/len(row.types))
    frame['n_tum'] = type_tiles
    frame['tum_percent'] = type_percent
    frame = frame[frame.n_tum > n_min]
    return frame

##############################
# Classes
##############################

class SlideDataSet(Dataset):
    """
    Loads all features of a patient at once 
    """
    def __init__(self, dataframe, patient_column, feature_column, event_column, duration_column, ipcw_column, nbr_features, tissue_type):
        self.df = dataframe
        self.features = L(*self.df[feature_column])  
        self.patient_id = L(*self.df[patient_column])
        self.event_column =  torch.as_tensor(self.df[event_column].astype(int).values) 
        self.duration_column = torch.as_tensor(self.df[duration_column].astype(int).values) 
        self.ipcw = torch.as_tensor(self.df[ipcw_column].astype(float).values) 
        self.nbr_features = nbr_features
        self.tissue_type = tissue_type
        
    def __len__(self):
        return len(self.patient_id)
    
    def __getitem__(self, index):
        patient = self.patient_id[index]
        feature_path = self.features[index]
        ipcw = self.ipcw[index]
        if self.tissue_type == 'STR':
            feature_path =  feature_path.replace('TUM', 'STR')
            features = torch.load(feature_path)
        elif self.tissue_type == 'TUM+STR':
            feature_path_str =  feature_path.replace('TUM', 'STR')
            feature_paths = [feature_path, feature_path_str]
            features = []
            for fp in feature_paths:
                if path.exists(fp) == True:
                    features.append(torch.load(fp))
            features = torch.concat(features)
        elif self.tissue_type == 'ALL':
            feature_path_str =  feature_path.replace('TUM', 'STR')
            feature_path_lym =  feature_path.replace('TUM', 'LYM')
            feature_path_muc =  feature_path.replace('TUM', 'MUC')
            feature_paths = [feature_path, feature_path_str, feature_path_lym, feature_path_muc]
            features = []
            for fp in feature_paths:
                if path.exists(fp) == True:
                    features.append(torch.load(fp))
            features = torch.concat(features)
        elif self.tissue_type == 'other':
            feature_path_str =  feature_path.replace('TUM', 'STR')
            feature_path_lym =  feature_path.replace('TUM', 'LYM')
            feature_path_muc =  feature_path.replace('TUM', 'MUC')
            feature_paths = [feature_path_str, feature_path_lym, feature_path_muc]
            features = []
            for fp in feature_paths:
                if path.exists(fp) == True:
                    features.append(torch.load(fp))
            features = torch.concat(features)
        else:
            features = torch.load(feature_path)
            
        if (self.nbr_features != None):
            if features.shape[0] >= 1000: 
                nbr = random.randint(1000,features.shape[0]-1)
            else: 
                nbr = features.shape[0]-1
            indices = random.sample(range(1, features.shape[0]), nbr)
            features = features[indices]
            
        event = self.event_column[index]
        duration = self.duration_column[index]
        return patient, features, event, duration, ipcw

class SlideDataModule(pl.LightningDataModule): 
    def __init__(self, train, val, test, patient_column, feature_column, event_column, duration_column, ipcw_column, nbr_features, tissue_type):
        super().__init__()
        self.train_df = train
        self.valid_df = val
        self.test_df = test
        self.patient_column = patient_column
        self.feature_column = feature_column
        self.event_column = event_column
        self.duration_column = duration_column
        self.ipcw_column = ipcw_column
        self.nbr_features = nbr_features
        self.tissue_type = tissue_type

    def prepare_data(self):
        pass
    
    def setup(self, stage): 
        if stage == "fit": 
            self.train_ds = SlideDataSet(self.train_df, self.patient_column, self.feature_column, self.event_column, self.duration_column, self.ipcw_column, self.nbr_features,  self.tissue_type)
            self.nbr_features = None
            self.valid_ds = SlideDataSet(self.valid_df, self.patient_column, self.feature_column, self.event_column, self.duration_column, self.ipcw_column, self.nbr_features,  self.tissue_type)
            
        if stage == "test":
            self.nbr_features = None 
            self.test_ds = SlideDataSet(self.test_df, self.patient_column, self.feature_column, self.event_column, self.duration_column, self.ipcw_column, self.nbr_features,  self.tissue_type)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=None, batch_sampler=None, num_workers=6, pin_memory=True) # shuffle = True 
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, batch_size=None, batch_sampler=None, num_workers=6)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=None, batch_sampler=None, num_workers=6)


    
class SlideModel_Ilse(pl.LightningModule):
    def __init__(self, num_durations, feature_length, survnet_l1, dropout_survnet, survnet_l2, lr, wd, survival_train, survival_test, num_warmup_epochs):
        super().__init__()
        self.save_hyperparameters()  
        self.criterion = nn.BCEWithLogitsLoss()
     
        self.out_features = num_durations
        self.L = feature_length
        self.S_l1 = survnet_l1
        self.dropout_S = dropout_survnet
        self.S_l2 = survnet_l2
        self.lr = lr
        self.wd = wd
        
        self.survival_train = survival_train
        self.survival_test = survival_test
        self.num_warmup_epochs = num_warmup_epochs
        
        self.norm1 = nn.LayerNorm(self.L)
        self.norm2 = nn.LayerNorm(self.L)
        
        self.Ilse_attention = nn.Sequential(
            nn.Linear(self.L, self.L), 
            nn.Tanh(), 
            nn.Linear(self.L, 1)
        )
        
        self.surv_net = nn.Sequential(
            nn.Linear(self.L,  self.S_l1), nn.ReLU(),
            nn.Dropout(self.dropout_S),
            nn.Linear(self.S_l1,  self.S_l2), nn.ReLU(), 
            nn.Dropout(self.dropout_S),
            nn.Linear(self.S_l2, self.out_features))
        
    def forward(self, features, patient, event, duration): 
        features = self.norm1(features) # 1xNx512
        A = self.Ilse_attention(features) # out = NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A,features) # Nx512
        M = self.norm2(M)
        phi = self.surv_net(M)
        return phi, A, duration    
    
    def training_step(self, batch, batch_idx):
        patient, features, event, duration, ipcw = batch
        logits, A, duration = self(features, patient, event, duration) 
        event = event.unsqueeze(0).unsqueeze(0)
        loss = self.criterion(logits, event.float())*ipcw
        return {"loss": loss,
                "logits": logits.detach().sigmoid().cpu(), 
                "event": event.detach().cpu(), 
                "duration": duration.detach().cpu()}
    
    def training_epoch_end(self, outs):
        predictions = torch.vstack([1-x['logits'] for x in outs])
        durations = torch.vstack([x['duration'] for x in outs])
        events = torch.vstack([x['event'] for x in outs])
        c_index = concordance_index(durations.numpy().reshape(-1, ), np.asarray(predictions), events.numpy().reshape(-1, ))
        
        self.log("train/loss", torch.mean(torch.stack([x['loss'] for x in outs])), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/Cindex", c_index, on_epoch=True, logger=True)

        
    def validation_step(self, batch, batch_idx):
        patient, features, event, duration, ipcw = batch
        logits, A, duration = self(features, patient, event, duration) 
        event = event.unsqueeze(0).unsqueeze(0)
        loss = self.criterion(logits, event.float())*ipcw
        
        return {"loss": loss,
                "logits": logits.detach().sigmoid().cpu(), 
                "event": event.detach().cpu(), 
                "duration": duration.detach().cpu()}
    
    
    def validation_epoch_end(self, outs):
        predictions = torch.vstack([1-x['logits'] for x in outs])
        durations = torch.vstack([x['duration'] for x in outs])
        events = torch.vstack([x['event'] for x in outs])
        c_index = concordance_index(durations.numpy().reshape(-1, ), np.asarray(predictions), events.numpy().reshape(-1, ))
        
        self.log("valid/loss", torch.mean(torch.stack([x['loss'] for x in outs])), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid/Cindex", c_index, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outs):
        return self.validation_epoch_end(outs)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay=self.wd) # lr=0.00001, wd = 0.00001
        num_epoch_steps = (
            len(self.trainer.datamodule.train_dataloader())
            // self.trainer.accumulate_grad_batches
            // self.trainer.world_size
        )
        num_warmup_steps = num_epoch_steps * self.num_warmup_epochs
        
        if self.trainer.max_steps != -1:
            num_training_steps = self.trainer.max_steps
        else:
            num_training_steps = num_epoch_steps * self.trainer.max_epochs

        kwargs = {
            "optimizer": opt,
            "num_warmup_steps": num_warmup_steps,
            "num_training_steps": num_training_steps,
        }
        shd = {"interval": "step"}
        shd["scheduler"] = transformers.get_cosine_schedule_with_warmup(**kwargs)
        return [opt], [shd]

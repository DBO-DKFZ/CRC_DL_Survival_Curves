"""
This script includes all necessary classes for the Time-to-Event Slide-Level-Pipeline (TtoE SLP): "SlideDataSet", "SlideDataModule" and "SlideModel"
The SlideModel receives all tiles of a histological whole slide image and predicts the monthly hazards of the first 60 months after diagnosis which is transformed to a five-year survival curve (piecewise constant hazard assumption)

1) SlideDataSet: provides for each patient
- patient: the patient id 
- features: pre-saved features of a specified feature extractor of all tiles of specified tissue types (tumor, stroma, lymphocytes and/or mucus)
- event: the event indicator (death yes/no) 
- time: the time from the beginning of an observation period to an event or end of study or loss of contact or withdrawal from the study
- idx_duration: time interval the time falls into (e.g. first month, second month)
- interval_frac: fraction of the last time interval before an event or censoring (e.g. if the time was given in days instead of months, then the hazard of the month the time falls into is considered weighted by its fraction (e.g. april 15th -> half of the hazard of april)) 

2) SlideModel:consists of a 
- tile-to-slide aggregation: attention-based fusion of tile feature vectors to one slide feature vector
- survival network: predicts the monthly hazards for the first five years based on the slide feature
- loss function: negative log likelihood (pycox package)
- optimizer: AdamW
- schedular: cosine schedule with warm up 
- performance metrics: time dependent c-index (pycox package), integrated brier score (sksurv)

Relevant references:
Kvamme H, Borgan Ø. Continuous and Discrete-Time Survival Prediction with Neural Networks. arXiv [stat.ML], http://arxiv.org/abs/1910.06724 (2019)
"""

import os
import math
import torch
import random
from os import path
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from fastcore.foundation import L
import transformers 
import torch.nn.functional as F

from pycox.models import PCHazard
from pycox.models.utils import pad_col, make_subgrid
from pycox.evaluation import EvalSurv
from pycox.models.loss import  nll_pc_hazard_loss

from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score

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

def predict_surv_df(preds, sub, duration_index):
    n = preds.shape[0]
    hazard = F.softplus(preds).view(-1, 1).repeat(1, sub).view(n, -1).div(sub) # Formel 19
    hazard = pad_col(hazard, where='start')
    surv = hazard.cumsum(1).mul(-1).exp() # Formal 20 
    surv = surv.detach().cpu().numpy()
    index = None
    if duration_index is not None:
        index = make_subgrid(duration_index, sub)
    return pd.DataFrame(surv.transpose(), index) # shape [num-duration+1 x samples N]

##############################
# Classes
##############################

class SlideDataSet(Dataset):
    """
    Loads all features of a patient at once 
    """
    def __init__(self, dataframe, patient_column, feature_column, event_column, duration_column, idx_dur_column, interval_frac_column, nbr_features, tissue_type):
        self.df = dataframe
        self.features = L(*self.df[feature_column])  
        self.patient_id = L(*self.df[patient_column])
        self.event_column =  torch.as_tensor(self.df[event_column].astype(int).values) 
        self.duration_column = torch.as_tensor(self.df[duration_column].astype(int).values) 
        self.idx_duration = torch.as_tensor(self.df[idx_dur_column].astype(int).values) 
        self.interval_frac = torch.as_tensor(self.df[interval_frac_column].values) 
        self.nbr_features = nbr_features
        self.tissue_type = tissue_type
        
    def __len__(self):
        return len(self.patient_id)
    
    def __getitem__(self, index):
        patient = self.patient_id[index]
        feature_path = self.features[index]
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
        idx_duration = self.idx_duration[index]
        interval_frac = self.interval_frac[index]
        return patient, features, event, duration, idx_duration, interval_frac

class SlideDataModule(pl.LightningDataModule): 
    def __init__(self, train, val, test, patient_column, feature_column, event_column, duration_column, idx_dur_column, interval_frac_column, nbr_features, tissue_type):
        super().__init__()
        self.train_df = train
        self.valid_df = val
        self.test_df = test
        self.patient_column = patient_column
        self.feature_column = feature_column
        self.event_column = event_column
        self.duration_column = duration_column
        self.idx_dur_column = idx_dur_column
        self.interval_frac_column = interval_frac_column
        self.nbr_features = nbr_features
        self.tissue_type = tissue_type

    def prepare_data(self):
        pass
    
    def setup(self, stage): 
        if stage == "fit": 
            self.train_ds = SlideDataSet(self.train_df, self.patient_column, self.feature_column, self.event_column, self.duration_column, self.idx_dur_column, self.interval_frac_column, self.nbr_features,  self.tissue_type)
            self.nbr_features = None
            self.valid_ds = SlideDataSet(self.valid_df, self.patient_column, self.feature_column, self.event_column, self.duration_column, self.idx_dur_column, self.interval_frac_column, self.nbr_features,  self.tissue_type)
            
        if stage == "test":
            self.nbr_features = None 
            self.test_ds = SlideDataSet(self.test_df, self.patient_column, self.feature_column, self.event_column, self.duration_column, self.idx_dur_column, self.interval_frac_column, self.nbr_features,  self.tissue_type)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=None, batch_sampler=None, num_workers=6, pin_memory=True) # shuffle = True 
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, batch_size=None, batch_sampler=None, num_workers=6)
       
    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=None, batch_sampler=None, num_workers=6)


    
class SlideModel_Ilse(pl.LightningModule):
    def __init__(self, duration_index, num_durations, feature_length, survnet_l1, dropout_survnet, survnet_l2, lr, wd, survival_train, survival_test, num_warmup_epochs):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nll_pc_hazard_loss
        self.duration_index = duration_index
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
        
    def forward(self, features, patient, event, duration, idx_duration): 
        features = self.norm1(features) # 1xNx512
        A = self.Ilse_attention(features) # out = NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A,features) # Nx512
        M = self.norm2(M)
        phi = self.surv_net(M)
        return phi, A, duration    
    
    def training_step(self, batch, batch_idx):
        patient, features, event, duration, idx_duration, interval_frac = batch
        logits, A, duration = self(features, patient, event, duration, idx_duration) 
        loss = self.criterion(logits, idx_duration, event.float(), interval_frac)
        return {"loss": loss,
                "logits": logits.detach().cpu(), 
                "event": event.detach().cpu(), 
                "duration": duration.detach().cpu()}
    
    def training_epoch_end(self, outs):
        predictions = torch.vstack([x['logits'] for x in outs])
        durations = torch.vstack([x['duration'] for x in outs])
        events = torch.vstack([x['event'] for x in outs])
        surv_df = predict_surv_df(predictions, sub=1, duration_index=self.duration_index)
        ev = EvalSurv(surv_df, durations.numpy().reshape(-1, ), events.numpy().reshape(-1, ))
        surv_df = np.transpose(surv_df.to_numpy())
        n_times = np.arange(self.survival_train['time_curated'].min(),self.survival_train['time_curated'].max()+1)
        
        surv_df = surv_df[:,n_times]
        ibs = integrated_brier_score(self.survival_train, self.survival_train, surv_df, n_times)
        self.log("train/loss", torch.mean(torch.stack([x['loss'] for x in outs])), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/Cindex", ev.concordance_td(), on_epoch=True, logger=True)
        self.log("train/IBS", ibs, on_epoch=True, logger=True)

        
    def validation_step(self, batch, batch_idx):
        patient, features, event, duration, idx_duration, interval_frac = batch
        logits, att, duration = self(features, patient, event, duration, idx_duration) 
        loss = self.criterion(logits, idx_duration, event.float(), interval_frac)
        return {"loss": loss,
                "logits": logits.detach().cpu(), 
                "event": event.detach().cpu(), 
                "duration": duration.detach().cpu()}
    
    def validation_epoch_end(self, outs):
        predictions = torch.vstack([x['logits'] for x in outs])
        durations = torch.vstack([x['duration'] for x in outs])
        events = torch.vstack([x['event'] for x in outs])
        surv_df = predict_surv_df(predictions, sub=1, duration_index=self.duration_index)
        ev = EvalSurv(surv_df, durations.numpy().reshape(-1, ), events.numpy().reshape(-1, ))
        ci_last = concordance_index(durations.numpy().reshape(-1, ), np.asarray(surv_df.iloc[self.out_features,:]), events.numpy().reshape(-1, ))
        surv_df = np.transpose(surv_df.to_numpy())
        n_times = np.arange(self.survival_test['time_curated'].min(),self.survival_test['time_curated'].max()+1)
        print(n_times)
        surv_df = surv_df[:,n_times]
        ibs = integrated_brier_score(self.survival_train, self.survival_test, surv_df, n_times)
        self.log("valid/loss", torch.mean(torch.stack([x['loss'] for x in outs])), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid/Cindex", ev.concordance_td(), on_step=False, on_epoch=True, logger=True)
        self.log("valid/Cindex_last", ci_last, on_step=False, on_epoch=True, logger=True)
        self.log("valid/IBS", ibs, on_epoch=True, logger=True)

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

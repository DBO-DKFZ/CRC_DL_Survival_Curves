import os
import torch
import random
import timm
import re
import transformers
import pandas as pd
import pytorch_lightning as pl
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import Normalize
from torchmetrics import Accuracy 

from collections import OrderedDict
from tqdm.notebook import tqdm

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.transforms import ToTensor

from torchvision import transforms
from os import walk



class Subtyper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics 
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

        
        model = timm.create_model('resnet18')
        model.fc = nn.Conv2d(512,1,1)
        ckpt = torch.load(os.path.join("/nvidia-resnet18.pt"), map_location="cpu") # path to the nvidia-checkpoint has to be added 
        model.load_state_dict(OrderedDict(zip(model.state_dict().keys(), ckpt.values())))
        model.reset_classifier(0)
        
        self.model = model 
        self.classifier = nn.Sequential(nn.Dropout(0.70), nn.Linear(512*1, 9))
        
        
    def forward(self, tiles): 
        rep = self.model(tiles)
        y_prob = self.classifier(rep)
        return y_prob  
    
    def training_step(self, batch, batch_idx):
        tiles, y = batch
        logits, y = self(tiles, y) 
        logits = logits.squeeze(1).float()
        loss = self.criterion(logits, y)
        self.log("train/acc", self.train_acc(logits.sigmoid(), y),on_step=False, on_epoch=True, logger=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
        "loss": loss, 
        "slide_score": logits.detach().sigmoid(), 
        "y": y.detach()
        }
    
    def training_epoch_end(self, outs):
        pass

        
    def validation_step(self, batch, batch_idx):
        tiles, y = batch
        logits, y = self(tiles, y)
        logits = logits.squeeze(1).float()
        loss = self.criterion(logits, y)
        self.log("valid/acc", self.valid_acc(logits.sigmoid(), y), on_epoch=True, logger=True)
        self.log("valid/loss", loss)
        return {
        "loss": loss, 
        "slide_score": logits.detach().sigmoid(), 
        "y": y.detach()
        }
    
    def validation_epoch_end(self, outs):
        pass
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outs):
        return self.validation_epoch_end(outs)
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr = 0.000001, weight_decay=0.000001) 
        return opt
        
class Subtyper_Finetune(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics 
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

        self.model = model.model
        # split model into seperate blocks to decide which to fine-tune
        self.model_part1 =  nn.Sequential(*list(self.model.children())[:-3])
        last_resblock = nn.Sequential(*list(self.model.children())[7])
        self.basicblock0 = nn.Sequential(list(last_resblock.children())[0])
        self.basicblock1 = nn.Sequential(list(last_resblock.children())[1])
        self.model_part2 =  nn.Sequential(*list(self.model.children())[8:])
        self.classifier = model.classifier
        
        self.model_part1.requires_grad_(False) 
        self.basicblock0.requires_grad_(True) 
        self.basicblock1.requires_grad_(True)
        self.model_part2.requires_grad_(True)
        self.classifier.requires_grad_(True)
        
        
    def forward(self, tiles): 
        tiles = self.model_part1(tiles)
        tiles = self.basicblock0(tiles)
        tiles = self.basicblock1(tiles)
        rep = self.model_part2(tiles)
        y_prob = self.classifier(rep)
        return y_prob   
    
    def training_step(self, batch, batch_idx):
        tiles, y = batch
        logits, y = self(tiles, y) 
        logits = logits.squeeze(1).float()
        loss = self.criterion(logits, y)
        self.log("train/acc", self.train_acc(logits.sigmoid(), y),on_step=False, on_epoch=True, logger=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
        "loss": loss, 
        "slide_score": logits.detach().sigmoid(), 
        "y": y.detach()
        }
    
    def training_epoch_end(self, outs):
        pass

        
    def validation_step(self, batch, batch_idx):
        tiles, y = batch
        logits, y = self(tiles, y)
        logits = logits.squeeze(1).float()
        loss = self.criterion(logits, y)
        self.log("valid/acc", self.valid_acc(logits.sigmoid(), y), on_epoch=True, logger=True)
        self.log("valid/loss", loss)
        return {
        "loss": loss, 
        "slide_score": logits.detach().sigmoid(), 
        "y": y.detach()
        }
    
    def validation_epoch_end(self, outs):
        pass
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outs):
        return self.validation_epoch_end(outs)
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr = 0.000001, weight_decay=0.000001)
        return opt


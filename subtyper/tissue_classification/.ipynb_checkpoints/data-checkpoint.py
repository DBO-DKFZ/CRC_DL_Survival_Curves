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
from torchvision.transforms import ToTensor

from torchvision import transforms
from os import walk


class DataSet_Subtyper(Dataset):
    def __init__(self, dataframe,tile_column, tile_tfms):
        self.df = dataframe
        self.tiles = self.df[tile_column]
        self.tile_tfms = tile_tfms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(self.tiles[index]) 
        return self.tile_tfms(image)
    
class DataModule(pl.LightningDataModule): 
    def __init__(self, train, val, test, tile_column, label_column, batch_size):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.tile_column = tile_column
        self.label_column = label_column
        self.bs = batch_size 
        
    def prepare_data(self):
        pass
    
    def setup(self, stage): 
        if stage == "fit": 
            self.train_df = pd.read_pickle(self.train)
            self.valid_df =  pd.read_pickle(self.val)
            self.train_ds = SlideDataSet(self.train_df, self.tile_column, self.label_column) 
            self.valid_ds = SlideDataSet(self.valid_df, self.tile_column, self.label_column)
        if stage == "test":
            self.test_df = pd.read_pickle(self.test)
            self.test_ds = SlideDataSet(self.test_df, self.tile_column, self.label_column)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.bs, batch_sampler=None, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, batch_size=self.bs, batch_sampler=None, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.bs, batch_sampler=None, num_workers=24)        
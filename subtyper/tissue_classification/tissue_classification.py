"""
Tissue classification script that classifies each tile image as one of nine different colorectal tissue types 
"""

import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from tqdm import *

from subtyper import Subtyper, Subtyper_Finetune
from data import DataSet_Subtyper, DataModule


def subtyper(args): # data, ckpt
    df = pd.read_pickle(args.pickle_file)
    
    train_tfms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.5), 
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,5)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    test_tfms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    # initialize model
    subtyper_wo_finetuning = Subtyper()
    subtyper = Subtyper_Finetune(subtyper_wo_finetuning)

    # load weights 
    ckpt_details = torch.load(args.ckpt, map_location="cpu")
    subtyper.load_state_dict(OrderedDict(zip(subtyper.state_dict().keys(), ckpt_details['state_dict'].values())))
    subtyper = subtyper.eval() 
    subtyper.freeze()
    device = torch.device("cuda")
    
    subtyper.to(device); 
    
    types = []
    for i, row in tqdm(list(df.iterrows()), desc="Patient"):
        if row.tiles == []:
            types.append(np.nan)
        else:
            df_patient = pd.DataFrame({'tiles': row.tiles})
            ds = DataSet_Subtyper(df_patient, 'tiles', test_tfms) 
            dl = DataLoader(ds, shuffle=False, batch_size=args.bs, batch_sampler=None, num_workers=args.num_workers, pin_memory=True)
            y_pred = []
            y_scores = []
            with torch.inference_mode():
                for image in dl:
                    y_prob = subtyper(image.to(device, non_blocking=True))
                    y_prob = torch.softmax(y_prob, dim=1)
                    y_pred.extend(torch.argmax(y_prob, dim=1).cpu().tolist())
                types.append(y_pred)
    df['types'] = types
    df.to_pickle(args.pickle_file.split('.')[0]+'_types.pkl')

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='You can add a description here')
    parser.add_argument('--pickle_file', default='/path/to/pickle_file.pkl', type=str)
    parser.add_argument('--bs', default=1000, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--ckpt', default='/.../subtyper.ckpt', type=str) 
    args = parser.parse_args()
    subtyper(args)
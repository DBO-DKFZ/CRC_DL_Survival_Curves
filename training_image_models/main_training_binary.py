import os
import math
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import argparse
import TtoE_SLP_binary as TtoE_SLP

from lifelines import KaplanMeierFitter

def class_label_5(df):
    df['labels'] = 0  # Initialize the "labels" column with default value 0

    # Create conditions for assigning labels
    condition_1 = (df['time_curated'] <= 59) & (df['status_curated'] == 1)
    condition_2 = (df['time_curated'] == 59) & (df['status_curated'] == 0)
    condition_3 = (df['time_curated'] < 59) & (df['status_curated'] == 0)

    # Assign labels based on conditions
    df.loc[condition_1, 'labels'] = 1
    df.loc[condition_2, 'labels'] = 0
    df.loc[condition_3, 'labels'] = 2
    return df

def calculate_ipcw(df):
    time = np.asarray(df.Fumonths.tolist())# Time of events or censoring
    event = np.asarray(df.death_event.tolist())  # Event indicator (1 for event, 0 for censoring)
    label = np.asarray(df.labels.tolist())  
    kmf = KaplanMeierFitter()
    kmf.fit(time, 1-event)
    unique_times = np.unique(time)
    no_censoring_probability = kmf.survival_function_at_times(unique_times).values.flatten() # g 
    dic = dict(zip(unique_times, no_censoring_probability))
    ipcw = []
    for t,e,l in zip(time, event, label):
        if l == 2:
            ipcw.append(0)
        elif l == 1: 
            ipcw.append(1/dic[t])
        else:
            ipcw.append(1/dic[59])
    df['ipcw'] = ipcw
    print(f'IPCW sanity check: {len(df)} ~ {df.ipcw.sum()}')
    return df, dic

def apply_ipcw(df, ipcw_dict):
    time = np.asarray(df.Fumonths.tolist())# Time of events or censoring
    event = np.asarray(df.death_event.tolist())  # Event indicator (1 for event, 0 for censoring)
    label = np.asarray(df.labels.tolist())
    ipcw = []
    for t,e,l in zip(time, event, label):
        if l == 2:
            ipcw.append(0)
        elif l == 1: 
            ipcw.append(1/ipcw_dict[t])
        else:
            ipcw.append(1/ipcw_dict[59])
    df['ipcw'] = ipcw
    return df
    
def train_binary(args):
        # training set was split into five folds -> five models are trained on four folds and validated on the fiths
    for fold in ['0', '1', '2', '3', '4']:
        exp_name = args.exp_name + '_' + args.subgroup + '_' + args.tissue_type + '_' + fold +'_' + args.feature_column + '_' + args.attention
        print('Experiment name: ', exp_name)
        
        # load the data sets 
        train_pkl = args.train_pkl.split('.')[0] + '_' + fold + '.pkl'
        val_pkl = args.val_pkl.split('.')[0] + '_' + fold + '.pkl'
        df_train = pd.read_pickle(train_pkl)
        df_val = pd.read_pickle(val_pkl)
        df_test = pd.read_pickle(args.test_pkl)
        print(f'Size of Training/Validation/Testset: {len(df_train)}/{len(df_val)}/{len(df_test)}')
        
        # 5 year label: 0 no death within 5 years, 1: death within 5 years, 2: censored  
        df_train = class_label_5(df_train)
        df_val = class_label_5(df_val)
        df_test = class_label_5(df_test)
        
        # ipcw to account for censoring (will be used to weight the loss) 
        df_train, ipcw_dict = calculate_ipcw(df_train)
        df_val = apply_ipcw(df_val, ipcw_dict)
        df_test = apply_ipcw(df_test, ipcw_dict)
        
        # exclude all censored cases
        df_train = df_train[df_train.labels != 2]
        df_val = df_val[df_val.labels != 2]
        df_test = df_test[df_test.labels != 2]
        print(f'Exclude censored data (label=2): {len(df_train)}/{len(df_val)}/{len(df_test)}')
        
        # time is given in months; transform [0,..., 59] into [0, ..., 60]
        df_train[args.duration_column] = df_train[args.duration_column]+1
        df_val[args.duration_column] = df_val[args.duration_column]+1
        df_test[args.duration_column] = df_test[args.duration_column]+1
        
        #cleaning step: patients have to be excluded if the slide of the patient had not enough tiles classified as the tissue type of interest 
        df_train = TtoE_SLP.check_tissue(df_train, args.tissue_check, args.tissue_min)
        df_val = TtoE_SLP.check_tissue(df_val, args.tissue_check, args.tissue_min)
        df_test = TtoE_SLP.check_tissue(df_test, args.tissue_check, args.tissue_min)
        print(f'Exclude Patients with less than {args.tissue_min} tumor tiles. Training/Validation/Testset: {len(df_train)}/{len(df_val)}/{len(df_test)}')
        
        # makes sure that the time is given as integer 
        df_train[args.duration_column] = df_train[args.duration_column].apply(lambda x:int(x))
        df_val[args.duration_column] = df_val[args.duration_column].apply(lambda x:int(x))
        df_test[args.duration_column] = df_test[args.duration_column].apply(lambda x:int(x))
        
        # Dataloader
        dm = TtoE_SLP.SlideDataModule(df_train, df_val, df_test, args.patient_column, args.feature_column, args.label, args.duration_column, 'ipcw', args.nbr_features, args.tissue_type)
        
        # brier score (is not traced in this training script) 
        survival_train =  df_train[[args.event_column, args.duration_column]].astype(int)
        survival_test =  df_val[[args.event_column, args.duration_column]].astype(int)
        survival_train[args.event_column] = survival_train[args.event_column].astype(bool)
        survival_test[args.event_column] = survival_test[args.event_column].astype(bool)
        survival_train = survival_train.to_records(index=False)
        survival_test = survival_test.to_records(index=False)
        print(df_train.time_curated.min(), df_train.time_curated.max())
        print(df_val.time_curated.min(), df_val.time_curated.max())
        
        # slide model 
        m = TtoE_SLP.SlideModel_Ilse(args.num_durations, args.feature_length, args.survnet_l1, args.dropout_survnet, args.survnet_l2, args.lr, args.wd, survival_train, survival_test, args.num_warmup_epochs)

        # training 
        logg_path = os.path.join(args.logg_path, "logs", exp_name)
        logger = pl.loggers.TensorBoardLogger(logg_path, name=None)
        lr_monitor = pl.callbacks.lr_monitor.LearningRateMonitor()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=args.monitor, save_top_k=1, mode=args.monitor_mode, save_last=True)
        trainer = pl.Trainer(accelerator='gpu', devices=args.gpu, callbacks=[checkpoint_callback, lr_monitor], logger=logger, num_sanity_val_steps=0, max_epochs=args.epochs, accumulate_grad_batches=args.acc_grad_batches)
        trainer.fit(m, datamodule=dm)

        print('Done')

if __name__ == "__main__":
    #Parser
    parser = argparse.ArgumentParser(description='You can add a description here')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--exp_name', default='OS', type=str)
    parser.add_argument('--fold', default='0', type=str)
    parser.add_argument('--subgroup', default='all', type=str)
    parser.add_argument('--tissue_type', default='TUM', type=str)
    parser.add_argument('--train_pkl', default='/path_to/dachs_train_fold.pkl', type=str)
    parser.add_argument('--val_pkl', default='/path_to/dachs_val_fold.pkl', type=str)
    parser.add_argument('--test_pkl', default='/path_to/dachs_test.pkl', type=str) # not used during training
    parser.add_argument('--patient_column', default='tn_id', type=str)
    parser.add_argument('--label', default='labels', type=str)
    parser.add_argument('--feature_column', default='rand_features', type=str)
    parser.add_argument('--event_column', default='status_curated', type=str)
    parser.add_argument('--duration_column', default='time_curated', type=str)
    parser.add_argument('--num_durations', default=60, type=int) 
    parser.add_argument('--tissue_check', default=8, type=int)
    parser.add_argument('--tissue_min', default=0, type=int)
    parser.add_argument('--feature_length', default=512, type=int)
    parser.add_argument('--survnet_l1', default=512, type=int) # 256
    parser.add_argument('--survnet_l2', default=256, type=int) # 128
    parser.add_argument('--dropout_survnet', default=0.5, type=float)
    parser.add_argument('--reduction_l1', default=256, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--wd', default=1e-6, type=float)
    parser.add_argument('--nbr_features', default=None, type=int)
    parser.add_argument('--monitor', default="valid/loss", type=str)
    parser.add_argument('--monitor_mode', default="min", type=str)
    parser.add_argument('--epochs', default=200, type=int) #100
    parser.add_argument('--num_warmup_epochs', default=20, type=int) #100
    parser.add_argument('--acc_grad_batches', default=1, type=int)
    parser.add_argument('--gpu', default="0,", type=str)
    parser.add_argument('--logg_path', default="/path_to_log/", type=str) # checkpoints are saved 
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    
    train_binary(args)

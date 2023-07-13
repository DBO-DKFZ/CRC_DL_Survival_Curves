import os
import math
import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pycox.models import PCHazard
import TtoE_SLP_curve as TtoE_SLP

def train_TtoE(args):
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
        
        # time is given in months; transform [0,..., 59] into [0, ..., 60]
        df_train[args.duration_column] = df_train[args.duration_column]+1
        df_val[args.duration_column] = df_val[args.duration_column]+1
        df_test[args.duration_column] = df_test[args.duration_column]+1
        
        #cleaning step: patients have to be excluded if the slide of the patient had not enough tiles classified as the tissue type of interest 
        df_train = TtoE_SLP.check_tissue(df_train, args.tissue_check, args.tissue_min)
        df_val = TtoE_SLP.check_tissue(df_val, args.tissue_check, args.tissue_min)
        df_test = TtoE_SLP.check_tissue(df_test, args.tissue_check, args.tissue_min)
        print(f'Exclude Patients with less than {args.tissue_min} tiles of the given tissue type. Training/Validation/Testset: {len(df_train)}/{len(df_val)}/{len(df_test)}')

        # makes sure that the time is given as integer 
        df_train[args.duration_column] = df_train[args.duration_column].apply(lambda x:int(x))
        df_val[args.duration_column] = df_val[args.duration_column].apply(lambda x:int(x))
        df_test[args.duration_column] = df_test[args.duration_column].apply(lambda x:int(x))

        # transforms necessary for the piecewise constant hazard approach
        labtrans = PCHazard.label_transform(args.num_durations)
        get_target = lambda df: (df[args.duration_column].values, df[args.event_column].values)
        y_train_surv = labtrans.fit_transform(*get_target(df_train))
        y_val_surv = labtrans.transform(*get_target(df_val))
        y_test_surv = labtrans.transform(*get_target(df_test))
        df_train[args.idx_dur_column] = y_train_surv[0]
        df_val[args.idx_dur_column] = y_val_surv[0]
        df_test[args.idx_dur_column] = y_test_surv[0]
        df_train[args.interval_frac_column] = y_train_surv[2] # anteil des events innerhalb des Intervals, zw 0 ...1; 0 am Anfang des Intervals, 1 am Ende
        df_val[args.interval_frac_column] = y_val_surv[2]
        df_test[args.interval_frac_column] = y_test_surv[2]

        # Dataloader
        dm = TtoE_SLP.SlideDataModule(df_train, df_val, df_test, args.patient_column, args.feature_column, args.event_column, args.duration_column, args.idx_dur_column, args.interval_frac_column, args.nbr_features, args.tissue_type)
        
        # if other than monthly hazards should be used 
        if args.num_durations !=60:
            print('adjust time_curated for one year')
            devider = int(60/args.num_durations)
            print(devider)
            df_train[args.duration_column] = [math.ceil(time/devider) for time in df_train[args.duration_column]]
            df_val[args.duration_column] = [math.ceil(time/devider) for time in df_val[args.duration_column]]
            
        # necessary for integrated brier score: 
        survival_train =  df_train[[args.event_column, args.duration_column]].astype(int)
        survival_test =  df_val[[args.event_column, args.duration_column]].astype(int)
        survival_train[args.event_column] = survival_train[args.event_column].astype(bool)
        survival_test[args.event_column] = survival_test[args.event_column].astype(bool)
        survival_train = survival_train.to_records(index=False)
        survival_test = survival_test.to_records(index=False)
        
        # Slide model 
        m = TtoE_SLP.SlideModel_Ilse(labtrans.cuts, args.num_durations, args.feature_length, args.survnet_l1, args.dropout_survnet, args.survnet_l2, args.lr, args.wd, survival_train, survival_test, args.num_warmup_epochs)

        # training: 
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
    parser.add_argument('--test_pkl', default='/path_to/dachs_test.pkl', type=str) # internal testset
    parser.add_argument('--patient_column', default='tn_id', type=str)
    parser.add_argument('--feature_column', default='rand_features', type=str)
    parser.add_argument('--event_column', default='status_curated', type=str)
    parser.add_argument('--duration_column', default='time_curated', type=str)
    parser.add_argument('--idx_dur_column', default='idx_dur_column', type=str)
    parser.add_argument('--interval_frac_column', default='interval_frac_column', type=str)
    parser.add_argument('--num_durations', default=60, type=int)
    parser.add_argument('--tissue_check', default=8, type=int) # 8 stands for tumor tissue; all nine tissue types are represented by numbers 0, ..., 8
    parser.add_argument('--tissue_min', default=0, type=int)
    parser.add_argument('--feature_length', default=512, type=int)
    parser.add_argument('--survnet_l1', default=512, type=int) 
    parser.add_argument('--survnet_l2', default=256, type=int) 
    parser.add_argument('--dropout_survnet', default=0.5, type=float)
    parser.add_argument('--attention', default='ilse', type=str)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--wd', default=1e-6, type=float)
    parser.add_argument('--nbr_features', default=None, type=int)
    parser.add_argument('--monitor', default="valid/loss", type=str)
    parser.add_argument('--monitor_mode', default="min", type=str)
    parser.add_argument('--epochs', default=200, type=int) #100
    parser.add_argument('--num_warmup_epochs', default=20, type=int) #100
    parser.add_argument('--acc_grad_batches', default=1, type=int)
    parser.add_argument('--gpu', default="0,", type=str)
    parser.add_argument('--logg_path', default="/path_to_log/", type=str) # checkpoints are saved here
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    
    train_TtoE(args)

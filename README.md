# CRC_DL_Survival_Curves

This code belongs to the paper "Investigating colorectal cancer risk stratification on histological slides based on survival curves". 
More code will be coming soon. 

![Pipeline](/images/DL_survival_curve_prediction.png)

## **Abstract**
Studies have shown that colorectal cancer prognosis can be predicted by deep learning-based analysis of histological tissue sections of the primary tumor. So far, this has been achieved using a binary prediction. Survival curves might contain more detailed information and thus enable a more fine-grained risk prediction. Therefore, we established survival curve-based CRC survival predictors and benchmarked them against standard binary survival predictors, comparing their performance extensively on the clinical high- and low-risk subsets of one internal and three external cohorts.
Survival curve-based risk prediction achieved a very similar risk stratification to binary risk prediction for this task. Exchanging other components of the pipeline, namely input tissue and feature extractor, had largely identical effects on model performance independently of the type of risk prediction. An ensemble of all survival curve-based models exhibited a more robust performance, as did a similar ensemble based on binary risk prediction. Patients could be further stratified within clinical risk groups. However, performance still varied across cohorts, indicating limited generalization of all investigated image analysis pipelines, whereas models using clinical data performed robustly on all cohorts.

## **Keywords**
deep learning, survival prediction, colorectal cancer, risk refinement, generalization

## **Code**
Code for training survival curve-based CRC survival predictors: training_image_models/main_training_curve.py

Code for training binary survival predictors: training_image_models/main_training_binary.py
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:27:55 2022

@author: DELL
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import pickle
import os

import streamlit as st

#%% load the model 
MODEL_PATH = os.path.join(os.getcwd(),'saved_model','model.pkl')
MMS_SCALER_PATH = os.path.join(os.getcwd(),'saved_model', 'mms_scaler.pkl')
OHE_SCALER_PATH = os.path.join(os.getcwd(),'saved_model', 'ohe_scaler.pkl')

# load scaller and model 
model_logistic = pickle.load(open(MODEL_PATH, 'rb'))
mms_scaler = pickle.load(open(MMS_SCALER_PATH, 'rb'))
ohe_scaler = pickle.load(open(OHE_SCALER_PATH, 'rb'))

heart_attack_chances = {0:'negative', 1:'positive'}



#%% build apps using streamlit
with st.form('Heart Attack Prediction Form'):
    st.header("Patient's Info")
    
    # ages info
    ages = int(st.number_input('Ages'))
    
    # sex info
    gender = ['Male', 'Female']
    gender_option = list(range(len(gender)))
    sex = st.selectbox('Sex', gender_option, format_func= lambda x : gender[x])
    
    # cp info : chestpain type
    # 0 : typical angina
    # 1 : atypical angina
    # 2 : non-anginal pain
    cp_type = ['typical angina', 'atypical angina', 'non-anginal pain']
    cp_option = list(range(len(cp_type)))
    cp = st.selectbox('cp : type of chest pain', cp_option, format_func= lambda x : cp_type[x])

    # trtbps : resting blood pressure (in mm Hg) info
    trtbps = int(st.number_input('trtbps: resting blood pressure'))
    
    # chol : cholestrol 
    chol = int(st.number_input('chol: cholestrol'))

    
    # fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
    fbs_true = ['no', 'yes']
    fbs_option = list(range(len(fbs_true)))
    fbs = st.selectbox('fbs: fasting blood sugar > 120 mg/dl', fbs_option, format_func= lambda x : fbs_true[x])

    # restecq : Represents the level of Resting electrocardiographic
    restecq_level = ['low', 'high']
    restecq_option = list(range(len(restecq_level)))
    restecq = st.selectbox('restecq: resting electrocardiographic', restecq_option, format_func= lambda x : restecq_level[x])

    # thalachh maximum heart rate achieved
    thalachh = int(st.number_input('thalachh: maximum heart rate achieve')) 
    
    # exnq : exercise induced angina (1 = yes; 0 = no)
    exnq_true = ['no', 'yes']
    exnq_option = list(range(len(exnq_true)))
    exnq = st.selectbox('exnq: exercise induced angina', exnq_option, format_func= lambda x : exnq_true[x])

    
    # oldpeak : ST level during the workout
    oldpeak = float(st.number_input('oldpeak: st level during workout'))
    
    # slp :The Slope of the Peak Exercise ST Segment
    # 0 = Downsloping
    # 1 = Flat
    # 2 = Upsloping
    slp_segment = ['downsloping', 'flat', 'upsloping']
    slp_option = list(range(len(slp_segment)))
    slp = st.selectbox('slp: The Slope of the Peak Exercise ST Segment', slp_option, format_func= lambda x : slp_segment[x])

    
    
    # caa Number of Major Vessels (0-4) Colored by Flourosopy
    caa_no = ['0', '1', '2','3','4']
    caa_option = list(range(len(caa_no)))
    caa = st.selectbox('caa: Number of Major Vessels (0-4) Colored by Flourosopy', caa_option, format_func= lambda x : caa_no[x])

    # thall Thallium Stress Test Result
    # 0 = Null
    # 1 = Fixed defect
    # 2 = Normal
    # 3 = Reversible defect
    thall_test = ['Null', 'Fixed defect', 'Normal','Reversible defect']
    thall_option = list(range(len(thall_test)))
    thall = st.selectbox('thall: thallium stress test result', thall_option, format_func= lambda x : thall_test[x])

    
    submitted = st.form_submit_button('Submit')
    st.write(submitted)
    
    
    if submitted == True:
        
        patient_info = np.array([ages,sex,cp,trtbps,
                          chol, fbs, restecq, thalachh, 
                          exnq, oldpeak, slp, caa, thall])
        
        patient_scale = mms_scaler.transform(np.expand_dims(patient_info,axis = 0))


        result = model_logistic.predict(patient_scale)
        outcome = ohe_scaler.transform(np.expand_dims(result, axis = -1))

        
        st.write(heart_attack_chances[np.argmax(outcome)])
        
        if result == 1:
            st.warning('this patient has higher chance to have a heart attack')
        else:
            st.success('this patient has lesser chance to have a heart attack')
    
    



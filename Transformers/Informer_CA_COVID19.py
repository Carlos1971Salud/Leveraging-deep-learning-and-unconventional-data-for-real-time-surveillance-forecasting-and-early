# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 2025

In this file, Informer, a tranformer-based model, is fine-tuned for forecasting
the number of COVID-19 cases in Ontario, Canada. 
For more information visit: https://huggingface.co/docs/transformers/en/model_doc/informer

The results of the model will be saved in a folder named "transformer_results".

@author: Zahra Movahedi Nia
"""

# Importing libraries
import torch
import torch.nn as nn
from transformers import AutoformerForPrediction, InformerForPrediction
from torchmetrics.regression import R2Score
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
import math
import scipy.stats as stt
from scipy import optimize

from sklearn.decomposition import PCA
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

# Reading data
path = '../Datasets/COVID19_Canada'

cases = pd.read_csv(path + '/CA_COVID19.csv')
gt = pd.read_csv(path + '/CA_GT.csv')
aq = pd.read_csv(path + '/CA_AirQuality.csv')
reddit = pd.read_csv(path + '/CA_Reddit_COVID19.csv', low_memory=False)

# Set results directory
os.mkdir('./transformer_results')

# %%

# Setting date
start = '2020-06-01'
complete_dates = [start]
end = pd.to_datetime('2022-11-17')
while pd.to_datetime(start) < end:
    start = datetime.datetime.strptime (start, '%Y-%m-%d') + datetime.timedelta (days= 1)
    start = start.strftime ('%Y-%m-%d')
    complete_dates.append (start)
    
 # %%
 
 #----------------- Building data -----------------
nodes = {'BC':'British Columbia', 'AB':'Alberta', 'SK':'Saskatchewan', 'MB':'Manitoba', 
          'ON':'Ontario', 'QC':'Quebec', 'NB':'New Brunswick', 'NS':'Nova Scotia', 
          'NL':'Newfoundland and Labrador', 'PE':'Prince Edward Island'}
 
 # For first derivative of Google trends
def linear(x, m, b):
    return m*x + b

# For second derivative of Google trends
def quad(x, n, m, b):
    return n*x**2 + m*x + b

gt['date'] = pd.to_datetime(gt['Date'])
x = np.arange(7)

c = {k:{} for k in nodes.keys()} # COVID-19 cases
g = {k:{} for k in nodes.keys()} # Google Trends
o1 = {k:{} for k in nodes.keys()} # First derivative of GT
o2 = {k:{} for k in nodes.keys()} # Second derivative of GT
r = {k:{} for k in nodes.keys()} # Reddit
co = {k:{} for k in nodes.keys()} # Air Qualit - CO
o3 = {k:{} for k in nodes.keys()} # Air Qualit - O3
no2 = {k:{} for k in nodes.keys()} # Air Qualit - NO2
so2 = {k:{} for k in nodes.keys()} # Air Qualit - SO2

for p in nodes.keys():
  cases_dummy = cases[cases['Province'] == p]
  gt_dummy = gt[gt['Province'] == p]
  r_dummy = reddit[reddit['Province'] == p]
  co_dummy = aq[(aq['Parameter'] == 'CO') & (aq['PRENAME'] == nodes[p])]
  o3_dummy = aq[(aq['Parameter'] == 'O3') & (aq['PRENAME'] == nodes[p])]
  no2_dummy = aq[(aq['Parameter'] == 'NO2') & (aq['PRENAME'] == nodes[p])]
  so2_dummy = aq[(aq['Parameter'] == 'SO2') & (aq['PRENAME'] == nodes[p])]

  for dt in complete_dates:
    # COVID-19 cases
    dummy = cases_dummy[cases_dummy['date'] == dt]
    c[p][dt] = np.nansum(dummy['change_cases'])
    # Google Trends
    dummy = gt_dummy[gt_dummy['Date'] == dt]
    g[p][dt] = np.nansum(dummy['Search'])
    # First derivative of GT
    start = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    dummy0 = gt_dummy[(gt_dummy['date'] > pd.to_datetime(start)) & (gt_dummy['date'] <= pd.to_datetime(dt))]
    dummy0 = dummy0.fillna (np.nanmean(dummy0['Search']))
    y = list(dummy0['Search'])
    fit = optimize.curve_fit(linear, x, y)
    o1[p][dt] = fit[0][0]
    # Second derivative of GT
    fit = optimize.curve_fit(quad, x, y)
    o2[p][dt] = fit[0][0]
    # Reddit
    dummy = r_dummy[r_dummy['Date'] == dt]
    r[p][dt] = len(dummy)
    # CO
    dummy = co_dummy[co_dummy['date'] == dt]
    co[p][dt] = np.nanmax([np.nan] + list(dummy['max']))
    # O3
    dummy = o3_dummy[o3_dummy['date'] == dt]
    o3[p][dt] = np.nanmax([np.nan] + list(dummy['max']))
    # NO2
    dummy = no2_dummy[no2_dummy['date'] == dt]
    no2[p][dt] = np.nanmax([np.nan] + list(dummy['max']))
    # SO2
    dummy = so2_dummy[so2_dummy['date'] == dt]
    so2[p][dt] = np.nanmax([np.nan] + list(dummy['max']))

cases = pd.DataFrame.from_dict(c)
gt = pd.DataFrame.from_dict(g)
o1 = pd.DataFrame.from_dict(o1)
o2 = pd.DataFrame.from_dict(o2)
reddit = pd.DataFrame.from_dict(r)
co = pd.DataFrame.from_dict(co)
o3 = pd.DataFrame.from_dict(o3)
no2 = pd.DataFrame.from_dict(no2)
so2 = pd.DataFrame.from_dict(so2)

p = 'ON' # Only for Ontario
data1 = pd.DataFrame(data= {'cases':cases[p], 'gt':gt[p], 'o1':o1[p], 'o2':o2[p], 'reddit':reddit[p], 'co':co[p], 'o3':o3[p], 'no2':no2[p], 'so2':so2[p]})


#----------------- Building masks -----------------
# mask1 is for masking the NAN input values, and mask2 is for masking the NAN output values or labels
mask1 = pd.DataFrame(data= {})
for c in list(data1.columns):
  mask1[c] = [1 if pd.isnull(item) == False else 0 for item in data1[c]]
mask1['pca'] = [0 if 0 in list(mask1.iloc[i]) else 1 for i in mask1.index]
columns = list(mask1.columns)
columns.remove('pca')
mask1 = mask1.drop(columns=columns)

mask2 = pd.DataFrame(data= {})
mask2['cases'] = [1 if pd.isnull(item) == False else 0 for item in data1['cases']]

#----------------- Building time features -----------------
month1 = pd.DataFrame(data= {})
month1['month'] = [datetime.datetime.strptime(dt, '%Y-%m-%d').month for dt in complete_dates]
month1['day'] = [datetime.datetime.strptime(dt, '%Y-%m-%d').day for dt in complete_dates]

#----------------- Principal Component Analysis -----------------
data1 = data1.fillna(0)
pca = PCA(n_components=len(data1.columns))
pca1 = pca.fit_transform(np.array(data1))

data1 = np.array(data1)
mask1 = np.array(mask1)
mask2 = np.array(mask2)
month1 = np.array(month1)

#%%

# Hyperparameters:
context_length = 61 
trainL = 320 # 20 * batch_size
epoch = 20

step_ahead = 24
testL = 64
obs = len(cases)

batch_size = 64
seq_length = context_length

# %%

# Build observations
sa1 = [0, 1, 2, 3]
sa2 = [1, 2, 3, 4]
st_idx = 0 # 0 for 1-14 day(s) ahead, 1 for 15-28 days ahead, 2 for 29 to 42 days ahead, and 3 for 43 to 56 days ahead
dataset = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
mask = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
month = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
static = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
for j in range (trainL + seq_length, obs - 3*sa2[st_idx]*step_ahead, testL): # the 2* was added here to adjust for the problem in my mind
  trainx = []
  trainy = []
  testx = []
  testy = []
  maskTrainx = []
  maskTrainy = []
  maskTestx = []
  maskTesty = []
  monthTrainx = []
  monthTrainy = []
  monthTestx = []
  monthTesty = []

  for i in range (j, j + trainL):
    trainx.append (pca1[i-seq_length-trainL:i-trainL, 0]) #scaled[:,:,i-seq_length-trainL:i-trainL])
    trainy.append (data1[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, 0]) #scaled[0,:,i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead]) # feature 0 is the labels
    maskTrainx.append(mask1[i-seq_length-trainL:i-trainL, 0])
    maskTrainy.append (mask2[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, 0])
    monthTrainx.append(month1[i-seq_length-trainL:i-trainL, :])
    monthTrainy.append (month1[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, :])

  for i in range(j+trainL +sa2[st_idx]*step_ahead, j+trainL+sa2[st_idx]*step_ahead +testL):
    testx.append (pca1[i-seq_length-trainL:i-trainL, 0]) #scaled[:,:,i-seq_length-trainL:i-trainL])
    testy.append (data1[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, 0]) #scaled[0,:,i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead]) # feature 0 is the labels
    maskTestx.append (mask1[i-seq_length-trainL:i-trainL, 0])
    maskTesty.append(mask2[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, 0])
    monthTestx.append (month1[i-seq_length-trainL:i-trainL, :])
    monthTesty.append(month1[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, :])

  trainx = [torch.tensor(np.array(trainx[b:b+batch_size])).to(torch.int64) for b in range(0, trainL, batch_size)]
  trainy = [torch.tensor(np.array(trainy[b:b+batch_size])).to(torch.int64) for b in range(0, trainL, batch_size)]
  maskTrainx = [torch.tensor(np.array(maskTrainx[b:b+batch_size])).to(torch.int64) for b in range(0, trainL, batch_size)]
  maskTrainy = [torch.tensor(np.array(maskTrainy[b:b+batch_size])).to(torch.int64) for b in range(0, trainL, batch_size)]
  monthTrainx = [torch.tensor(np.array(monthTrainx[b:b+batch_size])).to(torch.int64) for b in range(0, trainL, batch_size)]
  monthTrainy = [torch.tensor(np.array(monthTrainy[b:b+batch_size])).to(torch.int64) for b in range(0, trainL, batch_size)]

  testx = [torch.tensor(np.array(testx[b:b+batch_size])).to(torch.int64) for b in range(0, testL, batch_size)]
  testy = [torch.tensor(np.array(testy[b:b+batch_size])).to(torch.int64) for b in range(0, testL, batch_size)]
  maskTestx = [torch.tensor(np.array(maskTestx[b:b+batch_size])).to(torch.int64) for b in range(0, testL, batch_size)]
  maskTesty = [torch.tensor(np.array(maskTesty[b:b+batch_size])).to(torch.int64) for b in range(0, testL, batch_size)]
  monthTestx = [torch.tensor(np.array(monthTestx[b:b+batch_size])).to(torch.int64) for b in range(0, testL, batch_size)]
  monthTesty = [torch.tensor(np.array(monthTesty[b:b+batch_size])).to(torch.int64) for b in range(0, testL, batch_size)]

  #print(len(trainx), trainx[-1].shape, len(trainy), trainy[-1].shape, len(testx), testx[-1].shape, len(testy), testy[-1].shape)

  dataset['trainx'].append (trainx)
  dataset['trainy'].append (trainy)
  dataset['testx'].append (testx)
  dataset['testy'].append (testy)
  mask['trainx'].append (maskTrainx)
  mask['trainy'].append (maskTrainy)
  mask['testx'].append (maskTestx)
  mask['testy'].append (maskTesty)
  month['trainx'].append (monthTrainx)
  month['trainy'].append (monthTrainy)
  month['testx'].append (monthTestx)
  month['testy'].append (monthTesty)

static_categorical_features = torch.zeros(batch_size, 1).to(torch.int64)
static_real_features = torch.zeros(batch_size, 1).to(torch.int64)
static_categorical_features = static_categorical_features.to (device)
static_real_features = static_real_features.to (device)

# %%

# Initialize Informer
informer = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly")
informer.model.decoder.value_embedding.value_projection = nn.Linear(23, 32, bias=False)
informer.model.encoder.value_embedding.value_projection = nn.Linear(23, 32, bias=False)

# %%

# Training and evaluating
epoch = 10000
model = 1
p = []
a = []
r2score = R2Score(multioutput='raw_values')

for i in range(len(dataset['trainx'])):
  del model
  model = copy.deepcopy(informer)

  model = model.to (device)
  min_rmse = 2000
  max_corr = 0


  # Fine-tuning
  for e in range(epoch):
      
    # Training
    model.train()
    for j in range(len(dataset['trainx'][i])):
      trainx = dataset['trainx'][i][j]
      monthx = month['trainx'][i][j]
      maskx = mask['trainx'][i][j]
      trainy = dataset['trainy'][i][j]
      monthy = month['trainy'][i][j]
      masky = mask['trainy'][i][j]

      trainx = trainx.to (device)
      monthx = monthx.to (device)
      maskx = maskx.to (device)
      trainy = trainy.to (device)
      monthy = monthy.to (device)
      masky = masky.to (device)
      model = model.to (device)
      #print(trainx.shape, monthx.shape, maskx.shape, trainy.shape, monthy.shape, masky.shape)
      outputs1 = model(past_values=trainx, past_time_features=monthx, past_observed_mask=maskx, static_categorical_features=static_categorical_features,
                      static_real_features=static_real_features, future_values=trainy, future_time_features=monthy, future_observed_mask=masky)

      loss = outputs1.loss
      loss.backward()

    # Cross-validation
    model.eval()
    j = 0
    testx = dataset['testx'][i][j]
    monthx = month['testx'][i][j]
    maskx = mask['testx'][i][j]
    testy = dataset['testy'][i][j]
    monthy = month['testy'][i][j]
    masky = mask['testy'][i][j]

    testx = testx.to (device)
    monthx = monthx.to (device)
    maskx = maskx.to (device)
    testy = testy.to (device)
    monthy = monthy.to (device)
    masky = masky.to (device)

    outputs2 = model.generate(past_values=testx, past_time_features=monthx, past_observed_mask=maskx, static_categorical_features=static_categorical_features,
                               static_real_features=static_real_features, future_time_features=monthy)

    prediction = outputs2.sequences[:,-1,:]
    actual = testy

    for j in range(1, len(dataset['testx'][i])):
      testx = dataset['testx'][i][j]
      monthx = month['testx'][i][j]
      maskx = mask['testx'][i][j]
      testy = dataset['testy'][i][j]
      monthy = month['testy'][i][j]
      masky = mask['testy'][i][j]

      testx = testx.to (device)
      monthx = monthx.to (device)
      maskx = maskx.to (device)
      testy = testy.to (device)
      monthy = monthy.to (device)
      masky = masky.to (device)

      outputs2 = model.generate(past_values=testx, past_time_features=monthx, past_observed_mask=maskx, static_categorical_features=static_categorical_features,
                                 static_real_features=static_real_features, future_time_features=monthy)

      prediction = torch.cat((prediction, outputs2.sequences[:,-1,:]), 0)
      actual = torch.cat((actual, testy), 0)
 #   print(actual.shape, prediction.shape)

    rmse = torch.sqrt(torch.mean((prediction - actual)**2))
    r2 = r2score(prediction.flatten(), actual.flatten())
    corr = torch.corrcoef(torch.cat((prediction.flatten().unsqueeze(0), actual.flatten().unsqueeze(0)), 0))

    print('epoch:', e, ', rmse= ', rmse.item(), ', r2= ', r2.item(), ', corr= ', corr[0, 1].item())

#    if rmse < min_rmse:
#      min_rmse = rmse
#      count = 0
#      torch.save(model.state_dict(), 'Model1.pth')
#    else:
#      count = count + 1
    corr = corr[0,1].item()
    if corr > max_corr:
      max_corr = corr
      count = 0
      torch.save(model.state_dict(), 'Model1.pth')

  # Evaluation
  mymodel = copy.deepcopy(informer)
  mymodel.load_state_dict(torch.load('Model1.pth'))
  mymodel = mymodel.to(device)

  j = 0
  testx = dataset['testx'][i][j]
  monthx = month['testx'][i][j]
  maskx = mask['testx'][i][j]
  testy = dataset['testy'][i][j]
  monthy = month['testy'][i][j]
  masky = mask['testy'][i][j]

  testx = testx.to (device)
  monthx = monthx.to (device)
  maskx = maskx.to (device)
  testy = testy.to (device)
  monthy = monthy.to (device)
  masky = masky.to (device)

  outputs2 = mymodel.generate(past_values=testx, past_time_features=monthx, past_observed_mask=maskx, static_categorical_features=static_categorical_features,
                            static_real_features=static_real_features, future_time_features=monthy)

  prediction = outputs2.sequences[:,-1,:]
  actual = testy

  for j in range(1, len(dataset['testx'][i])):
    testx = dataset['testx'][i][j]
    monthx = month['testx'][i][j]
    maskx = mask['testx'][i][j]
    testy = dataset['testy'][i][j]
    monthy = month['testy'][i][j]
    masky = mask['testy'][i][j]

    testx = testx.to (device)
    monthx = monthx.to (device)
    maskx = maskx.to (device)
    testy = testy.to (device)
    monthy = monthy.to (device)
    masky = masky.to (device)

    outputs2 = mymodel.generate(past_values=testx, past_time_features=monthx, past_observed_mask=maskx, static_categorical_features=static_categorical_features,
                               static_real_features=static_real_features, future_time_features=monthy)

    prediction = torch.cat((prediction, outputs2.sequences[:,-1,:]), 0)
    actual = torch.cat((actual, testy), 0)

  p.append(prediction)
  a.append(actual)

actual = []
pred = []
for i in range(len(a)):
  dummy = pd.DataFrame(a[i].cpu().numpy(), index=list(range(64)), columns=list(range(1, 25)))
  actual.append(dummy)
  dummy = pd.DataFrame(p[i].cpu().numpy(), index=list(range(64)), columns=list(range(1, 25)))
  pred.append(dummy)
  
actual = pd.concat(actual)
pred = pd.concat(pred)
  
actual.to_csv('./transformer_results/actual_Informer.csv',index=False)
pred.to_csv('./transformer_results/predicted_Informer.csv',index=False)




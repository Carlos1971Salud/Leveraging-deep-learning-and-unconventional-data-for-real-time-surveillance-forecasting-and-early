# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 2025

In this file, Autoformer, a tranformer-based model, is fine-tuned for forecasting
the number of COVID-19 cases in South Africa. 
For more information visit: https://huggingface.co/docs/transformers/en/model_doc/autoformer

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
path = '../Datasets/COVID-19_Africa'

cases = pd.read_csv(path + '/Africa_COVID19.csv')
gt = pd.read_csv(path + '/Africa_GT.csv')
wiki = pd.read_csv(path + '/wiki_africa.csv')
co = pd.read_csv(path + '/Air Quality/co_africa.csv')
no2 = pd.read_csv(path + '/Air Quality/no2_africa.csv')
so2 = pd.read_csv(path + '/Air Quality/so2_africa.csv')
o3 = pd.read_csv(path + '/Air Quality/o3_africa.csv')
uv = pd.read_csv(path + '/Air Quality/uv_africa.csv')
#gn = pd.read_csv('/africa_covid_googlenews.csv')

# Set results directory
os.mkdir('./transformer_results')

# %%

# Setting date
start = '2020-06-01'
complete_dates = [start]
end = pd.to_datetime('2022-07-31')
while pd.to_datetime(start) < end:
    start = datetime.datetime.strptime (start, '%Y-%m-%d') + datetime.timedelta (days= 1)
    start = start.strftime('%Y-%m-%d')
    complete_dates.append (start)
    
# %%

#-------------------- Building data ----------------------
nodes = {'ZAF': 'South Africa', 'LSO':'Lesotho', 'SWZ':'Swaziland', 'BWA':'Botswana', 
         'NAM':'Namibia', 'ZWE':'Zimbabwe', 'MOZ':'Mozambique', 'AGO':'Angola',
         'ZMB':'Zambia', 'MWI':'Malawi', 'TZA':'Tanzania', 'BDI':'Burundi', 'RWA':'Rwanda',
         'KEN':'Kenya', 'UGA':'Uganda', 'COD': 'DR Congo'}

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
wk = {k:{} for k in nodes.keys()} # Wiki Trends
#g2 = {k:{} for k in nodes.keys()} # Google News
aq_co = {k:{} for k in nodes.keys()} # Air Quality - CO
aq_o3 = {k:{} for k in nodes.keys()} # Air Quality - O3
aq_so2 = {k:{} for k in nodes.keys()} # Air Qualit - CO
aq_no2 = {k:{} for k in nodes.keys()} # Air Qualit - O3
aq_uv = {k:{} for k in nodes.keys()} # Air Qulity - UV-index

for p in nodes.keys():
  cases_dummy = cases[cases['Country'] == p]
  gt_dummy = gt[gt['Country'] == p]
  wiki_dummy = wiki[wiki['ISO'] == p]
#  gn_dummy = gn[gn['Country'] == p]
  co_dummy = co[co['ISO'] == p]
  no2_dummy = no2[no2['ISO'] == p]
  so2_dummy = so2[so2['ISO'] == p]
  o3_dummy = o3[o3['ISO'] == p]
  uv_dummy = uv[uv['ISO'] == p]

  for dt in complete_dates:
    # COVID-19 cases
    dummy = cases_dummy[cases_dummy['Date'] == dt]
    c[p][dt] = np.nansum(dummy['Cases'])
    # Google Trends
    dummy = gt_dummy[gt_dummy['Date'] == dt]
    g[p][dt] = np.nansum(dummy['Searches'])
    # First derivative of GT
    start = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    dummy0 = gt_dummy[(gt_dummy['date'] > pd.to_datetime(start)) & (gt_dummy['date'] <= pd.to_datetime(dt))]
    dummy0 = dummy0.fillna (np.nanmean(dummy0['Searches']))
    y = list(dummy0['Searches'])
    if len(y) == 0:
      y = [0.01]*7
    fit = optimize.curve_fit(linear, x, y)
    o1[p][dt] = fit[0][0]
    # Second derivative of GT
    fit = optimize.curve_fit(quad, x, y)
    o2[p][dt] = fit[0][0]
    # Wiki Trends
    dummy = wiki_dummy[wiki_dummy['date'] == dt]
    wk[p][dt] = np.nanmin([np.nan] + list(dummy['wiki_views']))
    # Google News
#    dummy = gn_dummy[gn_dummy['date'] == dt]
#    g2[p][dt] = len(dummy)
    # CO
    dummy = co_dummy[co_dummy['date'] == dt]
    aq_co[p][dt] = np.nanmax([np.nan] + list(dummy['max']))
    # O3
    dummy = o3_dummy[o3_dummy['date'] == dt]
    aq_o3[p][dt] = np.nanmax([np.nan] + list(dummy['max']))
    # SO2
    dummy = so2_dummy[so2_dummy['date'] == dt]
    aq_so2[p][dt] = np.nanmax([np.nan] + list(dummy['max']))
    # NO2
    dummy = no2_dummy[no2_dummy['date'] == dt]
    aq_no2[p][dt] = np.nanmax([np.nan] + list(dummy['max']))
    # UV
    dummy = uv_dummy[uv_dummy['date'] == dt]
    aq_uv[p][dt] = np.nanmax([np.nan] + list(dummy['max']))


cases = pd.DataFrame.from_dict(c)
gt = pd.DataFrame.from_dict(g)
o1 = pd.DataFrame.from_dict(o1)
o2 = pd.DataFrame.from_dict(o2)
wiki = pd.DataFrame.from_dict(wk)
#gn = pd.DataFrame.from_dict(g2)
co = pd.DataFrame.from_dict(aq_co)
o3 = pd.DataFrame.from_dict(aq_o3)
no2 = pd.DataFrame.from_dict(aq_no2)
so2 = pd.DataFrame.from_dict(aq_so2)
uv = pd.DataFrame.from_dict(aq_uv)

p = 'ZAF' # Only for South Africa
data1 = pd.DataFrame(data={'cases':cases[p], 'gt':gt[p], 'o1':o1[p], 'o2':o2[p], 'wiki':wiki[p], #'gn':gn[p],
                           'co':co[p], 'o3':o3[p], 'no2':no2[p], 'so2':so2[p], 'uv':uv[p]})


#------------------- Building Masks -------------------
# mask1 is for masking the NAN input values, and mask2 is for masking the NAN output values
mask1 = pd.DataFrame(data= {})
for c in list(data1.columns):
  mask1[c] = [1 if pd.isnull(item) == False else 0 for item in data1[c]]
mask1['pca'] = [0 if 0 in list(mask1.iloc[i]) else 1 for i in mask1.index]
columns = list(mask1.columns)
columns.remove('pca')
mask1 = mask1.drop(columns=columns)

mask2 = pd.DataFrame(data= {})
mask2['cases'] = [1 if pd.isnull(item) == False else 0 for item in data1['cases']]

#------------------ Building time features --------------------
month1 = pd.DataFrame(data= {})
month1['month'] = [datetime.datetime.strptime(dt, '%Y-%m-%d').month for dt in complete_dates]
month1['day'] = [datetime.datetime.strptime(dt, '%Y-%m-%d').day for dt in complete_dates]

#------------------ Pricnipal Component Analysis ------------------
data1 = data1.fillna(0)
pca = PCA(n_components=len(data1.columns))
pca1 = pca.fit_transform(np.array(data1))

data1 = np.array(data1)
mask1 = np.array(mask1)
mask2 = np.array(mask2)
month1 = np.array(month1)

# %%

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

# Building observations
sa1 = [0, 1, 2, 3]
sa2 = [1, 2, 3, 4]
st_idx = 0 
dataset = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
mask = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
month = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
static = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
for j in range (trainL + seq_length, obs - 4*sa2[st_idx]*step_ahead, testL): 
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
    trainx.append (pca1[i-seq_length-trainL:i-trainL, 0]) 
    trainy.append (data1[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, 0]) # feature 0 is historical data used for building labels
    maskTrainx.append(mask1[i-seq_length-trainL:i-trainL, 0])
    maskTrainy.append (mask2[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, 0])
    monthTrainx.append(month1[i-seq_length-trainL:i-trainL, :])
    monthTrainy.append (month1[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, :])

  for i in range(j+trainL +sa2[st_idx]*step_ahead, j+trainL+sa2[st_idx]*step_ahead +testL):
    testx.append (pca1[i-seq_length-trainL:i-trainL, 0]) 
    testy.append (data1[i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead, 0]) # feature 0 is historical data used for building labels
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

# Initialize Autoformer
autoformer = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

# %%

# Training and evaluating
epoch = 10000
model = 1
p = []
a = []
r2score = R2Score(multioutput='raw_values')

for i in range(len(dataset['trainx'])):
  del model
  model = copy.deepcopy(autoformer)

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
                      future_values=trainy, future_time_features=monthy, future_observed_mask=masky)

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
                               future_time_features=monthy) #,static_real_features=static_real_features,

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
                                 future_time_features=monthy) #,static_real_features=static_real_features,

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

  mymodel = copy.deepcopy(autoformer)

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
                            future_time_features=monthy) #,static_real_features=static_real_features,

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
                               future_time_features=monthy) #,static_real_features=static_real_features,

    prediction = torch.cat((prediction, outputs2.sequences[:,-1,:]), 0)
    actual = torch.cat((actual, testy), 0)

  p.append(prediction)
  a.append(actual)
#  break

actual = []
pred = []
for i in range(len(a)):
  dummy = pd.DataFrame(a[i].cpu().numpy(), index=list(range(64)), columns=list(range(1, 25)))
  actual.append(dummy)
  dummy = pd.DataFrame(p[i].cpu().numpy(), index=list(range(64)), columns=list(range(1, 25)))
  pred.append(dummy)
  
actual = pd.concat(actual)
pred = pd.concat(pred)

actual.to_csv('./transformer_results/actual_Autoformer.csv',index=False)
pred.to_csv('./transformer_results/predicted_Autoformer.csv',index=False)
  

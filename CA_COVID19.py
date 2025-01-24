# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 2025

In this file a Graph Neural Networks (GNN)-based model is designed and trained to  
forecast the number of COVID-19 cases in different provinces of Canada:
    British Columbia, Alberta, Saskatchewan, Manitoba, Ontario, Quebec, 
    New Brunswick, Newfoundland and Labrador, Nova Scotia, and 
    Prince Edward Island.
The final model which is implemented in Pytorch includes 4 Neural Network Layers: 
    A Convolutional Neural Network (CNN), a Graph Convolutional Network (GCN), 
    a Gated Recurrent Unit (GRU), and a stacked linear layer. 
Multiple data sources including Historical number of cases, Google trends, 
Reddit, and Air Quality are combined to build a multi-variate sequential 
time-series predictoin model.

The results of the model will be saved in a folder named "results".

@author: Zahra Movahedi Nia
"""

# Importing libraries
import torch
import torch.optim as optim
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from torch_geometric.nn import GraphConv, GCNConv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
import math
import scipy.stats as stt
from scipy import optimize

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

# Reading Data 
path = './Datasets/COVID19_Canada'

cases = pd.read_csv(path + '/CA_COVID19.csv')
gt = pd.read_csv(path + '/CA_GT.csv')
aq = pd.read_csv(path + '/CA_AirQuality.csv')
reddit = pd.read_csv(path + '/CA_Reddit_COVID19.csv', low_memory=False)

# Set results directory
os.mkdir('./results')

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

# Building data
#nodes = {'BC':'British Columbia', 'AB':'Alberta', 'SK':'Saskatchewan', 'MB':'Manitoba', 
#         'ON':'Ontario', 'QC':'Quebec', 'NB':'New Brunswick', 'NS':'Nova Scotia', 
#         'NL':'Newfoundland and Labrador', 'PE':'Prince Edward Island'}
nodes = {'AB':'Alberta', 'BC':'British Columbia', 'SK':'Saskatchewan', 'QC':'Quebec', 
         'MB':'Manitoba', 'ON':'Ontario', 'NB':'New Brunswick', 
         'NL':'Newfoundland and Labrador', 'NS':'Nova Scotia', 'NT':'Northwest Territories', 
         'NU':'Nunavut', 'YT':'Yukon', 'PE':'Prince Edward Island'}

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

for p in nodes.keys():
  cases[p] = cases[p].fillna(np.nanmean(cases[p]))
  gt[p] = gt[p].fillna(np.nanmean(gt[p]))
  reddit[p] = reddit[p].fillna(np.nanmean(reddit[p]))
  co[p] = co[p].fillna(np.nanmean(co[p]))
  o3[p] = o3[p].fillna(np.nanmean(o3[p]))
  no2[p] = no2[p].fillna(np.nanmean(no2[p]))
  so2[p] = so2[p].fillna(np.nanmean(so2[p]))
  
# Correlations
print('Correlations')
for p in nodes.keys():
  print(nodes[p])
  print('Google Trends: ',stt.pearsonr(cases[p],gt[p])[0], ', p=',stt.pearsonr(cases[p],gt[p])[1])
  print('1st Derivative: ',stt.pearsonr(cases[p],o1[p])[0], ', p=',stt.pearsonr(cases[p],o1[p])[1])
  print('2nd Derivative: ',stt.pearsonr(cases[p],o2[p])[0], ', p=',stt.pearsonr(cases[p],o2[p])[1])
  print('Reddit: ',stt.pearsonr(cases[p],reddit[p])[0], ', p=',stt.pearsonr(cases[p],reddit[p])[1])
  print('CO: ',stt.pearsonr(cases[p],co[p])[0], ', p=',stt.pearsonr(cases[p],co[p])[1])
  print('O3: ',stt.pearsonr(cases[p],o3[p])[0], ', p=',stt.pearsonr(cases[p],o3[p])[1])
  print('NO2: ',stt.pearsonr(cases[p],no2[p])[0], ', p=',stt.pearsonr(cases[p],no2[p])[1])
  print('SO2: ',stt.pearsonr(cases[p],so2[p])[0], ', p=',stt.pearsonr(cases[p],so2[p])[1])
  
# %%
# Unifying the datasets and scaling
data1 = np.asarray([np.asarray(cases).transpose(), np.asarray(gt).transpose(), np.asarray(o1).transpose(),
                   np.asarray(o2).transpose(), np.asarray(reddit).transpose(), np.asarray(no2).transpose(),
                   np.asarray(so2).transpose(), np.asarray(co).transpose(), np.asarray(o3).transpose()])
# index 0 is the historical data for labels
n_features = data1.shape[0]
n_nodes = data1.shape[1]
obs = data1.shape[2] # Observations

# Scaling
mmscaler = MinMaxScaler()
scaled = mmscaler.fit_transform (data1.reshape(n_nodes*n_features, obs)).reshape(n_features, n_nodes, obs)

# %%

# Building Graph
# node_index = {0: 'BC', 1: 'AB', 2: 'SK', 3: 'MB', 4: 'ON', 5: 'QC', 6: 'NB', 7: 'NS', 8: 'NL', 9: 'PE'}
#edge_index = [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9],
#              [0, 1, 1, 0, 2, 2, 1, 3, 3, 2, 4, 4, 3, 5, 5, 4, 6, 8, 6, 5, 7, 9, 7, 6, 9, 8, 5, 9, 6, 7]]
#edge_index = np.asarray (edge_index)
edge_index = [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 7, 8, 8, 9, 9, 9, 9, 9, 9,10,10,10,11,11,11,12,12,  0,1,2,3,4,5,6,7,8,9,10,11,12],
              [1, 9, 2, 0, 9,11, 0, 9, 4, 5, 7, 6, 2, 9,10, 5, 4, 3, 3,12, 8, 3,12, 6, 0, 1, 4, 2,10,11, 4, 9,11,10, 1, 9, 6, 8,  0,1,2,3,4,5,6,7,8,9,10,11,12]]
edge_index = np.asarray (edge_index)

corr = cases.corr()
edge_weight = []
for i in range(edge_index.shape[1]):
    edge_weight.append (corr[corr.columns[edge_index[0, i]]][corr.index[edge_index[1, i]]])

# %%

# Building the model
class temporalGNN (torch.nn.Module):
  def __init__ (self, n_features, seq, hidden, n_hidden, n_layers, dropout1, dropout2, dropout3, step_ahead, activation): # out_channels or hidden
    super (temporalGNN, self).__init__()
    self.cnn = torch.nn.Conv2d (in_channels= n_features, out_channels= 1, kernel_size= (1,1)) # input: [n_features, n_nodes, seq] , output: [1, n_nodes, seq]
    self.dropout1 = torch.nn.Dropout (p= dropout1)
    self.gcn = GCNConv (in_channels= seq, out_channels= hidden) # input: [n_nodes, seq] , output: [n_nodes, hidden]
    self.dropout2 = torch.nn.Dropout (p= dropout2)
    if activation == 'relu':
      self.activation = torch.nn.ReLU()
    elif activation == 'tanh':
      self.activation = torch.nn.Tanh()
    else:
      self.activation = torch.nn.Dropout (p= 0) # We have no activation function otherwise.

    self.gru = torch.nn.GRU (input_size= n_nodes, hidden_size= n_nodes, batch_first=True, num_layers=n_layers, dropout= dropout3) # input: [hidden, n_nodes], output[hidden, n_nodes]
    self.linear = torch.nn.Linear (hidden, step_ahead) # input: [n_nodes, hidden] , output: [n_nodes, step_ahead]

  def forward (self, x, edge_index, edge_weight):
    out = self.cnn (x)
    out = self.dropout1 (out.squeeze())

    out = self.gcn (out, edge_index, edge_weight)
    out = self.activation (out)
    out = self.dropout2 (out)
    s = out.shape
 #   print (s)
    out = out.transpose(0, 1)
    out,_ = self.gru(out) 
    out = out.transpose(0, 1)
    out = self.linear (out)
    return out

# %%

# Hyperparameters
trainL = 300
seq_length = 15
testL = 20
step_ahead = 14
hidden = 64
n_hidden = 64
n_layers = 5
dropout1 = 0.2
dropout2 = 0.4
dropout3 = 0.2
activation = 'relu'
opt = 'adam'
lr = 0.0001

loss_criterion = torch.nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
order = list(nodes.keys())

# %%

# Creating the observations (the features and the labels)
sa1 = [0, 1, 2, 3]
sa2 = [1, 2, 3, 4]
st_idx = 0 # 0 for 1-14 day(s) ahead, 1 for 15-28 days ahead, 2 for 29 to 42 days ahead, and 3 for 43 to 56 days ahead

dataset = {'trainx':[], 'trainy':[], 'testx':[], 'testy':[]}
for j in range (trainL + seq_length, obs - 2*sa2[st_idx]*step_ahead, testL): # the 2* was added here to adjust for the problem in my mind
  trainx = []
  trainy = []
  testx = []
  testy = []

  for i in range (j, j + trainL):
    trainx.append (scaled[:,:,i-seq_length-trainL:i-trainL])
    trainy.append (scaled[0,:,i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead]) # feature 0 is the historical data which is used to build the labels

  for i in range(j+trainL +sa2[st_idx]*step_ahead, j+trainL+sa2[st_idx]*step_ahead +testL):
    testx.append (scaled[:,:,i-seq_length-trainL:i-trainL])
    testy.append (scaled[0,:,i-trainL+sa1[st_idx]*step_ahead:i-trainL+sa2[st_idx]*step_ahead]) # feature 0 is the historical data which is used to build the labels

  dataset['trainx'].append (trainx)
  dataset['trainy'].append (trainy)
  dataset['testx'].append (testx)
  dataset['testy'].append (testy)

#
train = []
test = []
for i in range (len(dataset['trainx'])):
  train.append (StaticGraphTemporalSignal (edge_index= edge_index, edge_weight= edge_weight, features= dataset['trainx'][i], targets= dataset['trainy'][i]))
  test.append (StaticGraphTemporalSignal (edge_index= edge_index, edge_weight= edge_weight, features= dataset['testx'][i], targets= dataset['testy'][i]))

# %%

# Training and Evaluating
labels = {}
predictions = {}
for item in order:
  labels[item] = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
  predictions[item] = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}

for i in tqdm (range(len(train))): 
  LOSS = 5
  checkpoint = 0
  model = temporalGNN (n_features= n_features, seq= seq_length, hidden= hidden, n_hidden= n_hidden, n_layers= n_layers, dropout1= dropout1, dropout2= dropout2, dropout3= dropout3, step_ahead= step_ahead, activation= activation).to(device)

  if opt == 'adam':
    optimizer = torch.optim.Adam (model.parameters(), lr= lr)
  elif opt == 'adamW':
    optimizer = torch.optim.AdamW (model.parameters(), lr= lr)

  for epoch in range(4000): # ----------------------- EPOCHS ---------------------------
    loss= 0
    step= 0
    train_loss = []
    model.train ()
    for data in train[i]:
      data = data.to(device)
      pred = model (data.x, data.edge_index, data.edge_weight)

      s = data.y.shape[1]
      loss = torch.mean ((pred[:,:s] - data.y) ** 2) 
      train_loss.append (loss.item())

      loss.backward ()
      optimizer.step ()
      optimizer.zero_grad ()

    tloss = np.mean(train_loss)
    # print ('train loss= ', tloss)
    #wandb.log ({"train_loss": tloss})

    # -------------- Validation ---------------
    loss = 0
    step = 0
    test_loss = []
    model.eval()
    for data in test[i]:
      data = data.to(device)
      pred = model (data.x, data.edge_index, data.edge_weight)

      s = data.y.shape[1]
      loss = torch.mean ((pred[:,:s] - data.y) ** 2) 
      test_loss.append (loss.item())
      step += 1

    loss = np.mean(test_loss)
    # print ('test loss= ', loss)
    if loss < LOSS:
      LOSS = loss
      torch.save(model.state_dict(), 'Model1.pth')
      checkpint = 0
    else:
      checkpoint += 1

    RMSE = math.sqrt(LOSS)

  #  wandb.log ({"val_loss": LOSS, 'RMSE':RMSE}) #, "R2": self.r2, 'MAPE':self.mape, 'RMSE':rmse})
    if checkpoint >= 100: # ----------------------------------------------- Patience ---------------------------------------
      break

  mymodel = temporalGNN (n_features= n_features, seq= seq_length, hidden= hidden, n_hidden= n_hidden, n_layers= n_layers, dropout1= dropout1, dropout2= dropout2, dropout3= dropout3, step_ahead= step_ahead, activation= activation).to(device)
  mymodel.load_state_dict(torch.load('Model1.pth'))
  mymodel = mymodel.to(device)

  p = []
  a = []
  for data in test[i]:
    data = data.to(device)
    pred = model (data.x, data.edge_index, data.edge_weight)

    s = data.y.shape[1]
    a.append (data.y)
    p.append (pred[:,:s])

  for country in range(len(order)):
    for ts in range(1, step_ahead+1):
      dummy1 = [pred[country][ts-1].item() if ts <= pred.shape[1] else 0 for pred in p]
      dummy2 = [label[country][ts-1].item() if ts <= label.shape[1] else 0 for label in a ]
      predictions[order[country]][ts] = predictions[order[country]][ts] + dummy1
      labels[order[country]][ts] = labels[order[country]][ts] + dummy2

    result3 = pd.DataFrame (data= {'Actual_1':labels[order[country]][1], 'Predicted_1':predictions[order[country]][1], 'Actual_2':labels[order[country]][2], 'Predicted_2':predictions[order[country]][2], 'Actual_3':labels[order[country]][3], 'Predicted_3':predictions[order[country]][3], 'Actual_4':labels[order[country]][4], 'Predicted_4':predictions[order[country]][4],
                              'Actual_5':labels[order[country]][5], 'Predicted_5':predictions[order[country]][5], 'Actual_6':labels[order[country]][6], 'Predicted_6':predictions[order[country]][6], 'Actual_7':labels[order[country]][7], 'Predicted_7':predictions[order[country]][7], 'Actual_8':labels[order[country]][8], 'Predicted_8':predictions[order[country]][8],
                              'Actual_9':labels[order[country]][9], 'Predicted_9':predictions[order[country]][9], 'Actual_10':labels[order[country]][10], 'Predicted_10':predictions[order[country]][10], 'Actual_11':labels[order[country]][11], 'Predicted_11':predictions[order[country]][11], 'Actual_12':labels[order[country]][12], 'Predicted_12':predictions[order[country]][12],
                              'Actual_13':labels[order[country]][13], 'Predicted_13':predictions[order[country]][13], 'Actual_14':labels[order[country]][14], 'Predicted_14':predictions[order[country]][14]})
    result3.to_csv ('./results/' + order[country] + '.csv', index= False)

os.remove('Model1.pth')

# %%

# Unscale the results
dummy = pd.read_csv ('./results/AB.csv')
zr = 0
for i in range(len(dummy)-1, -1, -1):
  if dummy['Predicted_14'][i] == 0:
    zr += 1
  else:
    break

preds = {}
files = {}
for i in range(len(order)):
  preds[order[i]] = {'actual':[],'Predicted_1':[],'Predicted_2':[],'Predicted_3':[],'Predicted_4':[],'Predicted_5':[],'Predicted_6':[],'Predicted_7':[],'Predicted_8':[],'Predicted_9':[],'Predicted_10':[],'Predicted_11':[],'Predicted_12':[],'Predicted_13':[],'Predicted_14':[]}
  dummy = pd.read_csv ('./results/' + order[i] + '.csv')
  files[order[i]] = dummy
  a = list(scaled[0,i, :])
  q = 13
  for k in preds[order[i]].keys():
    if k == 'actual':
      preds[order[i]]['actual'] = a +[0]*zr
    else:
      p = list(dummy[k]) +[0]*q
      z = len(preds[order[i]]['actual']) - len(p)
      preds[order[i]][k] = [0]*z + list(dummy[k]) +[0]*q
      q = q-1
  
for k in preds.keys():
  preds[k] = pd.DataFrame(data= preds[k])
  
preds2 = {}
for i in range(len(order)):
  preds2[order[i]] = {'actual':[],'Predicted_1':[],'Predicted_2':[],'Predicted_3':[],'Predicted_4':[],'Predicted_5':[],'Predicted_6':[],'Predicted_7':[],'Predicted_8':[],'Predicted_9':[],'Predicted_10':[],'Predicted_11':[],'Predicted_12':[],'Predicted_13':[],'Predicted_14':[]}
  preds2[order[i]]['actual'] = list(data1[0,i,:])
  dummy = scaled.copy()
  for sp in range(14):
    col = 'Predicted_'+str(sp+1)
    for j in range(len(preds[order[i]][col])-zr):
      dummy[0,i,j] = preds[order[i]][col][j]
    reverse = mmscaler.inverse_transform(dummy.reshape(n_nodes*n_features, obs)).reshape(n_features, n_nodes, obs)
    preds2[order[i]][col] = list(reverse[0,i,:])
    
for k in preds2.keys():
  preds2[k] = pd.DataFrame(data= preds2[k])
  
# %%

# Plotting
province = 'ON'
days_ahead = 14

def RMSE (a, p):
  r = []
  for i in range(len(a)):
    r.append( (a[i]-p[i])**2 )
  r = np.nanmean(r)
  r = math.sqrt(r)
  return r

def moving_average(a, n=7) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.rcParams.update({'font.size': 20})

f = plt.figure(figsize=(9, 7), dpi=80)
ax1 = f.add_subplot(111)
ax1.grid(linestyle='-', linewidth=0.2, alpha=0.8, color='k')

r2 = r2_score(np.abs(list(preds2[province]['actual'])[371:]), np.abs(list(preds2[province]['Predicted_'+str(days_ahead)])[371:]))
rmse = RMSE (np.abs(list(preds2[province]['actual'])[371:]), np.abs(list(preds2[province]['Predicted_'+str(days_ahead)])[371:]))
corr = stt.pearsonr (np.abs(list(preds2[province]['actual'])[371:]), np.abs(list(preds2[province]['Predicted_'+str(days_ahead)])[371:]))
p = corr[1]
corr = corr[0]
plt.title (nodes[province]+', '+str(sa2[st_idx]*days_ahead)+' days-ahead, RMSE= '+'{:.4f}'.format(rmse)+'\nR2= '+'{:.4f}'.format(r2)+', Corr= '+'{:.4f}'.format(corr)+' (p< .000001)', fontsize= 20)

ax1.plot (np.abs(list(preds2[province]['actual'])[315:]), 'b-')
ax1.plot (np.abs([np.nan]*56 + list(preds2[province]['Predicted_'+str(days_ahead)])[371:]), 'r-')

t = [complete_dates[i+315] for i in range (0, len(complete_dates)-315) if i % 100 == 0]
t = ['0'] + t + ['2022-12-03']
ax1.set_xticklabels(t, fontsize= 15, rotation=55)

ax1.set_ylabel ('COVID-19 Cases', {'fontsize': 19}, color='b')
ax1.legend (['Actual', 'Predicted'], loc='upper left', fontsize= 17)
plt.show()

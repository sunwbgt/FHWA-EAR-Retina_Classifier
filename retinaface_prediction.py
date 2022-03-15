import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import ConcatDataset, TensorDataset, Dataset, DataLoader, random_split 
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix

import os
os.chdir("U:\Research Projects\FHWA-EAR\RetinaFace")

# read in data
list_of_EventTable = ['eventID', 'anonymousParticipantID', 'eventStart', 
                      'secondaryTask1', 'secondaryTask1StartTime', 'secondaryTask1EndTime', 
                      'secondaryTask2', 'secondaryTask2StartTime', 'secondaryTask2EndTime',
                      'secondaryTask3', 'secondaryTask3StartTime', 'secondaryTask3EndTime']
orig_data = pd.read_excel('EventTableFull.xlsx', usecols = list_of_EventTable)

# only work on cell phone use
mask_bool1 = []
mask_bool2 = []
mask_bool3 = []
for i in orig_data.secondaryTask1:
    mask_bool1.append('Cell' in i)
for i in orig_data.secondaryTask2:
    mask_bool2.append('Cell' not in i)
for i in orig_data.secondaryTask3:
    mask_bool3.append('Cell' not in i)
mask_bool1 = pd.array(mask_bool1, dtype = 'boolean')
mask_bool2 = mask_bool2
mask_bool3 = mask_bool3

orig_data.loc[np.arange(0, len(mask_bool2))[mask_bool2], 'secondaryTask2StartTime'] = 0
orig_data.loc[np.arange(0, len(mask_bool2))[mask_bool2], 'secondaryTask2EndTime'] = 0
orig_data.loc[np.arange(0, len(mask_bool3))[mask_bool3], 'secondaryTask3StartTime'] = 0
orig_data.loc[np.arange(0, len(mask_bool3))[mask_bool3], 'secondaryTask3EndTime'] = 0

list_of_EventTable.remove('secondaryTask1')
list_of_EventTable.remove('secondaryTask2')
list_of_EventTable.remove('secondaryTask3')

orig_data_cell = orig_data.loc[mask_bool1, list_of_EventTable].to_numpy()
orig_data = orig_data.loc[:, list_of_EventTable].to_numpy()
#% extract normal and abnormal events from the whole folder
os.chdir("U:\Research Projects\FHWA-EAR\RetinaFace\RetinaFaceData")
list_of_filenames = os.listdir("./") 

#% extract distracted events
id_list = list(orig_data_cell[:, 0])
pitch_abnormal = []
yaw_abnormal = []
roll_abnormal = []
normalmax = 90
stepnum = 0
for i in list_of_filenames: # index from retina face folder
    stepnum += 1    
    num = int(i.split('_')[3]) # detach the index id
    try:
        rowID = list(id_list).index(num) # match the eventID in EventTable.csv
        # read first secondary task
        if orig_data_cell[rowID, 3] > 0:
            starttime = orig_data_cell[rowID, 3]
            endtime = orig_data_cell[rowID, 4]    
            temp_data = pd.read_csv(i, delimiter = ',', usecols = ['Pitch', 'Roll', 'Yaw', 'frameTime'])
            temp_data = temp_data.to_numpy()
            distractrange = np.arange(temp_data.shape[0])[(temp_data[:, 3] > starttime) & (temp_data[:, 3] < endtime)]
            if len(distractrange) > normalmax:
                pitch_abnormal.append(temp_data[distractrange, 0])
                roll_abnormal.append(temp_data[distractrange, 1])
                yaw_abnormal.append(temp_data[distractrange, 2])
            # read second secondary task
            if orig_data_cell[rowID, 5] > 0:
                starttime = orig_data_cell[rowID, 5]
                endtime = orig_data_cell[rowID, 6]    
                temp_data = pd.read_csv(i, delimiter = ',', usecols = ['Pitch', 'Roll', 'Yaw', 'frameTime'])
                temp_data = temp_data.to_numpy()
                distractrange = np.arange(temp_data.shape[0])[(temp_data[:, 3] > starttime) & (temp_data[:, 3] < endtime)]
                if len(distractrange) > normalmax:
                    pitch_abnormal.append(temp_data[distractrange, 0])
                    roll_abnormal.append(temp_data[distractrange, 1])
                    yaw_abnormal.append(temp_data[distractrange, 2])
                # read third secondary task
                if orig_data_cell[rowID, 7] > 0:
                    starttime = orig_data_cell[rowID, 7]
                    endtime = orig_data_cell[rowID, 8]    
                    temp_data = pd.read_csv(i, delimiter = ',', usecols = ['Pitch', 'Roll', 'Yaw', 'frameTime'])
                    temp_data = temp_data.to_numpy()
                    distractrange = np.arange(temp_data.shape[0])[(temp_data[:, 3] > starttime) & (temp_data[:, 3] < endtime)]
                    if len(distractrange) > normalmax:
                        pitch_abnormal.append(temp_data[distractrange, 0])
                        roll_abnormal.append(temp_data[distractrange, 1])
                        yaw_abnormal.append(temp_data[distractrange, 2])      
    except ValueError:
        flag = 0

# also extract normal events
id_list = list(orig_data[:, 0])
pitch_normal = []
yaw_normal = []
roll_normal = []
normalcount = 0
normalrange = np.arange(0, normalmax)
for i in list_of_filenames: # index from retina face folder
    stepnum += 1    
    num = int(i.split('_')[3]) # detach the index id
    try:
        rowID = list(id_list).index(num) # match the eventID in EventTable.csv
        # read first secondary task
        if orig_data[rowID, 3] == 0:
            temp_data = pd.read_csv(i, delimiter = ',', usecols = ['Pitch', 'Roll', 'Yaw', 'frameTime'])
            temp_data = temp_data.to_numpy()
            if temp_data.shape[0] > normalmax:
                normalcount += 1
                pitch_normal.append(temp_data[normalrange, 0])
                roll_normal.append(temp_data[normalrange, 1])
                yaw_normal.append(temp_data[normalrange, 2])              
    except ValueError:
        flag = 0
        
#% some descriptive plots
len_abnormal = np.zeros(len(pitch_abnormal))
for i in range(len(pitch_abnormal)):
    len_abnormal[i] = len(pitch_abnormal[i])

plt.hist(len_abnormal, bins = np.arange(0, 150, 5))

#%% generate sliding windows
step = normalmax
def sliding_window(datas, steps = 2, width = step):
    win_set=[]
    for event in range(len(datas)):
        for i in np.arange(0, len(datas[event]), steps):
            temp = datas[event][i : i + width]
            if temp.shape[0] == width:
                win_set.append(temp)
    return win_set

pitch_abnormal_window = np.array(sliding_window(pitch_abnormal))
roll_abnormal_window = np.array(sliding_window(roll_abnormal))
yaw_abnormal_window = np.array(sliding_window(yaw_abnormal))
pitch_normal_window = np.array(sliding_window(pitch_normal))
roll_normal_window = np.array(sliding_window(roll_normal))
yaw_normal_window = np.array(sliding_window(yaw_normal))

data_normal_window = np.zeros((pitch_normal_window.shape[0], pitch_normal_window.shape[1], 3))
data_normal_window[:, :, 0] = pitch_normal_window
data_normal_window[:, :, 1] = roll_normal_window
data_normal_window[:, :, 2] = yaw_normal_window

data_abnormal_window = np.zeros((pitch_abnormal_window.shape[0], pitch_abnormal_window.shape[1], 3))
data_abnormal_window[:, :, 0] = pitch_abnormal_window
data_abnormal_window[:, :, 1] = roll_abnormal_window
data_abnormal_window[:, :, 2] = yaw_abnormal_window

data_full_window = np.concatenate([data_normal_window, data_abnormal_window], axis = 0)
label_full = np.concatenate((np.zeros(data_normal_window.shape[0]), np.ones(data_abnormal_window.shape[0])))

non_missing_idx = ~np.isnan(data_full_window).any(axis=1).any(axis=1)
data_full_window = data_full_window[non_missing_idx, :, :]
label_full = label_full[non_missing_idx]

data_full_window = np.divide((data_full_window-np.min(data_full_window)), (np.max(data_full_window)-np.min(data_full_window)))

## shuffle data
idx = np.arange(data_full_window.shape[0])
np.random.seed(2022)
np.random.shuffle(idx)
train_window = data_full_window[idx, :, :]
label_full = label_full[idx]
train_window = torch.tensor(train_window, dtype = torch.float32)
label_full = torch.tensor(label_full, dtype = torch.int)

train_window_, val_window_ = train_test_split(train_window, test_size = 0.2, random_state = 2022)
train_label_, val_label_ = train_test_split(label_full, test_size = 0.2, random_state = 2022)
train_data_class = TensorDataset(train_window_, train_label_)
test_data_class = TensorDataset(val_window_, val_label_)
train_loader_class = torch.utils.data.DataLoader(train_data_class, batch_size = 10, shuffle = True)
test_loader_class = torch.utils.data.DataLoader(test_data_class, batch_size = 10, shuffle = True)

#%% define LSTM auto-encoder
class Lstm_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=normalmax, hidden_size=16)
        self.lstm2 = nn.LSTM(input_size=16, hidden_size=4)
        
    def forward(self, x):
        #reshape x to fit the input requirement of lstm
        x = x.permute(0, 2, 1)
        output, hn = self.lstm1(x)
        output, (hidden, cell) = self.lstm2(output)
        # output include all timestep, while hidden just include the last timestep.
        hidden = hidden.repeat((output.shape[0], 1, 1))
        return hidden

class Lstm_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=16)
        self.lstm2 = nn.LSTM(input_size=16, hidden_size=normalmax)
    
    def forward(self, x):
        # not need to reshape
        output, hn = self.lstm1(x)
        output, hn = self.lstm2(output)
        #reshape output
        output = output.permute(0, 2, 1)
        return output
    
class net(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.Lstm_encoder1 = args[0]
        self.Lstm_decoder1 = args[1]
        self.Lstm_encoder2 = args[2]
        self.Lstm_decoder2 = args[3]
        self.Lstm_encoder3 = args[4]
        self.Lstm_decoder3 = args[5]

    def forward(self, x):
        output1 = self.Lstm_encoder1(x[:, :, 0].unsqueeze(2))
        output1 = self.Lstm_decoder1(output1)
        output2 = self.Lstm_encoder2(x[:, :, 1].unsqueeze(2))
        output2 = self.Lstm_decoder2(output2)
        output3 = self.Lstm_encoder3(x[:, :, 2].unsqueeze(2))
        output3 = self.Lstm_decoder3(output3)
        output = torch.cat((output1, output2, output3), dim = 2)
        return output
    
#% model functions
def train(model, device, train_loader, optimizer, epoch):
  model.train() #trian model
  for batch_idx, data in enumerate(train_loader):
    data = data.to(device)
    optimizer.zero_grad()
    output = model(data)

    ##calculate loss
    loss = 0
    for i in range(data.shape[0]):
      loss += F.mse_loss(output[i], data[i], reduction='mean')

    #loss = F.mse_loss(output, data)
    loss.backward()
    optimizer.step()
    # print result every 10 batch
    if batch_idx % 10 == 0:
      print('Train Epoch: {} ... Batch: {} ... Loss: {:.8f}'.format(epoch, batch_idx, loss))

def test(model, device, test_loader):
  model.eval() #evaluate model
  test_loss = 0
  with torch.no_grad():
    for data in test_loader:
      data = data.to(device)
      output = model(data)
      #calculate sum loss
      test_loss += F.mse_loss(output, data, reduction='mean').item()

    print('------------------- Test set: Average loss: {:.4f} ... Samples: {}'.format(test_loss, len(test_loader.dataset)))


#% model training
train_window_, val_window_ = train_test_split(train_window, test_size = 0.2, random_state = 2022)

train_label_, val_label_ = train_test_split(label_full, test_size = 0.2, random_state = 2022)

train_loader = torch.utils.data.DataLoader(train_window_, batch_size = 256, shuffle = True)
test_loader = torch.utils.data.DataLoader(val_window_, batch_size = 256, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = net(Lstm_encoder(), Lstm_decoder())
model = net(Lstm_encoder(), Lstm_decoder(), Lstm_encoder(), Lstm_decoder(), Lstm_encoder(), Lstm_decoder())
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.00001)

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)

epochs = 500

for epoch in range(1, epochs + 1):
  train(model, device, train_loader, optimizer, epoch)
  test(model, device, test_loader)
  
#%% unsupervised clustering
def model_embedding(model, input):
  model.eval()
  with torch.no_grad():
    output0 = model.Lstm_encoder1(input[:, :, 0].unsqueeze(2))
    output1 = model.Lstm_encoder2(input[:, :, 1].unsqueeze(2))
    output2 = model.Lstm_encoder3(input[:, :, 2].unsqueeze(2))
    output = torch.cat((output0, output1, output2), dim = 2)
    output = output.to('cpu').numpy()
  return output

flag = False

for data in test_loader:
  data = data.to(device)
  output_ = model_embedding(model, data)
  if not flag:
    output = output_.copy()
    flag = True
  else:
    output = np.concatenate([output, output_])

flag = False
val_loader = torch.utils.data.DataLoader(train_window_, batch_size=256,shuffle=False)

for data in val_loader:
  data = data.to(device)
  output_ = model_embedding(model, data)
  if not flag:
    output = output_.copy()
    flag = True
  else:
    output = np.concatenate([output, output_])

output.shape, train_window.shape

# try pca to reduce dimension
from sklearn.decomposition import PCA

pca = PCA(n_components = 5)
pca.fit(output.squeeze(1))

print(pca.explained_variance_ratio_)

valid_2 = pca.transform(output.squeeze(1))

l = ['blue' if t == 0 else 'red' for t in train_label_]

plt.scatter(valid_2[:, 0], valid_2[:, 1], c=l)
plt.show()

#% unsupervised spectral clustering
clustering = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(output.squeeze(1))

result_label = clustering.labels_

true_label = train_label_.numpy()

print(confusion_matrix(true_label, result_label))

#%% supervised SVM
from sklearn import svm
clf = svm.SVC(kernel = 'poly', gamma = 200)
clf.fit(output.squeeze(1), train_label_)
svm_label = clf.predict(output.squeeze(1))

print(confusion_matrix(train_label_.numpy(), svm_label))

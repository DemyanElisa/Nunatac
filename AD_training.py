# Python libraries:
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import BernoulliRBM
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import svm
import random
import numpy as np
import pickle
import tqdm
# Initialization:
random.seed(42)
np.random.seed(42)


class CNNModel(nn.Module):
    def __init__(self):
        """
        Initialises CNN model (customized for the default preprocessing)

        """
        super(CNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,  stride=1, padding=2)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(197120, 2)

    def forward(self, x):
        """
        Makes feed-forward pass of the input data

        PARAMETERS
        -------
            x (array) - array of signals (input data)

        RETURNS
        -------
            out (array) - output of the CNN model

        """
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        Initialises RNN model

        PARAMETERS
        -------
            input_dim (int) - input dimension
            hidden_dim (int) - the number of features in the hidden state h
            layer_dim (int) - number of recurrent layers
            output_dim (int) - output dimension

        """
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, device=torch.device(0)):
        """
        Makes feed-forward pass of the input data

        PARAMETERS
        -------
            x (array) - array of signals (input data)

        RETURNS
        -------
            out (array) - output of the RNN model

        """
    
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0 = torch.Tensor(h0).to(device=device)
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())
        # Index hidden state of last time step
        # out.size() --> 100, 28, 10
        # out[:, -1, :] --> 100, 10 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out



class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, output_dim):
        """
        Initialises autoencoder

        PARAMETERS
        -------
            input_dim (int) - input dimension
            hidden_dim (int) - hidden layer dimension
            bottleneck_dim (int) - bottleneck (central) layer dimension
            output_dim (int) - output dimension

        """
        super(AE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc4 = nn.Linear(bottleneck_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Makes feed-forward pass of the input data

        PARAMETERS
        -------
            x (array) - array of signals (input data)

        RETURNS
        -------
            out (array) - output of the RNN model

        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        return out




class Training:

    def __init__(self, data, labels, model_choice='SVM', model = None,
                input_dim = None, hidden_dim = None, layer_dim = None, bottleneck_dim = None, output_dim = None, rbm_hidden_units=None):
        
        """
        Initialises AD inference class

        PARAMETERS
        -------
            data (array)
            model_choice (str)
            model
            input_dim (int) - input dimension
            hidden_dim (int) - hidden layer dimension
            bottleneck_dim (int) - bottleneck (central) layer dimension for AE
            layer_dim (int) - number of recurrent layers for RNN
            output_dim (int) - output dimension
            rbm_hidden_units (int) - number of hidden units for the RBM model
        
        """
        self.model_choice = model_choice
        input_dim = 320
        hidden_dim = 20
        layer_dim = 1
        output_dim = 2
        if self.model_choice == 'KNN':
            self.model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
        elif self.model_choice == 'DBSCAN':
            self.model = DBSCAN(eps=3, min_samples=2) #tuning eps parameter?
        elif self.model_choice == 'ISOF':
            self.model = IsolationForest(random_state=0)
        elif self.model_choice == 'LOF':
            self.model = LocalOutlierFactor(n_neighbors=2) #tune n_neighbors parameter?
        elif self.model_choice == 'AUTOENCODER':
            self.model = AE(input_dim, hidden_dim, bottleneck_dim, output_dim)
        elif self.model_choice == 'SVM':
            self.model = svm.LinearSVC(C = 0.01, max_iter = 1e4)
        elif self.model_choice == 'RNN':
            print('input_dim {}, hidden_dim {}, layer_dim {}, output_dim {}'.format(input_dim, hidden_dim, layer_dim, output_dim))
            self.model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
        elif self.model_choice == 'CNN':
            self.model = CNNModel()
        elif self.model_choice == 'RBM':
            self.model = BernoulliRBM(n_components=rbm_hidden_units)
     

    def cent_acc_loader(self, model, loader, loss_fn, acc_fn, device):
         """
          Computes mean loss over the batches given a loss function
   
          PARAMETERS
          -------
              model
              loader (torch DataLoader) - generator of data and labels batches
              loss_fn (function) - function computing loss between predictions and true labels
              acc_fn (function) - number of epochs for training
              device (int/str) - gpu id to train on, o.w. "cpu"
          """
         with torch.no_grad():
             cent_list = []
             acc_list = []
             for data, targets in loader:
                 data, targets = data.to(device), targets.to(device)
                 preds = model(data)
                 cent_list.append(loss_fn(preds, torch.reshape(targets, (-1,))).cpu().item())
                 acc_list.append(acc_fn(preds, targets).cpu().item())
      
         return np.mean(cent_list), np.mean(acc_list)

    def train_nn_model(self, train_loader, num_epochs=100, batch_size=12, device=torch.device(0), lr=0.01):
        """
        Trains selected neural net model

        PARAMETERS
        -------
            model
            data (numpy array) - array of signals
            labels (numpy array) - array of data labels
            num_epochs (int) - number of epochs for training
            batch_size (int) - batch size for training
    
        """
        acc_fn = lambda preds, targets: torch.mean((torch.reshape(targets, (-1,)) == torch.argmax(preds, 1)).float())
        pbar = tqdm(range(num_epochs))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            cent_train_list = []
            acc_train_list = []
            print(epoch)
            for data, targets in train_loader:
                data, targets = data.to(device, dtype=torch.float), targets.to(device)
                print('batch shape is ', data.shape)
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                # Forward pass to get output/logits
                outputs = self.model(data)
                print(outputs.shape, ' output shape')
                loss = criterion(outputs, targets)
                loss.backward()
                cent_train = loss(outputs, torch.reshape(targets, (-1,)) )
                acc_train = acc_fn(outputs, targets)

                # Updating parameters
                optimizer.step()
                cent_train_list.append(cent_train.item())
                acc_train_list.append(acc_train.item())
            pbar.set_description("LOSS (TRAIN): {} ACC (TRAIN): {} ".format( np.mean(cent_train_list), np.mean(acc_train_list))) 
     

    def fit_model(self, data, labels=None, device=torch.device(0), batch_size=12):
        
        """
        Train selected model
    
        PARAMETERS
        -------
            data (numpy array) - array of signals
            labels (numpy array) - data binary labels 
    
        """
        if self.model_choice in ['KNN', 'DBSCAN', 'ISOF', 'LOF', 'SVM', 'RBM']:
            if len(data.shape) > 2:
                data = np.reshape(data, (-1, data.shape[-1])) #assumed that features correspond to the first n-1 dimensions
            print("fitting {} model".format(self.model_choice))
            if self.model_choice in ['SVM']:
                #m = self.model
                self.model.fit(data.T, labels, verbose=True)
                #m.fit(data.T, labels)
            else:
                self.model.fit(data, verbose = True)
            print("model fitted")
            
        elif self.model_choice in ['RNN', 'CNN', 'AUTOENCODER']:
            data = torch.Tensor(data).to(device=device).double()
            if self.model_choice == 'CNN':
                data = torch.reshape(data, (data.shape[-1], 1, data.shape[0], data.shape[1]))
            else:
                #data = torch.reshape(data, (data.shape[-1], -1))
                data = torch.reshape(data, (data.shape[-1], data.shape[0], data.shape[1]))
            self.model = self.model.to(device=device)
            print('data reshaped as ', data.shape)
            if not (labels is None):
                labels = torch.Tensor(labels).to(device=device).long()
            train_loader = DataLoader(TensorDataset(data, labels), batch_size=batch_size, shuffle=True)
            print("training {} model".format(self.model_choice))
            self.train_nn_model(self.model, train_loader)
            print("model trained")
            
    def save_model(self, directory):
        pickle.dump(self.model, open('model.pkl', 'wb'))

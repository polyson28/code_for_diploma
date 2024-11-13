# imports 
import pandas as pd
import numpy as np
from numpy.random import RandomState
import random
import os
from scipy.stats import qmc, norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    TensorDataset, 
    DataLoader, 
    random_split, 
    SubsetRandomSampler
)
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import optuna
import torch.optim as optim
from optuna.trial import TrialState
from optuna.samplers import TPESampler, BaseSampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import random
import collections
import itertools
from joblib import Parallel, delayed
from typing import Dict, Union, TypedDict, Tuple
SEED = 42
# ---------------------------------------------------------
# Data generation 
def generate_data(
    s_min: Union[int, float] = 3100,
    s_max: Union[int, float] = 3500,
    k_min: Union[int, float] = 2800,
    k_max: Union[int, float] = 3700,
    t_min: Union[int, float] = 7/365,
    t_max: Union[int, float] = 91/365,
    rf_min: float = 0.08,
    rf_max: float = 0.16,
    vol_min: float = 0.1,
    vol_max: float = 0.25,
    n: int = 1000000
) -> pd.DataFrame:
    """The function for data generation

    Args:
        s_min (Union[int, float], optional): minimum value of the underlying price (rub). Defaults to 3100.
        s_max (Union[int, float], optional): maximum value of the underlying price (rub). Defaults to 3500.
        k_min (Union[int, float], optional): minimum value of the strike (rub). Defaults to 2800.
        k_max (Union[int, float], optional): maximum value of the strike (rub). Defaults to 3700.
        t_min (Union[int, float], optional): minimum value of time-to-maturity (years). Defaults to 7/365.
        t_max (Union[int, float], optional): maximum value of time-to-maturity (years). Defaults to 91/365.
        rf_min (float, optional): minimum value of risk-free rate. Defaults to 0.08.
        rf_max (float, optional): maximum value of risk-free rate. Defaults to 0.16.
        vol_min (float, optional): minimum value of volatility. Defaults to 0.1.
        vol_max (float, optional): maximum value of volatility. Defaults to 0.25.
        n (int, optional): number of data samples for generation. Defaults to 1000000.

    Returns:
        pd.DataFrame: dataframe with generated features in the defined intervals 
    """
    sampler = qmc.LatinHypercube(d=5, seed=SEED)
    sample = sampler.random(n=n)
  
    l_bounds = [s_min, k_min, t_min, rf_min, vol_min]
    u_bounds = [s_max, k_max, t_max, rf_max, vol_max]
  
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    sample_scaled = pd.DataFrame(sample_scaled, columns=['s', 'k', 't', 'rf', 'volatility'])
  
    return sample_scaled
# ---------------------------------------------------------
# Compute the Black-Scholes proce of the options 
def bs_price_calculator(row) -> float:
    """The function for computing the Black-Scholes formula 

    Args:
        row (_type_): the row of the dataframe, in which the price needs to be computed

    Returns:
        float: computed price
    """
    d1 = (
        (np.log(row['s'] / row['k']) + (row['rf'] - .5 * row['volatility'] ** 2) * row['t']) /
        (row['volatility'] * np.sqrt(row['t']))
    )
    d2 = d1 - row['volatility'] * np.sqrt(row['t'])

    price = row['s'] * norm.cdf(d1) - row['k'] * np.exp(-row['rf'] * row['t']) * norm.cdf(d2)

    return price
# ---------------------------------------------------------
# Compute moneyness and scaled price of the option, scale data and divide into training, validation and test sets 
def data_preprocess(
    sample_scaled: pd.DataFrame,
    test_size: float = 0.4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The function for data preprocessing before constructing the model

    Args:
        sample_scaled (pd.DataFrame): dataframe with generated features and computed Black-Scholes option price
        test_size (float, optional): the size of the test set relative to the overall dataset. Defaults to 0.4.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: numpy arrays with train and test features, 
        train and test scaled features, train and test targets
    """
    sample_scaled['bs_price'] = sample_scaled.apply(lambda row: bs_price_calculator(row), axis=1)
    sample_scaled['s_scaled'] = sample_scaled['s'] / sample_scaled['k']
    sample_scaled['p_scaled'] = np.log(sample_scaled['volatility'] / sample_scaled['k'])
    sample_scaled = sample_scaled.drop(['s', 'k', 'bs_price'], axis=1)
  
    target = sample_scaled['volatility']
    features = sample_scaled.drop(['volatility'], axis=1)
    
    # data scaling
    numeric = ['t', 'rf', 's_scaled', 'p_scaled']
    features_scaled = features.copy()
    # use StandardScaler method
    scaler = StandardScaler()
    scaler.fit_transform(features_scaled[numeric])
    features_scaled[numeric] = scaler.transform(features_scaled[numeric])
    
    # not scaled features and target 
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=test_size, random_state=SEED
    )
    
    # scaled features and target 
    features_train_scaled, features_test_scaled, target_train, target_test = train_test_split(
        features_scaled, target, test_size=test_size, random_state=SEED
    )
    
    features_train = features_train.values
    features_train_scaled = features_train_scaled.values
    
    features_test = features_test.values
    features_test_scaled = features_test_scaled.values
    
    target_train = target_train.values
    target_test = target_test.values
  
    return features_train, features_train_scaled, features_test, features_test_scaled, target_train, target_test
# ---------------------------------------------------------
# Clustering
def cluster_data(
    features_train: np.ndarray,
    target_train: np.ndarray,
    features_train_scaled: np.ndarray,
    features_test_scaled: np.ndarray,
    features_test: np.ndarray,
    target_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The function for clustering data

    Args:
        features_train (np.ndarray): features train
        target_train (np.ndarray): target train
        features_train_scaled (np.ndarray): features train scaled
        features_test_scaled (np.ndarray): features test scaled
        features_test (np.ndarray): features test
        target_test (np.ndarray): target test 

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: numpy arrays 
        with features and target for training and test sets for ITM and OTM clusters
    """
    features_train = pd.DataFrame(
        features_train, 
        columns=['t', 'rf', 's_scaled', 'p_scaled']
    )
    features_test = pd.DataFrame(
        features_test, 
        columns=['t', 'rf', 's_scaled', 'p_scaled']
    )

    itm_index_train = features_train[
        features_train['s_scaled'] > 1
    ].index
    
    itm_index_test = features_test[
        features_test['s_scaled'] > 1
    ].index
    
    otm_index_train = features_train[
        features_train['s_scaled'] < 1
    ].index
    
    otm_index_test = features_test[
        features_test['s_scaled'] < 1
    ].index

    itm_features_train_scaled = features_train_scaled[itm_index_train]
    itm_target_train = target_train[itm_index_train]
    
    itm_features_test_scaled = features_test_scaled[itm_index_test]
    itm_target_test = target_test[itm_index_test]
    
    otm_features_train_scaled = features_train_scaled[otm_index_train]
    otm_target_train = target_train[otm_index_train]
    
    otm_features_test_scaled = features_test_scaled[otm_index_test]
    otm_target_test = target_test[otm_index_test]

    return (
        itm_features_train_scaled, 
        itm_target_train,
        itm_features_test_scaled, 
        itm_target_test, 
        otm_features_train_scaled, 
        otm_target_train,
        otm_features_test_scaled, 
        otm_target_test
    )
# ---------------------------------------------------------
# Initialize and train MLP neural network
class MLP(nn.Module):
    """
    Class for initialization and training MLP neural network
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        lr: float, 
        optimizer: str, 
        activation: str, 
        epochs: int, 
        batch_size: int, 
        cv_splits: int,
        lr_scheduler: Union[str, None] = None,
        dropout: float = 0.0
    ):
        """
        Args:
            input_size (int): input size (number of features)
            hidden_size (int): hidden size (number of neurons in the hidden layer)
            lr (float): learning rate
            optimizer (str): optimizer
            activation (str): activation function
            epochs (int): number of epochs for training
            batch_size (int): batch size
            cv_splits (int): number of splits in the cross validation procedure
            lr_scheduler (Union[str, None], optional): dynamic adjustment of the learning rate. Defaults to None.
            dropout (float, optional): dropout rate. Defaults to 0.0.
        """
        super(MLP, self).__init__()
        
        torch.manual_seed(SEED)

        self.epochs = epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
        self.cv_splits = cv_splits
        
        # use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
            
        # MLP initialization with 4 hidden layers 
        self.net = nn.Sequential(
            # input layer
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation,

            # first hidden layer
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation,
            nn.Dropout(dropout),
            
            # second hidden layer
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation,
            nn.Dropout(dropout),
            
            # third hidden layer
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation,
            nn.Dropout(dropout),
            
            # fourth hidden layer
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(dropout),

            # output layer
            nn.Linear(self.hidden_size, 1),
            self.activation  
        )
        self.net = self.net.to(self.device)

        # initialization
        for name, param in self.net.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)   

        self.criterion = nn.MSELoss()

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        elif optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            
        if lr_scheduler == 'reduce_lr_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3
            )
        elif lr_scheduler == 'step_lr':
            self.scheduler = StepLR(
                self.optimizer, step_size=6, gamma=0.8
            )
        elif lr_scheduler == 'cosine_annealing_lr':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=20
            )
        else:
            self.scheduler = None
            
        self.lr_scheduler_name = lr_scheduler

    def forward(
        self, 
        features: torch.Tensor
    ):
        """Forward pass of the neural network

        Args:
            features (torch.Tensor): features for performing the forward pass

        Returns:
            _type_: predictions of the model
        """
        return self.net(features)

    def train_model(
        self, 
        features_train: np.ndarray, 
        target_train: np.ndarray,
        features_test: np.ndarray,
        target_test: np.ndarray, 
        verbose: bool = True
    ): 
        """The function for training the MLP neural network

        Args:
            features_train (np.ndarray): features from the training dataset
            target_train (np.ndarray): target values from the training dataset
            features_test (np.ndarray): features from the test datase
            target_test (np.ndarray): target values from the test dataset
            verbose (bool, optional): whether to print the results while training (print if True). Defaults to True.

        Returns:
            _type_: list of training losses during training, list of validation losses during training, aggregated test loss
        """
        # split the data into K folds
        # for each fold train and evaluate the model
        fold_num = 0
        self.all_train_losses = []
        self.all_val_losses = []
        
        KFold_split = KFold(n_splits=self.cv_splits, random_state=42, shuffle=True)
        for train_index, valid_index in KFold_split.split(features_train):
            features_train_fold, target_train_fold = (
                torch.from_numpy(features_train[train_index]), 
                torch.from_numpy(target_train[train_index].reshape(-1, 1))
            )
            features_valid_fold, target_valid_fold = (
                torch.from_numpy(features_train[valid_index]), 
                torch.from_numpy(target_train[valid_index].reshape(-1, 1))
            )
            
            # set to GPU
            features_train_fold = features_train_fold.to(self.device, dtype=torch.float32)
            target_train_fold = target_train_fold.to(self.device, dtype=torch.float32)
            features_valid_fold = features_valid_fold.to(self.device, dtype=torch.float32)
            target_valid_fold = target_valid_fold.to(self.device, dtype=torch.float32)

            train_dataset = TensorDataset(features_train_fold, target_train_fold)
            valid_dataset = TensorDataset(features_valid_fold, target_valid_fold)

            # Create DataLoaders (using mini-batches)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

            self.train_losses = []
            self.val_losses = []
            
            for epoch in range(self.epochs):
                train_loss = 0
                self.net.train()
                # for each mini-batch train the model
                for inputs, labels in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                
                    self.optimizer.step()
                    train_loss += loss.item()
                    
                train_loss_overall = train_loss / len(train_loader)
                self.train_losses.append(train_loss_overall)
                
                if verbose:
                    print(
                        f"Fold {fold_num+1}/{self.cv_splits},"
                        f"Epoch {epoch+1}/{self.epochs},"
                        f"Training Loss: {np.round(train_loss_overall, 7)}"
                    )           
                
                self.net.eval()
                val_loss = 0.0
                # Validation loop
                with torch.no_grad():
                    # for each mini-batch evaluate the model
                    for inputs, labels in valid_loader:
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        
                    val_loss_overall = val_loss / len(valid_loader)
                    self.val_losses.append(val_loss_overall)
                    if self.verbose:
                        print(f"Validation Loss: {np.round(val_loss_overall, 7)}")
                        
                if self.scheduler is not None:
                    if self.lr_scheduler_name == 'reduce_lr_on_plateau':
                        self.scheduler.step(val_loss_overall)
                    else:
                        self.scheduler.step()
                        
            self.all_train_losses.append(self.train_losses)
            self.all_val_losses.append(self.val_losses)
            
            fold_num += 1

        # evaluate the model on test dataset 
        if not isinstance(features_test, torch.Tensor):
            features_test = torch.from_numpy(features_test)
            features_test = features_test.to(self.device, dtype=torch.float32)
        
        if not isinstance(target_test, torch.Tensor):
            target_test = torch.from_numpy(target_test.reshape(-1, 1))
            target_test = target_test.to(self.device, dtype=torch.float32)
                
        test_dataset = TensorDataset(features_test, target_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
       
        self.net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
        test_loss_overall = test_loss / len(test_loader)
        if verbose:
            print('')
            print(f"Train MSE loss overall: {np.mean(self.all_train_losses)}")
            print(f"Validation MSE loss overall: {np.mean(self.all_val_losses)}")
            print(f"Test MSE loss: {np.round(test_loss_overall, 7)}")
            
        return self.all_train_losses, self.all_val_losses, test_loss_overall
    
    def plot_train_valid_MSE(
        self
    ):
        """The function for plotting the graph of MSE losses on epochs during the training process
        """
        averaged_train_losses = []
        averaged_val_losses = []
        
        for epoch in range(self.epochs):
            avg_train_loss = 0
            for fold in self.all_train_losses:
                avg_train_loss += fold[epoch]
            averaged_train_losses.append(avg_train_loss)
                
        for epoch in range(self.epochs):
            avg_val_loss = 0
            for fold in self.all_val_losses:
                avg_val_loss += fold[epoch]
            averaged_val_losses.append(avg_val_loss)
        
        plt.figure(figsize=(8, 5))
        plt.plot(averaged_train_losses, label='train loss')
        plt.plot(averaged_val_losses, label='valid loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.show()
# ---------------------------------------------------------
# Hyperparameters tuning
class ParamDict(TypedDict):
    hidden_size: int
    input_size: int
    lr: float
    activation: str
    optimizer: str
    initialization: str
    batch_size: int
    dropout: float

class TuneHyperparameters:
    """
    Class for hyperparameters tuning for baseline model and networks for ITM and OTM cluaters 
    """
    def __init__(
        self,
        is_cluster: bool,
        features_train: pd.DataFrame, 
        features_test: pd.DataFrame, 
        target_train: pd.DataFrame, 
        target_test: pd.DataFrame, 
        param_dict: Union[ParamDict, None] = None, 
        verbose: bool = False
    ): 
        """
        Args:
            is_cluster (bool): whether we consider the baseline model (False) or model for clusters (True)
            features_train (pd.DataFrame): features train
            features_test (pd.DataFrame): features test
            target_train (pd.DataFrame): target train
            target_test (pd.DataFrame): target test 
            param_dict (Union[ParamDict, None], optional): dictionary with parameters ranges. Defaults to None.
            verbose (bool, optional): whether to print the outputs (True) or not (False). Defaults to False.
        """
        self.is_cluster = is_cluster
        
        if param_dict is None: 
            if self.is_cluster: 
                self.param_dict = {
                    'hidden_size': [40, 200],
                    'input_size': features_train.shape[1],
                    'lr': [0.0001, 0.01],
                    'activation': ['tanh', 'sigmoid'],
                    'optimizer': ['SGD', 'RMSprop', 'Adam'],
                    'initialization': ['glorot'],
                    'batch_size': [256, 512, 1024],
                    'dropout': [0.01, 0.1]
                }
            else:
                self.param_dict = {
                    'hidden_size': [80, 400],
                    'input_size': features_train.shape[1],
                    'lr': [0.0001, 0.01],
                    'activation': ['tanh', 'sigmoid'],
                    'optimizer': ['SGD', 'RMSprop', 'Adam'],
                    'batch_size': [256, 512, 1024, 2048],
                    'dropout': [0.05, 0.3]
                }
        else: 
            self.param_dict = param_dict
            
        self.features_train = features_train
        self.features_test = features_test
        self.target_train = target_train
        self.target_test = target_test 
        
        self.verbose = verbose
        
    def define_model(
        self,
        trial
    ) -> nn.Sequential:
        """The functions for defining the model for hypermarameters tuning 

        Args:
            trial (_type_): trial, corresponds to estimation of particular set of hyperparameters

        Returns:
            nn.Module: MLP neural network model with defined hyperparameters 
        """
        torch.manual_seed(SEED)
        
        hidden_size = trial.suggest_int(
                'hidden_size', self.param_dict['hidden_size'][0], self.param_dict['hidden_size'][1]
        )
        activation = trial.suggest_categorical(
                'activation', self.param_dict['activation']
        )
        dropout = trial.suggest_float(
                'dropout', self.param_dict['dropout'][0], self.param_dict['dropout'][1]
        )
        
        # activation
        if activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
            
        # use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        net = nn.Sequential(
            # input layer
            nn.Linear(self.param_dict['input_size'], hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation,

            # first hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation,
            nn.Dropout(dropout),
            
            # second hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation,
            nn.Dropout(dropout),
            
            # third hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation,
            nn.Dropout(dropout),
            
            # fourth hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),

            # output layer
            nn.Linear(hidden_size, 1),
            activation
        )
        net = net.to(device)

        # initialization
        for name, param in net.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
        return net
            
    def objective(
        self,
        trial
    ) -> float:
        """The function for suggesting the hyperparameters set from the defined

        Args:
            trial (_type_): trial, corresponds to estimation of particular set of hyperparameters

        Returns:
            float: validation loss (MSE)
        """
        model = self.define_model(
            trial=trial
        )
        
        epochs=20
        cv_splits=3
        
        optimizer = trial.suggest_categorical(
            'optimizer', self.param_dict['optimizer']
        )
        lr = trial.suggest_float(
            'lr', self.param_dict['lr'][0], self.param_dict['lr'][1]
        )
        batch_size = trial.suggest_categorical(
            'batch_size', self.param_dict['batch_size']
        )
        
        if optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        criterion = nn.MSELoss()
        
        # use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # split the data into K folds
        # for each fold train and evaluate the model
        fold_num = 0
        train_losses = []
        val_losses = []
        
        KFold_split = KFold(n_splits=cv_splits, random_state=42, shuffle=True)
        for train_index, valid_index in KFold_split.split(self.features_train):
            features_train_fold, target_train_fold = (
                torch.from_numpy(self.features_train[train_index]), 
                torch.from_numpy(self.target_train[train_index].reshape(-1, 1))
            )
            features_valid_fold, target_valid_fold = (
                torch.from_numpy(self.features_train[valid_index]), 
                torch.from_numpy(self.target_train[valid_index].reshape(-1, 1))
            )
                
            # set to GPU
            features_train_fold = features_train_fold.to(device, dtype=torch.float32)
            target_train_fold = target_train_fold.to(device, dtype=torch.float32)
            features_valid_fold = features_valid_fold.to(device, dtype=torch.float32)
            target_valid_fold = target_valid_fold.to(device, dtype=torch.float32)

            train_dataset = TensorDataset(features_train_fold, target_train_fold)
            valid_dataset = TensorDataset(features_valid_fold, target_valid_fold)

            # Create DataLoaders (using mini-batches)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
                
            for epoch in range(epochs):
                train_loss = 0
                # for each mini-batch train the model
                for inputs, labels in train_loader:
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    #  calculate new gradients
                    loss.backward()
                    # step of gradient descent algorithm
                    optimizer.step()
                    train_loss += loss.item()
                    # set gradient values to zero before next step 
                    optimizer.zero_grad()
                    
                train_loss_overall = train_loss / len(train_loader)
                train_losses.append(train_loss_overall)
                
                if self.verbose:
                    print(
                        f"Fold {fold_num+1}/{cv_splits},"
                        f"Epoch {epoch+1}/{epochs},"
                        f"Training Loss: {np.round(train_loss, 4)}"
                    )
                    
                # Validation loop
                with torch.no_grad():
                    val_loss = 0.0
                    # for each mini-batch evaluate the model
                    for inputs, labels in valid_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                    val_loss /= len(valid_loader)
                    val_losses.append(val_loss)
                    if self.verbose:
                        print(f"Validation Loss: {np.round(val_loss, 4)}")
                        
            fold_num += 1
                
        return np.mean(val_losses)
            
    def tune_hyperparameters(
        self,
        n_successfull_trials: int
    ) -> Dict:
        """The function for hyperparameters tuning

        Args:
            n_successfull_trials (int):  required number of successful trials 

        Returns:
            Dict: dictionary with optimal set of hyperparameters 
        """
        sampler = TPESampler(
            n_startup_trials=10,
            seed=SEED
            )
        pruner = MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=5,
            interval_steps=3
        )
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Ensure that we get n_successfull_trials successful trials
        successful_trials = 0
        
        pbar = tqdm(total=n_successfull_trials, desc="Successful Trials", unit="trial")
        
        while successful_trials < n_successfull_trials:
            study.optimize(self.objective, n_trials=1, n_jobs=-1)
            trial = study.trials[-1]
        
            if trial.state == TrialState.COMPLETE:
                successful_trials += 1
                pbar.update(1)  # Update progress bar
        
        # Close the progress bar
        pbar.close()
        
        # Print the best trial
        best_trial = study.best_trial
        print(f'Best trial: {best_trial.number}')
        print(f'  Value: {best_trial.value}')
        print(f'  Params: ')
        for key, value in best_trial.params.items():
            print(f'    {key}: {value}')
        
        # Print the number of completed trials
        print(f'Number of successful trials: {successful_trials}')
        
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()

        return best_trial.params.items()
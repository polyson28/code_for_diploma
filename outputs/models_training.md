# Prints during training process for models 

## Baseline model
```python
model = MLP(
    input_size=features_train_scaled.shape[1], 
    hidden_size=179, 
    lr=0.006604040491151783, 
    optimizer='Adam', 
    activation='sigmoid', 
    initialization='glorot',
    epochs=15, 
    batch_size=256, 
    cv_splits=3,
    dropout=0.1813728557164148,
    lr_scheduler=None
)

train_losses, val_losses = model.train_model(
    features_train=features_train_scaled, 
    target_train=target_train,
    features_test=features_test_scaled,
    target_test=target_test
)

model.plot_train_valid_MSE()
```
```
Fold 1/3,Epoch 1/15,Training Loss: 0.0016152
Validation Loss: 7.33e-05
Fold 1/3,Epoch 2/15,Training Loss: 0.000183
Validation Loss: 5.05e-05
Fold 1/3,Epoch 3/15,Training Loss: 0.0001171
Validation Loss: 3.24e-05
Fold 1/3,Epoch 4/15,Training Loss: 8.17e-05
Validation Loss: 3.01e-05
Fold 1/3,Epoch 5/15,Training Loss: 6.48e-05
Validation Loss: 2.98e-05
Fold 1/3,Epoch 6/15,Training Loss: 6.13e-05
Validation Loss: 4.29e-05
Fold 1/3,Epoch 7/15,Training Loss: 6.05e-05
Validation Loss: 3.23e-05
Fold 1/3,Epoch 8/15,Training Loss: 6.06e-05
Validation Loss: 3.21e-05
Fold 1/3,Epoch 9/15,Training Loss: 5.86e-05
Validation Loss: 3.67e-05
Fold 1/3,Epoch 10/15,Training Loss: 5.81e-05
Validation Loss: 4.63e-05
Fold 1/3,Epoch 11/15,Training Loss: 5.67e-05
Validation Loss: 3.94e-05
Fold 1/3,Epoch 12/15,Training Loss: 5.59e-05
Validation Loss: 4e-05
Fold 1/3,Epoch 13/15,Training Loss: 5.87e-05
Validation Loss: 3.54e-05
Fold 1/3,Epoch 14/15,Training Loss: 5.6e-05
Validation Loss: 3.07e-05
Fold 1/3,Epoch 15/15,Training Loss: 5.5e-05
Validation Loss: 3.23e-05
Fold 2/3,Epoch 1/15,Training Loss: 5.78e-05
Validation Loss: 4.12e-05
Fold 2/3,Epoch 2/15,Training Loss: 5.6e-05
Validation Loss: 3.05e-05
Fold 2/3,Epoch 3/15,Training Loss: 5.69e-05
Validation Loss: 3.52e-05
Fold 2/3,Epoch 4/15,Training Loss: 5.66e-05
Validation Loss: 3.46e-05
Fold 2/3,Epoch 5/15,Training Loss: 5.61e-05
Validation Loss: 5.24e-05
Fold 2/3,Epoch 6/15,Training Loss: 5.47e-05
Validation Loss: 5.78e-05
Fold 2/3,Epoch 7/15,Training Loss: 5.47e-05
Validation Loss: 3.87e-05
Fold 2/3,Epoch 8/15,Training Loss: 5.54e-05
Validation Loss: 4.33e-05
Fold 2/3,Epoch 9/15,Training Loss: 5.45e-05
Validation Loss: 3.56e-05
Fold 2/3,Epoch 10/15,Training Loss: 5.53e-05
Validation Loss: 3.27e-05
Fold 2/3,Epoch 11/15,Training Loss: 5.54e-05
Validation Loss: 3.56e-05
Fold 2/3,Epoch 12/15,Training Loss: 5.46e-05
Validation Loss: 3.76e-05
Fold 2/3,Epoch 13/15,Training Loss: 5.52e-05
Validation Loss: 3.15e-05
Fold 2/3,Epoch 14/15,Training Loss: 5.58e-05
Validation Loss: 3.5e-05
Fold 2/3,Epoch 15/15,Training Loss: 5.67e-05
Validation Loss: 3.37e-05
Fold 3/3,Epoch 1/15,Training Loss: 5.53e-05
Validation Loss: 3e-05
Fold 3/3,Epoch 2/15,Training Loss: 5.56e-05
Validation Loss: 2.92e-05
Fold 3/3,Epoch 3/15,Training Loss: 5.55e-05
Validation Loss: 3.03e-05
Fold 3/3,Epoch 4/15,Training Loss: 5.43e-05
Validation Loss: 2.99e-05
Fold 3/3,Epoch 5/15,Training Loss: 5.45e-05
Validation Loss: 3.94e-05
Fold 3/3,Epoch 6/15,Training Loss: 5.48e-05
Validation Loss: 4.31e-05
Fold 3/3,Epoch 7/15,Training Loss: 5.48e-05
Validation Loss: 3.22e-05
Fold 3/3,Epoch 8/15,Training Loss: 5.38e-05
Validation Loss: 3.23e-05
Fold 3/3,Epoch 9/15,Training Loss: 5.59e-05
Validation Loss: 3.07e-05
Fold 3/3,Epoch 10/15,Training Loss: 5.51e-05
Validation Loss: 3.68e-05
Fold 3/3,Epoch 11/15,Training Loss: 5.49e-05
Validation Loss: 4.64e-05
Fold 3/3,Epoch 12/15,Training Loss: 5.42e-05
Validation Loss: 3.11e-05
Fold 3/3,Epoch 13/15,Training Loss: 5.44e-05
Validation Loss: 2.94e-05
Fold 3/3,Epoch 14/15,Training Loss: 5.53e-05
Validation Loss: 3.43e-05
Fold 3/3,Epoch 15/15,Training Loss: 5.44e-05
Validation Loss: 2.98e-05

Train MSE loss overall: 9.559874651636152e-05
Validation MSE loss overall: 3.6991537898645695e-05
Test MSE loss: 2.99e-05
```

## Cluster-based models without hyperparameters tuning
```python
itm_model = MLP(
    input_size=itm_features_train_scaled.shape[1], 
    hidden_size=179, 
    lr=0.0066040404911517832, 
    optimizer='Adam', 
    activation='sigmoid', 
    initialization='glorot',
    epochs=15, 
    batch_size=256, 
    cv_splits=3,
    dropout=0.1813728557164148,
    lr_scheduler=None
)

itm_train_losses, itm_val_losses, itm_test_loss = itm_model.train_model(
    features_train=itm_features_train_scaled, 
    target_train=itm_target_train,
    features_test=itm_features_test_scaled,
    target_test=itm_target_test
)

itm_model.plot_train_valid_MSE()
```
```
Fold 1/3,Epoch 1/15,Training Loss: 0.0029109
Validation Loss: 9.33e-05
Fold 1/3,Epoch 2/15,Training Loss: 0.0002508
Validation Loss: 6.54e-05
Fold 1/3,Epoch 3/15,Training Loss: 0.0001929
Validation Loss: 0.0001113
Fold 1/3,Epoch 4/15,Training Loss: 0.0001551
Validation Loss: 6.27e-05
Fold 1/3,Epoch 5/15,Training Loss: 0.0001167
Validation Loss: 3.33e-05
Fold 1/3,Epoch 6/15,Training Loss: 9.58e-05
Validation Loss: 5.15e-05
Fold 1/3,Epoch 7/15,Training Loss: 8.16e-05
Validation Loss: 5.66e-05
Fold 1/3,Epoch 8/15,Training Loss: 7.3e-05
Validation Loss: 3.15e-05
Fold 1/3,Epoch 9/15,Training Loss: 6.53e-05
Validation Loss: 3.25e-05
Fold 1/3,Epoch 10/15,Training Loss: 6.31e-05
Validation Loss: 6.84e-05
Fold 1/3,Epoch 11/15,Training Loss: 6.22e-05
Validation Loss: 3e-05
Fold 1/3,Epoch 12/15,Training Loss: 6.12e-05
Validation Loss: 3.43e-05
Fold 1/3,Epoch 13/15,Training Loss: 6.03e-05
Validation Loss: 4.18e-05
Fold 1/3,Epoch 14/15,Training Loss: 6.21e-05
Validation Loss: 3.17e-05
Fold 1/3,Epoch 15/15,Training Loss: 6.06e-05
Validation Loss: 3.5e-05
Fold 2/3,Epoch 1/15,Training Loss: 6.05e-05
Validation Loss: 3.74e-05
Fold 2/3,Epoch 2/15,Training Loss: 5.9e-05
Validation Loss: 3.52e-05
Fold 2/3,Epoch 3/15,Training Loss: 6.03e-05
Validation Loss: 4.97e-05
Fold 2/3,Epoch 4/15,Training Loss: 5.98e-05
Validation Loss: 4.25e-05
Fold 2/3,Epoch 5/15,Training Loss: 5.6e-05
Validation Loss: 3.57e-05
Fold 2/3,Epoch 6/15,Training Loss: 5.87e-05
Validation Loss: 3.45e-05
Fold 2/3,Epoch 7/15,Training Loss: 5.83e-05
Validation Loss: 3.82e-05
Fold 2/3,Epoch 8/15,Training Loss: 5.74e-05
Validation Loss: 3.08e-05
Fold 2/3,Epoch 9/15,Training Loss: 5.68e-05
Validation Loss: 3.15e-05
Fold 2/3,Epoch 10/15,Training Loss: 5.64e-05
Validation Loss: 5.82e-05
Fold 2/3,Epoch 11/15,Training Loss: 5.81e-05
Validation Loss: 3.74e-05
Fold 2/3,Epoch 12/15,Training Loss: 5.63e-05
Validation Loss: 3.14e-05
Fold 2/3,Epoch 13/15,Training Loss: 5.82e-05
Validation Loss: 4.05e-05
Fold 2/3,Epoch 14/15,Training Loss: 5.65e-05
Validation Loss: 3.29e-05
Fold 2/3,Epoch 15/15,Training Loss: 5.66e-05
Validation Loss: 4.67e-05
Fold 3/3,Epoch 1/15,Training Loss: 5.78e-05
Validation Loss: 2.97e-05
Fold 3/3,Epoch 2/15,Training Loss: 5.63e-05
Validation Loss: 3.48e-05
Fold 3/3,Epoch 3/15,Training Loss: 5.68e-05
Validation Loss: 3.89e-05
Fold 3/3,Epoch 4/15,Training Loss: 5.63e-05
Validation Loss: 3.21e-05
Fold 3/3,Epoch 5/15,Training Loss: 5.78e-05
Validation Loss: 4.49e-05
Fold 3/3,Epoch 6/15,Training Loss: 5.55e-05
Validation Loss: 3.15e-05
Fold 3/3,Epoch 7/15,Training Loss: 5.77e-05
Validation Loss: 3.04e-05
Fold 3/3,Epoch 8/15,Training Loss: 5.9e-05
Validation Loss: 3.73e-05
Fold 3/3,Epoch 9/15,Training Loss: 5.67e-05
Validation Loss: 3.49e-05
Fold 3/3,Epoch 10/15,Training Loss: 5.72e-05
Validation Loss: 3.43e-05
Fold 3/3,Epoch 11/15,Training Loss: 5.52e-05
Validation Loss: 4.38e-05
Fold 3/3,Epoch 12/15,Training Loss: 5.55e-05
Validation Loss: 3.54e-05
Fold 3/3,Epoch 13/15,Training Loss: 5.7e-05
Validation Loss: 3.52e-05
Fold 3/3,Epoch 14/15,Training Loss: 5.8e-05
Validation Loss: 3.2e-05
Fold 3/3,Epoch 15/15,Training Loss: 5.66e-05
Validation Loss: 3.1e-05

Train MSE loss overall: 0.00013408738522482997
Validation MSE loss overall: 4.1959561635292746e-05
Test MSE loss: 3.12e-05
```

```python
otm_model = MLP(
    input_size=otm_features_train_scaled.shape[1], 
    hidden_size=179, 
    lr=0.0066040404911517832, 
    optimizer='Adam', 
    activation='sigmoid', 
    initialization='glorot',
    epochs=15, 
    batch_size=256, 
    cv_splits=3,
    dropout=0.1813728557164148,
    lr_scheduler=None
)

otm_train_losses, otm_val_losses, otm_test_loss = otm_model.train_model(
    features_train=otm_features_train_scaled, 
    target_train=otm_target_train,
    features_test=otm_features_test_scaled,
    target_test=otm_target_test
)

otm_model.plot_train_valid_MSE()
```
```
Fold 1/3,Epoch 1/15,Training Loss: 0.0036016
Validation Loss: 7.41e-05
Fold 1/3,Epoch 2/15,Training Loss: 0.0002744
Validation Loss: 8.02e-05
Fold 1/3,Epoch 3/15,Training Loss: 0.0002048
Validation Loss: 6.63e-05
Fold 1/3,Epoch 4/15,Training Loss: 0.0001757
Validation Loss: 0.0001446
Fold 1/3,Epoch 5/15,Training Loss: 0.0001442
Validation Loss: 5.58e-05
Fold 1/3,Epoch 6/15,Training Loss: 0.0001164
Validation Loss: 6.8e-05
Fold 1/3,Epoch 7/15,Training Loss: 9.41e-05
Validation Loss: 3.2e-05
Fold 1/3,Epoch 8/15,Training Loss: 8.62e-05
Validation Loss: 3.66e-05
Fold 1/3,Epoch 9/15,Training Loss: 7.89e-05
Validation Loss: 3.3e-05
Fold 1/3,Epoch 10/15,Training Loss: 6.61e-05
Validation Loss: 2.88e-05
Fold 1/3,Epoch 11/15,Training Loss: 6.22e-05
Validation Loss: 3.43e-05
Fold 1/3,Epoch 12/15,Training Loss: 6.16e-05
Validation Loss: 7.29e-05
Fold 1/3,Epoch 13/15,Training Loss: 6e-05
Validation Loss: 3.82e-05
Fold 1/3,Epoch 14/15,Training Loss: 5.98e-05
Validation Loss: 3.97e-05
Fold 1/3,Epoch 15/15,Training Loss: 5.84e-05
Validation Loss: 3.94e-05
Fold 2/3,Epoch 1/15,Training Loss: 5.92e-05
Validation Loss: 3.92e-05
Fold 2/3,Epoch 2/15,Training Loss: 5.7e-05
Validation Loss: 2.77e-05
Fold 2/3,Epoch 3/15,Training Loss: 5.75e-05
Validation Loss: 4.76e-05
Fold 2/3,Epoch 4/15,Training Loss: 5.71e-05
Validation Loss: 2.93e-05
Fold 2/3,Epoch 5/15,Training Loss: 5.66e-05
Validation Loss: 2.79e-05
Fold 2/3,Epoch 6/15,Training Loss: 5.41e-05
Validation Loss: 2.78e-05
Fold 2/3,Epoch 7/15,Training Loss: 5.7e-05
Validation Loss: 2.92e-05
Fold 2/3,Epoch 8/15,Training Loss: 5.55e-05
Validation Loss: 2.82e-05
Fold 2/3,Epoch 9/15,Training Loss: 5.62e-05
Validation Loss: 5.82e-05
Fold 2/3,Epoch 10/15,Training Loss: 5.22e-05
Validation Loss: 3.08e-05
Fold 2/3,Epoch 11/15,Training Loss: 5.33e-05
Validation Loss: 3.78e-05
Fold 2/3,Epoch 12/15,Training Loss: 5.75e-05
Validation Loss: 3.72e-05
Fold 2/3,Epoch 13/15,Training Loss: 5.45e-05
Validation Loss: 2.87e-05
Fold 2/3,Epoch 14/15,Training Loss: 5.61e-05
Validation Loss: 3.84e-05
Fold 2/3,Epoch 15/15,Training Loss: 5.66e-05
Validation Loss: 4.38e-05
Fold 3/3,Epoch 1/15,Training Loss: 5.36e-05
Validation Loss: 4.73e-05
Fold 3/3,Epoch 2/15,Training Loss: 5.38e-05
Validation Loss: 2.82e-05
Fold 3/3,Epoch 3/15,Training Loss: 5.41e-05
Validation Loss: 2.99e-05
Fold 3/3,Epoch 4/15,Training Loss: 5.47e-05
Validation Loss: 2.92e-05
Fold 3/3,Epoch 5/15,Training Loss: 5.3e-05
Validation Loss: 3.2e-05
Fold 3/3,Epoch 6/15,Training Loss: 5.59e-05
Validation Loss: 3.01e-05
Fold 3/3,Epoch 7/15,Training Loss: 5.7e-05
Validation Loss: 4.29e-05
Fold 3/3,Epoch 8/15,Training Loss: 5.32e-05
Validation Loss: 3.59e-05
Fold 3/3,Epoch 9/15,Training Loss: 5.45e-05
Validation Loss: 2.8e-05
Fold 3/3,Epoch 10/15,Training Loss: 5.24e-05
Validation Loss: 3.88e-05
Fold 3/3,Epoch 11/15,Training Loss: 5.29e-05
Validation Loss: 3.58e-05
Fold 3/3,Epoch 12/15,Training Loss: 5.35e-05
Validation Loss: 2.75e-05
Fold 3/3,Epoch 13/15,Training Loss: 5.37e-05
Validation Loss: 7.21e-05
Fold 3/3,Epoch 14/15,Training Loss: 5.67e-05
Validation Loss: 4.86e-05
Fold 3/3,Epoch 15/15,Training Loss: 5.4e-05
Validation Loss: 2.78e-05

Train MSE loss overall: 0.0001510610368901946
Validation MSE loss overall: 4.288802903253362e-05
Test MSE loss: 2.8e-05
```

```python
# MSE of the two-step model
torch.manual_seed(42)

if not isinstance(itm_features_test_scaled, torch.Tensor):
    itm_features_test_scaled_nn = torch.from_numpy(itm_features_test_scaled)

if not isinstance(itm_features_test_scaled, torch.Tensor):
    otm_features_test_scaled_nn = torch.from_numpy(otm_features_test_scaled)

if not isinstance(itm_target_test, torch.Tensor):
    itm_target_test_nn = torch.from_numpy(itm_target_test.reshape(-1, 1))

if not isinstance(otm_target_test, torch.Tensor):
    otm_target_test_nn = torch.from_numpy(otm_target_test.reshape(-1, 1))

if not isinstance(features_test_scaled, torch.Tensor):
    features_test_scaled_nn = torch.from_numpy(features_test_scaled)

if not isinstance(target_test, torch.Tensor):
    target_test_nn = torch.from_numpy(target_test.reshape(-1, 1))

itm_test_dataset = TensorDataset(itm_features_test_scaled_nn, itm_target_test_nn)
otm_test_dataset = TensorDataset(otm_features_test_scaled_nn, otm_target_test_nn)
overall_test_dataset = TensorDataset(features_test_scaled_nn, target_test_nn)

itm_test_loader = DataLoader(itm_test_dataset, batch_size=256, shuffle=False)
otm_test_loader = DataLoader(otm_test_dataset, batch_size=256, shuffle=False)
overall_test_loader = DataLoader(overall_test_dataset, batch_size=256, shuffle=False)

two_step_train_mse = (
    (np.mean(itm_train_losses) * len(itm_test_loader) + 
     np.mean(otm_train_losses) * len(otm_test_loader)) / 
    len(overall_test_loader)
)

two_step_val_mse = (
    (np.mean(itm_val_losses) * len(itm_test_loader) + 
     np.mean(otm_val_losses) * len(otm_test_loader)) / 
    len(overall_test_loader)
)

two_step_test_mse = (
    (itm_test_loss * len(itm_test_loader) + otm_test_loss * len(otm_test_loader)) / 
    len(overall_test_loader)
)

print(
    f'MSE of the two-step model on the training set: {two_step_train_mse}\n',
    f'MSE of the two-step on the validation set: {two_step_val_mse}\n',
    f'MSE of the two-step on the test set: {two_step_test_mse}'
)
```
```
MSE of the two-step model on the training set: 0.00014174235809494127
MSE of the two-step on the validation set: 4.240044540849312e-05
MSE of the two-step on the test set: 2.979579326313931e-05
```

## Cluster-based models with hyperparameters tuning
```python
itm_model = MLP(
    input_size=itm_features_train_scaled.shape[1], 
    hidden_size=186, 
    lr=0.008831322501128601, 
    optimizer='Adam', 
    activation='sigmoid', 
    initialization='glorot',
    epochs=15, 
    batch_size=256, 
    cv_splits=3,
    dropout=0.017795855225182816,
    lr_scheduler=None
)
​
itm_train_losses, itm_val_losses, itm_test_loss = itm_model.train_model(
    features_train=itm_features_train_scaled, 
    target_train=itm_target_train,
    features_test=itm_features_test_scaled,
    target_test=itm_target_test
)
​
itm_model.plot_train_valid_MSE()
```
```
Fold 1/3,Epoch 1/15,Training Loss: 0.0033323
Validation Loss: 0.0001126
Fold 1/3,Epoch 2/15,Training Loss: 0.0001678
Validation Loss: 0.0001377
Fold 1/3,Epoch 3/15,Training Loss: 0.0001357
Validation Loss: 8.86e-05
Fold 1/3,Epoch 4/15,Training Loss: 0.0001195
Validation Loss: 0.0001107
Fold 1/3,Epoch 5/15,Training Loss: 0.0001023
Validation Loss: 4.47e-05
Fold 1/3,Epoch 6/15,Training Loss: 8.39e-05
Validation Loss: 4.28e-05
Fold 1/3,Epoch 7/15,Training Loss: 7.44e-05
Validation Loss: 3.88e-05
Fold 1/3,Epoch 8/15,Training Loss: 5.85e-05
Validation Loss: 3.19e-05
Fold 1/3,Epoch 9/15,Training Loss: 5.67e-05
Validation Loss: 3.48e-05
Fold 1/3,Epoch 10/15,Training Loss: 5.12e-05
Validation Loss: 3.45e-05
Fold 1/3,Epoch 11/15,Training Loss: 5.09e-05
Validation Loss: 3.1e-05
Fold 1/3,Epoch 12/15,Training Loss: 4.88e-05
Validation Loss: 3.22e-05
Fold 1/3,Epoch 13/15,Training Loss: 5.03e-05
Validation Loss: 4.48e-05
Fold 1/3,Epoch 14/15,Training Loss: 4.92e-05
Validation Loss: 3.09e-05
Fold 1/3,Epoch 15/15,Training Loss: 4.93e-05
Validation Loss: 3.5e-05
Fold 2/3,Epoch 1/15,Training Loss: 5.02e-05
Validation Loss: 3.57e-05
Fold 2/3,Epoch 2/15,Training Loss: 4.93e-05
Validation Loss: 3.49e-05
Fold 2/3,Epoch 3/15,Training Loss: 4.94e-05
Validation Loss: 3.1e-05
Fold 2/3,Epoch 4/15,Training Loss: 5.08e-05
Validation Loss: 4.07e-05
Fold 2/3,Epoch 5/15,Training Loss: 4.86e-05
Validation Loss: 3.17e-05
Fold 2/3,Epoch 6/15,Training Loss: 4.83e-05
Validation Loss: 5.88e-05
Fold 2/3,Epoch 7/15,Training Loss: 4.99e-05
Validation Loss: 3.09e-05
Fold 2/3,Epoch 8/15,Training Loss: 4.88e-05
Validation Loss: 3.1e-05
Fold 2/3,Epoch 9/15,Training Loss: 4.88e-05
Validation Loss: 3.35e-05
Fold 2/3,Epoch 10/15,Training Loss: 4.87e-05
Validation Loss: 4.24e-05
Fold 2/3,Epoch 11/15,Training Loss: 4.75e-05
Validation Loss: 3.07e-05
Fold 2/3,Epoch 12/15,Training Loss: 5.09e-05
Validation Loss: 3.94e-05
Fold 2/3,Epoch 13/15,Training Loss: 4.96e-05
Validation Loss: 3.1e-05
Fold 2/3,Epoch 14/15,Training Loss: 4.73e-05
Validation Loss: 4.84e-05
Fold 2/3,Epoch 15/15,Training Loss: 4.92e-05
Validation Loss: 3.92e-05
Fold 3/3,Epoch 1/15,Training Loss: 4.72e-05
Validation Loss: 3.52e-05
Fold 3/3,Epoch 2/15,Training Loss: 4.9e-05
Validation Loss: 3.61e-05
Fold 3/3,Epoch 3/15,Training Loss: 5e-05
Validation Loss: 3.49e-05
Fold 3/3,Epoch 4/15,Training Loss: 4.79e-05
Validation Loss: 3.2e-05
Fold 3/3,Epoch 5/15,Training Loss: 4.86e-05
Validation Loss: 3.53e-05
Fold 3/3,Epoch 6/15,Training Loss: 5.04e-05
Validation Loss: 3.55e-05
Fold 3/3,Epoch 7/15,Training Loss: 4.79e-05
Validation Loss: 4.87e-05
Fold 3/3,Epoch 8/15,Training Loss: 4.79e-05
Validation Loss: 4.36e-05
Fold 3/3,Epoch 9/15,Training Loss: 4.92e-05
Validation Loss: 6.36e-05
Fold 3/3,Epoch 10/15,Training Loss: 4.73e-05
Validation Loss: 3.1e-05
Fold 3/3,Epoch 11/15,Training Loss: 4.79e-05
Validation Loss: 3.24e-05
Fold 3/3,Epoch 12/15,Training Loss: 4.99e-05
Validation Loss: 3.01e-05
Fold 3/3,Epoch 13/15,Training Loss: 5.01e-05
Validation Loss: 3.44e-05
Fold 3/3,Epoch 14/15,Training Loss: 4.91e-05
Validation Loss: 3.16e-05
Fold 3/3,Epoch 15/15,Training Loss: 4.83e-05
Validation Loss: 3.15e-05

Train MSE loss overall: 0.00013109357684348166
Validation MSE loss overall: 4.369251108616509e-05
Test MSE loss: 3.17e-05
```

```python
otm_model = MLP(
    input_size=otm_features_train_scaled.shape[1], 
    hidden_size=126, 
    lr=0.007782470516967486, 
    optimizer='RMSprop', 
    activation='sigmoid', 
    initialization='glorot',
    epochs=15, 
    batch_size=256, 
    cv_splits=3,
    dropout=0.029803917346959335,
    lr_scheduler=None
)
​
otm_train_losses, otm_val_losses, otm_test_loss = otm_model.train_model(
    features_train=otm_features_train_scaled, 
    target_train=otm_target_train,
    features_test=otm_features_test_scaled,
    target_test=otm_target_test
)
​
otm_model.plot_train_valid_MSE()
```
```
Fold 1/3,Epoch 1/15,Training Loss: 0.0041016
Validation Loss: 0.0003344
Fold 1/3,Epoch 2/15,Training Loss: 0.0002774
Validation Loss: 0.0001201
Fold 1/3,Epoch 3/15,Training Loss: 0.0001482
Validation Loss: 6.06e-05
Fold 1/3,Epoch 4/15,Training Loss: 0.0001039
Validation Loss: 2.79e-05
Fold 1/3,Epoch 5/15,Training Loss: 7.62e-05
Validation Loss: 3.55e-05
Fold 1/3,Epoch 6/15,Training Loss: 6.09e-05
Validation Loss: 4.24e-05
Fold 1/3,Epoch 7/15,Training Loss: 6.13e-05
Validation Loss: 2.99e-05
Fold 1/3,Epoch 8/15,Training Loss: 6.71e-05
Validation Loss: 6.04e-05
Fold 1/3,Epoch 9/15,Training Loss: 6.33e-05
Validation Loss: 3.6e-05
Fold 1/3,Epoch 10/15,Training Loss: 7.12e-05
Validation Loss: 3.48e-05
Fold 1/3,Epoch 11/15,Training Loss: 6.08e-05
Validation Loss: 5.49e-05
Fold 1/3,Epoch 12/15,Training Loss: 5.72e-05
Validation Loss: 4.16e-05
Fold 1/3,Epoch 13/15,Training Loss: 5.96e-05
Validation Loss: 5.82e-05
Fold 1/3,Epoch 14/15,Training Loss: 6.96e-05
Validation Loss: 3.07e-05
Fold 1/3,Epoch 15/15,Training Loss: 5.4e-05
Validation Loss: 5.34e-05
Fold 2/3,Epoch 1/15,Training Loss: 6.36e-05
Validation Loss: 7.42e-05
Fold 2/3,Epoch 2/15,Training Loss: 6.5e-05
Validation Loss: 2.76e-05
Fold 2/3,Epoch 3/15,Training Loss: 6.14e-05
Validation Loss: 8.03e-05
Fold 2/3,Epoch 4/15,Training Loss: 6.03e-05
Validation Loss: 6.41e-05
Fold 2/3,Epoch 5/15,Training Loss: 6.43e-05
Validation Loss: 2.72e-05
Fold 2/3,Epoch 6/15,Training Loss: 5.68e-05
Validation Loss: 3.99e-05
Fold 2/3,Epoch 7/15,Training Loss: 7.2e-05
Validation Loss: 3.54e-05
Fold 2/3,Epoch 8/15,Training Loss: 6.09e-05
Validation Loss: 4.4e-05
Fold 2/3,Epoch 9/15,Training Loss: 5.6e-05
Validation Loss: 7.01e-05
Fold 2/3,Epoch 10/15,Training Loss: 6.14e-05
Validation Loss: 3.72e-05
Fold 2/3,Epoch 11/15,Training Loss: 6.01e-05
Validation Loss: 3.71e-05
Fold 2/3,Epoch 12/15,Training Loss: 6.49e-05
Validation Loss: 3.35e-05
Fold 2/3,Epoch 13/15,Training Loss: 5.73e-05
Validation Loss: 3.34e-05
Fold 2/3,Epoch 14/15,Training Loss: 6.21e-05
Validation Loss: 2.84e-05
Fold 2/3,Epoch 15/15,Training Loss: 5.53e-05
Validation Loss: 2.93e-05
Fold 3/3,Epoch 1/15,Training Loss: 6.4e-05
Validation Loss: 4.13e-05
Fold 3/3,Epoch 2/15,Training Loss: 5.46e-05
Validation Loss: 5.22e-05
Fold 3/3,Epoch 3/15,Training Loss: 5.78e-05
Validation Loss: 3.16e-05
Fold 3/3,Epoch 4/15,Training Loss: 5.94e-05
Validation Loss: 3.21e-05
Fold 3/3,Epoch 5/15,Training Loss: 6.45e-05
Validation Loss: 6.93e-05
Fold 3/3,Epoch 6/15,Training Loss: 5.39e-05
Validation Loss: 2.73e-05
Fold 3/3,Epoch 7/15,Training Loss: 5.76e-05
Validation Loss: 2.82e-05
Fold 3/3,Epoch 8/15,Training Loss: 5.73e-05
Validation Loss: 3.01e-05
Fold 3/3,Epoch 9/15,Training Loss: 7.84e-05
Validation Loss: 7.42e-05
Fold 3/3,Epoch 10/15,Training Loss: 5.98e-05
Validation Loss: 3.4e-05
Fold 3/3,Epoch 11/15,Training Loss: 5.67e-05
Validation Loss: 3.08e-05
Fold 3/3,Epoch 12/15,Training Loss: 6.62e-05
Validation Loss: 2.83e-05
Fold 3/3,Epoch 13/15,Training Loss: 5.77e-05
Validation Loss: 6.71e-05
Fold 3/3,Epoch 14/15,Training Loss: 5.67e-05
Validation Loss: 4.33e-05
Fold 3/3,Epoch 15/15,Training Loss: 6.65e-05
Validation Loss: 3.84e-05

Train MSE loss overall: 0.00015922744584874324
Validation MSE loss overall: 5.1346962478456465e-05
Test MSE loss: 3.86e-05
```

```python
# MSE of the two-step model
torch.manual_seed(42)

if not isinstance(itm_features_test_scaled, torch.Tensor):
    itm_features_test_scaled_nn = torch.from_numpy(itm_features_test_scaled)

if not isinstance(itm_features_test_scaled, torch.Tensor):
    otm_features_test_scaled_nn = torch.from_numpy(otm_features_test_scaled)

if not isinstance(itm_target_test, torch.Tensor):
    itm_target_test_nn = torch.from_numpy(itm_target_test.reshape(-1, 1))

if not isinstance(otm_target_test, torch.Tensor):
    otm_target_test_nn = torch.from_numpy(otm_target_test.reshape(-1, 1))

if not isinstance(features_test_scaled, torch.Tensor):
    features_test_scaled_nn = torch.from_numpy(features_test_scaled)

if not isinstance(target_test, torch.Tensor):
    target_test_nn = torch.from_numpy(target_test.reshape(-1, 1))

itm_test_dataset = TensorDataset(itm_features_test_scaled_nn, itm_target_test_nn)
otm_test_dataset = TensorDataset(otm_features_test_scaled_nn, otm_target_test_nn)
overall_test_dataset = TensorDataset(features_test_scaled_nn, target_test_nn)

itm_test_loader = DataLoader(itm_test_dataset, batch_size=256, shuffle=False)
otm_test_loader = DataLoader(otm_test_dataset, batch_size=256, shuffle=False)
overall_test_loader = DataLoader(overall_test_dataset, batch_size=256, shuffle=False)

two_step_train_mse = (
    (np.mean(itm_train_losses) * len(itm_test_loader) + 
     np.mean(otm_train_losses) * len(otm_test_loader)) / 
    len(overall_test_loader)
)

two_step_val_mse = (
    (np.mean(itm_val_losses) * len(itm_test_loader) + 
     np.mean(otm_val_losses) * len(otm_test_loader)) / 
    len(overall_test_loader)
)

two_step_test_mse = (
    (itm_test_loss * len(itm_test_loader) + otm_test_loss * len(otm_test_loader)) / 
    len(overall_test_loader)
)

print(
    f'MSE of the two-step model on the training set: {two_step_train_mse}\n',
    f'MSE of the two-step on the validation set: {two_step_val_mse}\n',
    f'MSE of the two-step on the test set: {two_step_test_mse}'
)
```
```
MSE of the two-step model on the training set: 0.0001437233914778456
MSE of the two-step on the validation set: 4.713387073524586e-05
MSE of the two-step on the test set: 3.476166060833503e-05
```

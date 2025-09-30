import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import board_utils

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib

def plot_loss_curve(train_loss,val_loss):
    X = np.arange(0,len(train_loss))
    
    plt.plot(X,train_loss,label = 'Training Loss')
    plt.plot(X,val_loss,label = 'Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    
    plt.legend()
    plt.show()
    
def load_data(file_name,input_channels,height=8,width=8):
    df = pd.read_csv(file_name)
                     
    evaluation_data = df['evaluation'].values.reshape(-1, 1)
    material_imbalance_data = df['material_imbalance'].values.reshape(-1, 1)

    board_features = df.drop(columns=['evaluation', 'material_imbalance']).values
    
    num_samples = board_features.shape[0]
    board_data = board_features.reshape(num_samples, input_channels, height, width)
    
    extra_features = torch.from_numpy(material_imbalance_data).float()
    y = torch.from_numpy(evaluation_data).float()

    return board_data, extra_features, y


n_epochs = 100
batch_size = 32

n_input_channels = 20
extra_features_size = 1

X,X_extra,y = load_data('data.csv', 20)

    #shape (N,channels,h,w)


X_train, X_temp, extra_train, extra_temp, y_train, y_temp = train_test_split(
        X, X_extra, y, test_size=0.3, random_state=42
    )

X_val, X_test, extra_val, extra_test, y_val, y_test = train_test_split(
    X_temp, extra_temp, y_temp, test_size=0.5, random_state=42
)

norm_channels = [18,19]

scalers_board = [StandardScaler() for i in range(len(norm_channels))]
X_train_scaled = np.empty_like(X_train)
X_val_scaled = np.empty_like(X_val)
X_test_scaled = np.empty_like(X_test)

for idx,channel in enumerate(norm_channels):
    # Flatten the channel data for scaling (e.g., from 1200x8x8 to 1200x64)
    X_train_channel_flat = X_train[:, channel, :, :].reshape(X_train.shape[0], -1)
    X_val_channel_flat = X_val[:, channel, :, :].reshape(X_val.shape[0], -1)
    X_test_channel_flat = X_test[:, channel, :, :].reshape(X_test.shape[0], -1)
    
    # Fit the scaler on the training channel data and transform all three sets
    X_train_scaled_flat = scalers_board[idx].fit_transform(X_train_channel_flat)
    X_val_scaled_flat = scalers_board[idx].transform(X_val_channel_flat)
    X_test_scaled_flat = scalers_board[idx].transform(X_test_channel_flat)
    
    # Reshape the scaled data back to the original channel shape
    X_train[:, channel, :, :] = X_train_scaled_flat.reshape(X_train_scaled[:, channel, :, :].shape)
    X_val[:, channel, :, :] = X_val_scaled_flat.reshape(X_val_scaled[:, channel, :, :].shape)
    X_test[:, channel, :, :] = X_test_scaled_flat.reshape(X_test_scaled[:, channel, :, :].shape)


# Use StandardScaler for extra features (material imbalance)
scaler_extra = StandardScaler()
extra_train_scaled = scaler_extra.fit_transform(extra_train)
extra_val_scaled = scaler_extra.transform(extra_val)
extra_test_scaled = scaler_extra.transform(extra_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# Convert the scaled numpy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
X_test = torch.from_numpy(X_test).float()
extra_train = torch.from_numpy(extra_train_scaled).float()
extra_val = torch.from_numpy(extra_val_scaled).float()
extra_test = torch.from_numpy(extra_test_scaled).float()
y_train = torch.from_numpy(y_train_scaled).float()
y_val = torch.from_numpy(y_val_scaled).float()
y_test = torch.from_numpy(y_test_scaled).float()


    #----TRAINING----#

model = board_utils.CNN_Regressor(n_input_channels,extra_features_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    batch_no = 0
    for i in range(0, X_train.shape[0], batch_size):
        batch_no += 1
        batch_x = X_train[i:i + batch_size]
        batch_extra = extra_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]
        # Forward pass
        outputs = model(batch_x,batch_extra)
        loss = criterion(outputs, batch_y)
        train_loss += loss.item()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
    train_loss /= batch_no
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_predictions = model(X_val, extra_val)
        val_loss = criterion(val_predictions, y_val)
    
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{n_epochs}], Validation Loss: {val_loss:.4f}")

# After training, evaluate on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    y_pred = model(X_test,extra_test).detach().numpy().T.flatten()
    y_original = y_test.detach().numpy().T.flatten()
    
    plt.hist(y_pred-y_original,bins=30,edgecolor = 'black')
    plt.xlabel('$X-\hat{X}$')
    plt.yscale('log')
    plt.show()
    
    plt.scatter(y_original,y_pred)
    plt.xlabel('y')
    plt.ylabel('$y_{pred}$')
    plt.axline((0,0),slope=1,color='black',ls='--')
    plt.show()
    
    board_eval = extra_test.detach().numpy().flatten()
    plt.scatter(board_eval,y_pred)
    plt.xlabel('Material Imbalance')
    plt.ylabel('$y_{pred}$')
    plt.show()
    
    
model_filename = 'chess_cnn_model.pth'
scalers_board_filename = 'scalers_board.joblib'
scaler_extra_filename = 'scaler_extra.joblib'
scaler_y_filename = 'scaler_y.joblib'

board_utils.save_model(model,model_filename)
joblib.dump(scalers_board, "scalers/"+scalers_board_filename)
joblib.dump(scaler_extra, "scalers/"+scaler_extra_filename)
joblib.dump(scaler_y, "scalers/"+scaler_y_filename)

plot_loss_curve(train_losses, val_losses)

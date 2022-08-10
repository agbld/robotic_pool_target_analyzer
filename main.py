#%%
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from model import NeuralModel

#%%
# read dataset
dataset_df = pd.read_csv('pseudo_dataset.csv')

X = torch.tensor(dataset_df[['angle_1', 'angle_2', 'dist_1', 'dist_2']].values, dtype=torch.float) # unpack X from dataset_df

# normalize X for better training performance
for i in range(X.size(1)):
    X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

Y = torch.tensor(dataset_df['label'].values, dtype=torch.float).unsqueeze(1) # unpack Y from dataset_df

# split data into train and test (4:1)
train_size = int(0.8 * X.size(0))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

#%%
# initialize model, loss, optimizer
model = NeuralModel(input=4, dim=50, dropout=0.3)
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# %%
# train model
epochs = 5000
train_losses = []
test_losses = []

model.train()
with tqdm(total=epochs) as pbar:
    for epoch in range(epochs):
        # forward pass
        y_pred = model(X_train)
        loss_train = loss(y_pred, Y_train)
        
        # backward pass
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        # calculate loss on test set (validation)
        with torch.no_grad():
            y_pred = model(X_test)  
            loss_test = loss(y_pred, Y_test)
        
        # record loss
        train_losses.append(loss_train.item())
        test_losses.append(loss_test.item())
        
        # update progress bar
        pbar.set_postfix(loss_train=loss_train.item(), loss_test=loss_test.item())
        pbar.update()

#%%
with torch.no_grad():
    # plot train loss and test loss
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()

    # calculate accuracy on train, test dataset
    model.eval()
    y_pred = model(X_train)
    accuracy_train = (1 - abs(y_pred - Y_train)).mean()
    y_pred = model(X_test)
    accuracy_test = (1 - abs(y_pred - Y_test)).mean()
    print('train accuracy: {:.2f}%'.format(accuracy_train * 100))
    print('test accuracy: {:.2f}%'.format(accuracy_test * 100))

#%%
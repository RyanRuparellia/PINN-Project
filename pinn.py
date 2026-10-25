import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

'''
# -----------------------------
# Plot raw training data
# -----------------------------

plt.scatter(Ttrain.cpu().numpy(), Xtrain.cpu().numpy(), s=2, label="Training data")
plt.xlabel("time / s")
plt.ylabel("displacement / m")
plt.legend()
plt.show()
'''


#------------------------------
# Normalise and clean data
#------------------------------

#Normalisation
'''
Ttrain_mean = torch.mean(Ttrain).to(DEVICE)
Ttrain_std = torch.std(Ttrain).to(DEVICE)

Ttrain_norm = (Ttrain - Ttrain_mean) / Ttrain_std
Ttest_norm = (Ttest - Ttrain_mean) / Ttrain_std
'''


Ttrain_colloc = torch.linspace(300, 400, 500).view(-1, 1).to(DEVICE)

# -----------------------------
# Dataset class
# -----------------------------

class OscillatorDataset(Dataset):
    def __init__(self, t, x):
        self.t = t
        self.x = x

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.x[idx]


# -----------------------------
# Activation functions
# -----------------------------

class Sin(nn.Module):
    def forward(self, inp):
        return torch.sin(inp)

class DecaySin(nn.Module):
    def forward(self,inp):
        return torch.exp(-torch.abs(inp))*torch.sin(inp)

# -----------------------------
# Neural networks
# -----------------------------

class TanhNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 1)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.omega = nn.Parameter(torch.tensor(1.1))

    def forward(self, tnorm):
        return self.net(tnorm)


class SinNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 64),
            Sin(),

            nn.Linear(64, 64),
            Sin(),

            nn.Linear(64, 1)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.omega = nn.Parameter(torch.tensor(1.1))

    def forward(self, tnorm):
        return self.net(tnorm)


class DecaySinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            Sin(),

            nn.Linear(64, 64),
            Sin(),

            nn.Linear(64, 1)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.omega = nn.Parameter(torch.tensor(1.1))

    def forward(self, tnorm):
        t = tnorm #* Ttrain_std + Ttrain_mean
        envelope = torch.exp(-self.gamma*t)
        
        return envelope * self.net(tnorm)
        

# -----------------------------
# Choose model
# -----------------------------

# Choose one:
#model = TanhNet().to(DEVICE)
model = SinNet().to(DEVICE)
#model = DecaySinNet().to(DEVICE)

# -----------------------------
# Loss function and optimizer
# -----------------------------


loss_fcn = nn.MSELoss()
'''
def loss_fcn(model_x,x_real):
    alpha = 500 #increase this to increase how much large displacement is weighted
    weights = 1 + alpha*torch.abs(x_real)
    return torch.sum(weights*(x_real-model_x)**2)/torch.sum(weights)
    '''
optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 5000
gammaList = []
epochList = []
omegaList = []

#Lamb is the weighting of the physics loss when contributing
lamb = 1

trainIDs = np.arange(100)
errorstds = [0,0.001,0.01,0.1,0.2,0.5]

for trainID in trainIDs:
    for errorstd in errorstds:
        
        
        # -----------------------------
        # Load data
        # -----------------------------
        
        trainData = pd.read_csv("extrapolation/train.csv")
        testData = pd.read_csv("extrapolation/test.csv")
        
        # Sort by time so plotting looks sensible
        trainData = trainData.sort_values("time")
        testData = testData.sort_values("time")
        
        # Convert pandas columns to PyTorch tensors
        Ttrain = torch.tensor(trainData["time"].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
        Xtrain = torch.tensor(trainData["displacement"].values, dtype=torch.float32).view(-1, 1)
        
        Ttest = torch.tensor(testData["time"].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
        Xtest = torch.tensor(testData["displacement"].values, dtype=torch.float32).view(-1, 1)
        
        Ttrain_norm = Ttrain
        Ttest_norm = Ttest
        
        
        train_dataset = OscillatorDataset(Ttrain_norm, Xtrain)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        coll_dataloader = DataLoader(Ttrain_colloc, batch_size = 32, shuffle=True)
        
        
        
        # -----------------------------
        # Training loop
        # -----------------------------
        for epoch in range(epochs):
            model.train()
        
            for batch_t, batch_x in train_dataloader:
                batch_t = batch_t.to(DEVICE)
                batch_x = batch_x.to(DEVICE)
        
                # Forward pass
                batch_t = batch_t.clone().detach().requires_grad_(True)
        
                x_pred = model(batch_t)
        
                # Calculate loss
                data_loss = loss_fcn(x_pred, batch_x)
        
        
                dx_dt = torch.autograd.grad(
                            outputs = x_pred,
                            inputs = batch_t,
                            grad_outputs = torch.ones_like(x_pred),
                            create_graph = True
                            )[0] #/ Ttrain_std
                d2x_dt2 = torch.autograd.grad(
                            outputs = dx_dt,
                            inputs = batch_t,
                            grad_outputs = torch.ones_like(dx_dt),
                            create_graph = True
                            )[0] #/ Ttrain_std
                
                residual = d2x_dt2 + model.gamma*dx_dt + (model.omega**2) *x_pred
                phys_loss = torch.mean(residual**2)
                
                # Collocation points, same step
                tc = Ttrain_colloc.clone().detach().requires_grad_(True)
                x_c = model(tc)
                dx_c = torch.autograd.grad(x_c, tc, torch.ones_like(x_c), create_graph=True)[0]
                d2x_c = torch.autograd.grad(dx_c, tc, torch.ones_like(dx_c), create_graph=True)[0]
                colloc_residual = d2x_c + model.gamma*dx_c + (model.omega**2)*x_c
                colloc_phys_loss = torch.mean(colloc_residual**2)
        
                loss = phys_loss*lamb + colloc_phys_loss*lamb + data_loss
                
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
            # Evaluate training loss once per epoch
        
            model.eval()
            
            Ttrain_eval = Ttrain_norm.to(DEVICE).clone().detach().requires_grad_(True)
            Xtrain_eval = Xtrain.to(DEVICE)
            
            # Prediction
            Xpred_train = model(Ttrain_eval)
            
            # Data loss
            train_data_loss = loss_fcn(Xpred_train, Xtrain_eval)
            
            # First derivative dx/dt
            train_dx_dt = torch.autograd.grad(
                outputs=Xpred_train,
                inputs=Ttrain_eval,
                grad_outputs=torch.ones_like(Xpred_train),
                create_graph=True
            )[0] #/ Ttrain_std
            
            # Second derivative d²x/dt²
            train_d2x_dt2 = torch.autograd.grad(
                outputs=train_dx_dt,
                inputs=Ttrain_eval,
                grad_outputs=torch.ones_like(train_dx_dt),
                create_graph=True
            )[0] #/ Ttrain_std
            
            # Physics residual
            train_residual = train_d2x_dt2 + model.gamma * train_dx_dt + (model.omega**2)*Xpred_train
            
            # Physics loss
            train_phys_loss = torch.mean(train_residual**2)
            
            # Collocation points, same step
            tc = Ttrain_colloc.clone().detach().requires_grad_(True)
            x_c = model(tc)
            dx_c = torch.autograd.grad(x_c, tc, torch.ones_like(x_c), create_graph=True)[0]
            d2x_c = torch.autograd.grad(dx_c, tc, torch.ones_like(dx_c), create_graph=True)[0]
            colloc_residual = d2x_c + model.gamma*dx_c + (model.omega**2)*x_c
            colloc_phys_loss = torch.mean(colloc_residual**2)
        
            train_loss = data_loss + lamb*phys_loss + colloc_phys_loss*lamb
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train total loss = {train_loss.item():.8f} | "
                    f"Train data loss = {colloc_phys_loss.item():.8f} | "
                    f"Train colloc loss = {train_data_loss.item():.8f} | "
                    f"Train physics loss = {train_phys_loss.item():.8f}\n"
                    "gamma:", model.gamma.item(), '\n'
                    "omega:", model.omega.item(), '\n'
                )
                gammaList.append(model.gamma.item())
                omegaList.append(model.omega.item())
                epochList.append(epoch)

           
                
        
        # -----------------------------
        # Test the model
        # -----------------------------
        
        Ttest_eval = Ttest_norm.to(DEVICE).clone().detach().requires_grad_(True)
        Xtest_eval = Xtest.to(DEVICE)
        
        Xpred_test = model(Ttest_eval)
        test_data_loss = loss_fcn(Xpred_test, Xtest_eval)
        
        test_dx_dt = torch.autograd.grad(
            outputs=Xpred_test,
            inputs=Ttest_eval,
            grad_outputs=torch.ones_like(Xpred_test),
            create_graph=True
        )[0] #/ Ttrain_std
        
        test_d2x_dt2 = torch.autograd.grad(
            outputs=test_dx_dt,
            inputs=Ttest_eval,
            grad_outputs=torch.ones_like(test_dx_dt),
            create_graph=True
        )[0] #/ Ttrain_std
        
        test_residual = test_d2x_dt2 + model.gamma * test_dx_dt + (model.omega**2)*Xpred_test
        test_phys_loss = torch.mean(test_residual**2)
        
        test_loss = test_data_loss + test_phys_loss
        
        print(
            f"Final test loss = {test_loss.item():.8f} | "
            f"Final data loss = {test_data_loss.item():.8f} | "
            f"Final physcs loss = {test_phys_loss.item():.8f}"
        )
        


'''
# -----------------------------
# Plot training prediction
# -----------------------------

Xpred_train = Xpred_train.cpu().detach()
Ttrain = Ttrain.cpu().detach()
Xpred_test = Xpred_test.cpu().detach()
Ttest = Ttest.cpu().detach()

plt.scatter(Ttrain.numpy(), Xtrain.numpy(), s=2, label="Training data")
plt.plot(Ttrain.numpy(), Xpred_train.numpy(), label="NN prediction")
plt.xlabel("time / s")
plt.ylabel("displacement / m")
plt.title("Training region")
plt.legend()
plt.show()


# -----------------------------
# Plot extrapolation prediction
# -----------------------------

plt.scatter(Ttest.numpy(), Xtest.numpy(), s=2, label="Test data")
plt.plot(Ttest.numpy(), Xpred_test.numpy(), label="NN prediction")
plt.xlabel("time / s")
plt.ylabel("displacement / m")
plt.title("Extrapolation region")
plt.legend()
plt.show()


# -----------------------------
# Plot train and test together
# -----------------------------

plt.scatter(Ttrain.numpy(), Xtrain.numpy(), s=2, label="Training data")
plt.scatter(Ttest.numpy(), Xtest.numpy(), s=2, label="Test data")

plt.plot(Ttrain.numpy(), Xpred_train.numpy(), label="NN train prediction")
plt.plot(Ttest.numpy(), Xpred_test.numpy(), label="NN test prediction", color = 'red')

plt.xlabel("time / s")
plt.ylabel("displacement / m")
plt.title("Damped oscillator NN prediction")
plt.legend()
plt.show()

#----------------------------
#Plot gamma against epochs
#----------------------------
plt.scatter(epochList, gammaList)
plt.show()
plt.scatter(epochList, omegaList)
plt.show()
'''
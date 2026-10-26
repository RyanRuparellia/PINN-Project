import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)




#------------------------------
# Collocation points
#------------------------------

Ttrain_colloc = torch.linspace(300, 400, 500).view(-1, 1).to(DEVICE)

#------------------------------
# True solution (for plotting)
#------------------------------

TRUE_GAMMA = 0.1   # c/m
TRUE_OMEGA = 1.0   # sqrt(k/m)

def true_solution(t):
    '''Analytic solution of x'' + gamma*x' + omega^2*x = 0 with x(0) = 1, x'(0) = 0 (ICs from Kaggle dataset).'''
    decay = TRUE_GAMMA/2
    omega_d = np.sqrt(TRUE_OMEGA**2 - decay**2)
    return np.exp(-decay*t)*(np.cos(omega_d*t) + (decay/omega_d)*np.sin(omega_d*t))

# Dense grid over the training and extrapolation regions, used for plots
Tplot = torch.linspace(0, 400, 2000).view(-1, 1).to(DEVICE)

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

    def forward(self, t):
        return self.net(t)


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

    def forward(self, t):
        return self.net(t)


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

    def forward(self, t):
        envelope = torch.exp(-self.gamma*t/2)

        return envelope * self.net(t)
        

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


#loss_fcn = nn.MSELoss()

def loss_fcn(model_x,x_real):
    alpha = 500 #increase this to increase how much large displacement is weighted
    weights = 1 + alpha*torch.abs(x_real)
    return torch.sum(weights*(x_real-model_x)**2)/torch.sum(weights)
    
optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 5000
snapshot_every = 500   # how often to save the prediction for the snapshot plot

# Recorded every epoch
gammaList = []
epochList = []
omegaList = []
dataLossList = []
physLossList = []
collocLossList = []

snapshots = []   # list of (epoch, prediction on Tplot)

num_trials = 30

#Lamb is the weighting of the physics loss
lamb = 1

trialIDs = np.arange(num_trials)
errorstds = [0,0.001,0.01,0.1,0.2,0.5]
# -----------------------------
# Load data
# -----------------------------

trainData = pd.read_csv("reconstruction/train.csv")

# Sort by time so plotting looks sensible
trainData = trainData.sort_values("time")

# Convert pandas columns to PyTorch tensors
Ttrain = torch.tensor(trainData["time"].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
Xtrain = torch.tensor(trainData["displacement"].values, dtype=torch.float32).view(-1, 1)




train_dataset = OscillatorDataset(Ttrain, Xtrain)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)



# -----------------------------
# Training loop
# -----------------------------
for epoch in range(epochs):
    model.train()
    epoch_data_loss = 0.0
    epoch_phys_loss = 0.0
    epoch_colloc_loss = 0.0

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
                    )[0]
        d2x_dt2 = torch.autograd.grad(
                    outputs = dx_dt,
                    inputs = batch_t,
                    grad_outputs = torch.ones_like(dx_dt),
                    create_graph = True
                    )[0]
        
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

        epoch_data_loss += data_loss.item()
        epoch_phys_loss += phys_loss.item()
        epoch_colloc_loss += colloc_phys_loss.item()

    # Record batch-averaged losses and parameters every epoch
    n_batches = len(train_dataloader)
    dataLossList.append(epoch_data_loss/n_batches)
    physLossList.append(epoch_phys_loss/n_batches)
    collocLossList.append(epoch_colloc_loss/n_batches)
    gammaList.append(model.gamma.item())
    omegaList.append(model.omega.item())
    epochList.append(epoch)

    # Save the prediction over the whole domain for the snapshot plot
    if epoch % snapshot_every == 0 or epoch == epochs - 1:
        with torch.no_grad():
            snapshots.append((epoch, model(Tplot).cpu().numpy().ravel()))

    if epoch % 100 == 0:
        # Evaluate training loss once per 100 epochs

        model.eval()
        
        Ttrain_eval = Ttrain.to(DEVICE).clone().detach().requires_grad_(True)
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
        )[0]
        
        # Second derivative d²x/dt²
        train_d2x_dt2 = torch.autograd.grad(
            outputs=train_dx_dt,
            inputs=Ttrain_eval,
            grad_outputs=torch.ones_like(train_dx_dt),
            create_graph=True
        )[0]
        
        # Physics residual
        train_residual = train_d2x_dt2 + model.gamma * train_dx_dt + (model.omega**2)*Xpred_train
        
        # Physics loss
        train_phys_loss = torch.mean(train_residual**2)
        
        # Collocation points, same step
        tc = Ttrain_colloc.clone().detach().requires_grad_(True)
        x_c = model(tc)
        dx_c = torch.autograd.grad(x_c, tc, torch.ones_like(x_c), create_graph=True)[0]
        d2x_c = torch.autograd.grad(dx_c, tc, torch.ones_like(dx_c), create_graph=False)[0]

        colloc_residual = d2x_c + model.gamma*dx_c + (model.omega**2)*x_c
        colloc_phys_loss = torch.mean(colloc_residual**2)

        train_loss = train_data_loss + lamb*train_phys_loss + colloc_phys_loss*lamb
        print(
            f"Epoch {epoch:4d} | "
            f"Train total loss = {train_loss.item():.8f} | "
            f"Train data loss = {train_data_loss.item():.8f} | "
            f"Train colloc loss = {colloc_phys_loss.item():.8f} | "
            f"Train physics loss = {train_phys_loss.item():.8f}\n"
            "gamma:", model.gamma.item(), '\n'
            "omega:", model.omega.item(), '\n'
        )

    
        

# -----------------------------
# Test the model
# -----------------------------
errorstd = errorstds[1]
for trialID in trialIDs:
    testData = pd.read_csv("generated_data/oscillator_data__noise" + str(errorstd) + "__trial_ID" + str(trialID) + ".csv")

    # Sort by time so plotting looks sensible
    testData = testData.sort_values("time")

    # Convert pandas columns to PyTorch tensors
    Ttest = torch.tensor(testData["time"].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
    Xtest = torch.tensor(testData["displacement"].values, dtype=torch.float32).view(-1, 1).to(DEVICE)

    Ttest_eval = Ttest.to(DEVICE).clone().detach().requires_grad_(True)
    Xtest_eval = Xtest.to(DEVICE)

    Xpred_test = model(Ttest_eval)
    test_data_loss = loss_fcn(Xpred_test, Xtest_eval)

    test_dx_dt = torch.autograd.grad(
        outputs=Xpred_test,
        inputs=Ttest_eval,
        grad_outputs=torch.ones_like(Xpred_test),
        create_graph=True
    )[0]

    test_d2x_dt2 = torch.autograd.grad(
        outputs=test_dx_dt,
        inputs=Ttest_eval,
        grad_outputs=torch.ones_like(test_dx_dt),
        create_graph=True
    )[0]

    test_residual = test_d2x_dt2 + model.gamma * test_dx_dt + (model.omega**2)*Xpred_test
    test_phys_loss = torch.mean(test_residual**2)

    test_loss = test_data_loss + test_phys_loss

    print(
        f"Final test loss = {test_loss.item():.8f} | "
        f"Final data loss = {test_data_loss.item():.8f} | "
        f"Final physics loss = {test_phys_loss.item():.8f}"
    )



# -----------------------------
# Plots
# -----------------------------

plot_dir = Path(__file__).resolve().parent / "plots"
plot_dir.mkdir(exist_ok=True)

Tplot_np = Tplot.cpu().numpy().ravel()
Xtrue_plot = true_solution(Tplot_np)
train_end = Ttrain.max().item()

# 1. Prediction snapshots during training
fig, ax = plt.subplots(figsize=(10, 4))
ax.axvspan(train_end, Tplot_np[-1], color="grey", alpha=0.15, label="Extrapolation region")

# Earlier snapshots: thin and faded, coloured by epoch
cmap = plt.cm.viridis
norm = plt.Normalize(0, epochs - 1)
for i, (snap_epoch, snap_pred) in enumerate(snapshots[:-1]):
    ax.plot(Tplot_np, snap_pred, color=cmap(norm(snap_epoch)), lw=0.8, alpha=0.5,
            label="NN prediction (earlier epochs)" if i == 0 else None)
ax.scatter(Ttrain.cpu().numpy(), Xtrain.numpy(), s=4, color="black", label="Training data", zorder=3)

# Final prediction: thick and on top
final_epoch, final_pred = snapshots[-1]
ax.plot(Tplot_np, final_pred, color="tab:blue", lw=2, zorder=4,
        label=f"NN prediction (final, epoch {final_epoch})")

ax.plot(Tplot_np, Xtrue_plot, "r--", lw=1.2, zorder=5, label="True solution")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="epoch")
ax.set_xlabel("time / s")
ax.set_ylabel("displacement / m")
ax.set_title("NN prediction during training")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(plot_dir / "prediction_snapshots.png", dpi=150)
plt.show()

# 2. gamma and omega against epochs
fig, (ax_gamma, ax_omega) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
ax_gamma.plot(epochList, gammaList)
ax_gamma.axhline(TRUE_GAMMA, color="r", ls="--", label=f"True = {TRUE_GAMMA}")
ax_gamma.set_ylabel("gamma")
ax_gamma.legend()

ax_omega.plot(epochList, omegaList)
ax_omega.axhline(TRUE_OMEGA, color="r", ls="--", label=f"True = {TRUE_OMEGA}")
ax_omega.set_ylabel("omega")
ax_omega.set_xlabel("epoch")
ax_omega.legend()

fig.suptitle("Learned parameters")
fig.tight_layout()
fig.savefig(plot_dir / "parameters.png", dpi=150)
plt.show()

# 3. Loss curves
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(epochList, dataLossList, label="Data loss")
ax.semilogy(epochList, physLossList, label="Physics loss (training points)")
ax.semilogy(epochList, collocLossList, label="Physics loss (collocation points)")
ax.set_xlabel("epoch")
ax.set_ylabel("loss (batch average)")
ax.set_title("Training losses")
ax.legend()
fig.tight_layout()
fig.savefig(plot_dir / "losses.png", dpi=150)
plt.show()

# 4. Final error and physics residual against time
Tplot_eval = Tplot.clone().detach().requires_grad_(True)
Xpred_plot = model(Tplot_eval)
dx_plot = torch.autograd.grad(Xpred_plot, Tplot_eval, torch.ones_like(Xpred_plot), create_graph=True)[0]
d2x_plot = torch.autograd.grad(dx_plot, Tplot_eval, torch.ones_like(dx_plot))[0]
residual_plot = d2x_plot + model.gamma*dx_plot + (model.omega**2)*Xpred_plot

Xpred_plot_np = Xpred_plot.detach().cpu().numpy().ravel()
residual_plot_np = residual_plot.detach().cpu().numpy().ravel()

fig, (ax_err, ax_res) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
ax_err.semilogy(Tplot_np, np.abs(Xpred_plot_np - Xtrue_plot))
ax_err.set_ylabel("|x_pred - x_true| / m")
ax_err.set_title("Final error against the true (noise-free) solution")

ax_res.semilogy(Tplot_np, residual_plot_np**2)
ax_res.set_ylabel("physics residual²")
ax_res.set_xlabel("time / s")

for ax in (ax_err, ax_res):
    ax.axvspan(train_end, Tplot_np[-1], color="grey", alpha=0.15)
fig.tight_layout()
fig.savefig(plot_dir / "error_and_residual.png", dpi=150)
plt.show()
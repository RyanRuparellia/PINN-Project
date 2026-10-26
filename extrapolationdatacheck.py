"""Plot the reconstruction training data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

data_file = Path(__file__).resolve().parent / "reconstruction" / "train.csv"
trainData = pd.read_csv(data_file).sort_values("time")

plt.scatter(trainData["time"], trainData["displacement"], s=2, label="Training data")
plt.xlabel("time / s")
plt.ylabel("displacement / m")
plt.title("Reconstruction training data")
plt.legend()
plt.show()

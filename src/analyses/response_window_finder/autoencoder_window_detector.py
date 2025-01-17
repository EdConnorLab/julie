import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from clat.intan.channels import Channel
from torch.utils.data import DataLoader, TensorDataset
from monkey_names import Zombies, BestFrans, Instigators, StrangerThings
from spike_count import get_spike_counts_for_time_chunks
from spike_rate_computation import get_raw_data_and_channels_from_files


# Function to trim lists to the minimum length
def trim_list(lst):
    return lst[:min_length]

zombies = [member.value for name, member in Zombies.__members__.items()]
del zombies[6]
del zombies[-1]
bestfrans = [member.value for name, member in BestFrans.__members__.items()]
del bestfrans[-1]
instigators = [member.value for name, member in Instigators.__members__.items()]
del instigators [-1]
strangerthings = [member.value for name, member in StrangerThings.__members__.items()]
del strangerthings[-1]
monkeys = zombies + strangerthings + bestfrans + instigators

date = "2023-09-26"
round_no = 1
raw_unsorted_data, valid_channels, sorted_data = get_raw_data_and_channels_from_files(date, round_no)
time_chunk_size = 0.02  # in sec
spike_counts = get_spike_counts_for_time_chunks(monkeys, raw_unsorted_data, valid_channels, time_chunk_size)
spike_counts_zombies = get_spike_counts_for_time_chunks(['94B'], raw_unsorted_data, valid_channels, time_chunk_size)
# min_length = spike_counts.applymap(len).min().min()
# print(min_length)
min_length = 100
trimmed_df = spike_counts.map(trim_list)
zombies_trimmed = spike_counts_zombies.map(trim_list)

# Combine lists in each row into a 2D array
combined_arrays = trimmed_df.apply(lambda row: np.array(row.tolist()), axis=1)
combined_test_arrays = zombies_trimmed.apply(lambda row: np.array(row.tolist()), axis=1)

train_data = combined_arrays[Channel.C_027]
train_data = train_data.astype(np.float32)
print(train_data.shape)
test_data = combined_test_arrays[Channel.C_027]
test_data = test_data.astype(np.float32)
print(test_data.shape)

# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 64),  # Input features to hidden size
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 100),  # Compressed form back to original size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for data in train_loader:
        inputs = data[0]
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Anomaly detection on the test data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_loader = DataLoader(test_data, batch_size=20)

    for data in test_loader:
        inputs = data[0]
        reconstructions = model(inputs)
        print("Inputs shape:", inputs.shape)
        print("Reconstructions shape:", reconstructions.shape)

        mse = torch.mean((inputs - reconstructions) ** 2)
        mse_threshold = torch.quantile(mse, 0.99)  # Setting a threshold as the 99th percentile of MSE
        outliers = mse > mse_threshold
        print("Anomaly detected indices in batch:", torch.where(outliers)[0].numpy())

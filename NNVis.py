import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1) Create sample data (x, sin(x))
x_data = np.random.uniform(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
x_data = np.sort(x_data)
y_data = (np.sin(0.5 * x_data)
          + np.sin(x_data)
          + 0.5 * np.sin(2 * x_data)
          + np.random.uniform(-0.1, 0.1, size=x_data.shape)) # this last line is just to introduce some noise

# Convert to torch tensors
# We'll keep them as float32 to match typical PyTorch defaults
X = torch.tensor(x_data, dtype=torch.float32)
Y = torch.tensor(y_data, dtype=torch.float32)

# 2) Define a small neural network
#    Let's just do a 1-hidden-layer MLP with [Leaky]ReLU/Tanh/Sigmoid/etc.
model = nn.Sequential(
    nn.Linear(1, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 1)
)

# 3) Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # can try Adam, SGD, RMSprop, etc.

# 4) Training + Animation setup
plt.ion()  # turn interactive mode on
fig, ax = plt.subplots()

num_epochs = 2001
for epoch in range(num_epochs):
    # 4a) Forward pass
    y_pred = model(X)

    # 4b) Compute loss
    loss = criterion(y_pred, Y)  # defined above as MSE

    # 4c) Backprop and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 4d) Occasionally update our plot
    if epoch % 10 == 0:  # lower this number for slower more granular animation, larger for bigger steps (& faster)
        ax.clear()
        # Plot the original data
        ax.scatter(x_data, y_data, color='#fd0400', label="True Data", s=10)

        # Plot the network's prediction (convert to numpy)
        # We'll pass a set grid of points for a smooth line (gonna call it "test" cause that's basically what it is):
        test_x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)  # essentially creating datapoints for test set
        test_X = torch.tensor(test_x, dtype=torch.float32)
        with torch.no_grad():
            test_pred = model(test_X).numpy()

        ax.plot(test_x, test_pred, label="NN Prediction")

        ax.set_title(f"Epoch {epoch}, Loss: {loss.item(): .6f}")
        ax.legend()
        plt.pause(0.05)  # lower for faster animation, higher for slower (this is the pause in seconds between "frames")

plt.ioff()
plt.show()

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred


def get_params(w, b):
    return w[0][0].item(), b[0].item()


def plot_fit(X, Y, w, b):
    w1, b1 = get_params(w, b)
    x1 = np.array([-30, 30])
    y1 = w1 * x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X, Y)
    plt.show()


def main():
    # Dataset creation
    X = torch.randn(100, 1) * 10
    Y = X + 3 * torch.randn(100, 1)

    model = LR(1, 1)
    [w, b] = model.parameters()
    plot_fit(X, Y, w, b)

    criterion = nn.MSELoss()    # set loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)    # set SGD algorithm

    # Training
    epochs = 100
    losses = []
    for i in range(epochs):
        y_pred = model.forward(X)
        loss = criterion(y_pred, Y)
        print("epoch:", i, "loss:", loss.item())
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Plot the improvement in loss per epoch
    plt.plot(range(epochs), losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.show()
    plot_fit(X, Y, w, b)


if __name__ == '__main__':
    main()

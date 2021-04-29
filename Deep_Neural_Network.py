import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_size, H1, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, H1)
        self.linear2 = nn.Linear(H1, output_size)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        pred = torch.sigmoid(self.linear2(pred))
        return pred

    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0


def create_dataset():
    n_pts = 500
    X, Y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
    # random_state makes the dataset persistent
    return X, Y


def scatter_plot(X, Y):
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1])
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1])
    plt.show()


def plot_decision_boundary(X, Y, model):
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    pred_func = model.forward(grid)
    z = pred_func.view(xx.shape).detach().numpy()
    plt.contourf(xx, yy, z)
    scatter_plot(X, Y)


def main():
    X, Y = create_dataset()
    x_data = torch.Tensor(X)
    y_data = torch.Tensor(Y.reshape(500, 1))
    torch.manual_seed(2)    # make the model consistent
    model = Model(2, 4, 1)
    print(model.parameters())
    # Model training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    epochs = 1000
    losses = []
    for i in range(epochs):
        y_pred = model.forward(x_data)
        loss = criterion(y_pred, y_data)
        print("epoch:", i, "loss:", loss.item())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Plot the improvement in loss per epoch
    plt.plot(range(epochs), losses)
    plt.title('Loss error')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.show()

    # Model testing
    x = 0.025
    y = 0.025
    point = torch.Tensor([x, y])
    prediction = model.predict(point)
    plt.plot([x], [y], marker='o', markersize=10, color='red')
    print("Prediction is", prediction)
    plot_decision_boundary(X, Y, model)


if __name__ == '__main__':
    main()

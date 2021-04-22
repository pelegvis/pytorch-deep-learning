import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred

    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0


def create_dataset():
    n_pts = 100
    centers = [[-0.5, 0.5], [0.5, -0.5]]
    X, Y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)
    # random_state makes the dataset persistent
    return X, Y


def get_params(w, b):
    w1, w2 = w.view(2)
    return w1.item(), w2.item(), b[0].item()


def plot_fit(X, Y, w, b, title):
    w1, w2, b1 = get_params(w, b)
    x1 = np.array([-2.0, 2.0])
    x2 = (w1 * x1 + b1) / -w2
    plt.title(title)
    plt.plot(x1, x2, 'r')
    scatter_plot(X, Y)


def scatter_plot(X, Y):
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1])
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1])
    plt.show()


def main():
    X, Y = create_dataset()
    x_data = torch.Tensor(X)
    y_data = torch.Tensor(Y.reshape(100, 1))
    torch.manual_seed(2)    # make the model consistent
    model = Model(2, 1)
    [w, b] = model.parameters()
    plot_fit(X, Y, w, b, "Initial Model")
    # Model training
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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
    plot_fit(X, Y, w, b, "Trained Model")

    # Model testing
    point1 = torch.Tensor([1.0, -1.0])
    point2 = torch.Tensor([-1.0, 1.0])
    plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
    plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')
    print("Red point positive probability = {}".format(model.forward(point1).item()))
    print("Black point positive probability = {}".format(model.forward(point2).item()))
    print("Red point in class {}".format(model.predict(point1)))
    print("Black point in class {}".format(model.predict(point2)))
    plot_fit(X, Y, w, b, "Testing Data")


if __name__ == '__main__':
    main()

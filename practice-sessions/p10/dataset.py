import numpy as np
import matplotlib.pyplot as plt



def get_data():
    n0 = 50
    mean0 = [0, 0]
    cov0 = [[1, 0], [0, 12]]

    n1 = 40
    mean1 = [6, 10]
    cov1 = [[1, 0], [0, 12]]

    neg = np.hstack([np.random.multivariate_normal(mean0, cov0, n0), np.zeros((n0, 1))])
    pos = np.hstack([np.random.multivariate_normal(mean1, cov1, n1), np.ones((n1, 1))])

    data = np.vstack([neg, pos])
    return data



def plot_data(data):
    plt.figure()
    plt.axis('equal')
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    plt.show()

def euclidean_distance(x0, x1):
    x0 = np.array(x0)
    x1 = np.array(x1)
    diff = x0 - x1
    return np.sqrt(np.sum(np.square(diff)))

def nearest_neighbor(data, x):
    dists = [euclidean_distance(vec[:-1], x) for vec in data]
    argmin = np.argmin(dists)
    x_class = data[argmin][-1]
    return x_class


def plot_nn(data, coords, class_func):
    plt.figure()
    plt.axis('equal')
    classes = [class_func(data, coord) for coord in coords]
    neg_x = [coords[i,0] for i in range(len(coords)) if classes[i] == 0]
    neg_y = [coords[i,1] for i in range(len(coords)) if classes[i] == 0]
    pos_x = [coords[i, 0] for i in range(len(coords)) if classes[i] == 1]
    pos_y = [coords[i, 1] for i in range(len(coords)) if classes[i] == 1]

    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])

    plt.scatter(neg_x, neg_y, c='red')
    plt.scatter(pos_x, pos_y, c='blue')

    plt.show()

data = get_data()


n0 = 20
mean0 = [3, 5]
cov0 = [[2, 0], [0, 2]]

examples = np.random.multivariate_normal(mean0, cov0, n0)
print(examples)

plot_nn(data, examples,nearest_neighbor)

# nearest_neighbor(data, [10, 10])



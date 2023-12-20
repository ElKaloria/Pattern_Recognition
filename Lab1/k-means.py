from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (6, 6)
plt.style.use('ggplot')


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def initialization(n: int):
    f1 = []
    f2 = []
    for i in range(n):
        f1.append(np.random.randint(0, 100))
        f2.append(np.random.randint(0, 100))

    x = np.array(list(zip(f1, f2)))
    return x, f1, f2

def k_means(n: int, k: int):

    X, f1, f2 = initialization(n)
    # Set centroids
    C_x = np.random.randint(0, np.max(X)-20, size=k)
    C_y = np.random.randint(0, np.max(X)-20, size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    print("Initial Centroids")
    print(C)

    # Plotting along with the Centroids
    plt.scatter(f1, f2, c='#050505', s=7)
    plt.scatter(C_x, C_y, marker='*', s=200, c='g')
    plt.show()

    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)

    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    plt.show()


def simple_claster_alg(n, teta):
    X, f1, f2 = initialization(n)
    # Set centroids
    C = [X[0]]

    teta_mas = [teta]
    clusters = np.zeros(len(X))
    for i in range(1, n):
        distances = dist(X[i], C)
        if (distances > teta_mas).all():
            C.append(X[i])
            teta_mas.append(teta)
            clusters[i] = len(C) - 1
        else:
            cluster = np.argmin(distances)
            clusters[i] = cluster

    fig, ax = plt.subplots()
    for i in range(len(C)):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=np.random.rand(3,))
    for i in range(len(C)):
        ax.scatter(C[i][0], C[i][1], marker='*', s=200, c='#050505')
    plt.show()





if __name__ == "__main__":
    #k_means(3000, 5)
    simple_claster_alg(3000, 50)



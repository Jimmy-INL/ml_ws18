import numpy as np

def load_data(file_name):
    """
    Loads the data stored in file_name into a numpy array.
    """
    result = None
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
        result = np.loadtxt(lines, dtype=float)
    return result


assert load_data('em_normdistdata.txt').shape == (200,) , "The data was not properly loaded."


###########################


import numpy as np
class NormPDF():
    """
    A representation of the probability density function of the normal distribution
    for the EM Algorithm.
    """

    def __init__(self, mu=0, sigma=1, alpha=1):
        """
        Initializes the normal distribution with mu, sigma and alpha.
        The defaults are 0, 1, and 1 respectively.
        """
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha


    def __call__(self, x):
        """
        Returns the evaluation of this normal distribution at x.
        Does not take alpha into account!
        """
        return np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) / (np.sqrt(np.pi * 2) * self.sigma)


    def __repr__(self):
        """
        A simple string representation of this instance.
        """
        return 'NormPDF({self.mu:.2f},{self.sigma:.2f},{self.alpha:.2f})'.format(self=self)


a = NormPDF()             # No parameters: mu = 0, sigma = 1, alpha = 1
b = NormPDF(1)            # mu = 1, sigma = 1, alpha = 1
c = NormPDF(1, alpha=0.4) # skips sigma but sets alpha, thus: mu = 1, sigma = 1, alpha = 0.4
d = NormPDF(0, 0.5)       # mu = 0, sigma = 0.5, alpha = 1
e = NormPDF(0, 0.5, 0.9)  # mu = 0, sigma = 0.5, alpha = 0.9


# normpdf = NormPDF()
# print(normpdf(0))
# print(normpdf(0.5))
# print(normpdf(np.linspace(-2, 2, 10)))

# normpdf1 = NormPDF()
# normpdf2 = NormPDF(1, 0.5, 0.9)
# print(normpdf1)
# print([normpdf1, normpdf2])

# normpdf1 = NormPDF()
# print(normpdf1)
# print(normpdf1(np.linspace(-2, 2, 10)))
#
# normpdf1.mu = 1
# normpdf1.sigma = 2
# normpdf1.alpha = 0.9
# print(normpdf1)
# print(normpdf1(np.linspace(-2, 2, 10)))


def initialize_EM(data, num_distributions):
    """
    Initializes the EM algorithm by calculating num_distributions NormPDFs
    from a random partitioning of data. I.e., the data set is randomly
    divided into num_distribution parts, and each part is used to initialize
    mean, standard deviation and alpha parameter of a NormPDF object.

    Args:
        data (array): A collection of data.
        num_distributions (int): The number of distributions to return.

    Returns:
        A list of num_distribution NormPDF objects, initialized from a
        random partioning of the data.
    """
    gaussians = None

    data_partitions = [[] for i in range(num_distributions)]
    for point in data:
        ind = np.random.randint(0, num_distributions)
        data_partitions[ind].append(point)

    # means = np.mean(data_partitions_array, axis = 0, keepdims=True)

    # Because mean and co do not work on a list of lists of different shapes:
    means = [np.mean(part) for part in data_partitions]
    std_devs = [np.std(part) for part in data_partitions]
    alphas = [len(part)/len(data) for part in data_partitions]

    gaussians = [NormPDF(mu=means[i], sigma=std_devs[i], alpha=alphas[i]) for i in range(num_distributions)]
    return gaussians


normpdfs_ = initialize_EM(np.linspace(-1, 1, 100), 2)
assert len(normpdfs_) == 2, "The number of initialized distributions is not correct."
# 1e-10 is 0.0000000001
assert abs(1 - sum([normpdf.alpha for normpdf in normpdfs_])) < 1e-10, "Sum of all alphas is not 1.0!"


#############################

def expectation_step(gaussians, data):
    """
    Performs the expectation step of the EM.

    Args:
        gaussians (list): A list of NormPDF objects.
        data (array): The data vector.

    Returns:
        An array of shape (len(data), len(gaussians))
        which contains normalized likelihoods for each sample
        to denote to which of the normal distributions it
        most likely belongs to.
    """
    expectation = None

    expectation_nn = np.zeros(shape=(len(data), len(gaussians)))
    for gauss_idx, gaussian in enumerate(gaussians):
        expectation_nn[:, gauss_idx] = gaussian.alpha*gaussian(data)

    # Normalizing all expectations
    expectation = np.array([expect/np.sum(expect) for expect in expectation_nn])

    return expectation


assert expectation_step([NormPDF(), NormPDF()], np.linspace(-2, 2, 100)).shape == (100, 2), "Shape is not correct!"

#########################################

def maximization_step(gaussians, data, expectation):
    """
    Performs the maximization step of the EM.
    Modifies the gaussians by updating their mus and sigmas.

    Args:
        gaussians (list): A list of NormPDF objects.
        data (array): The data vector.
        expectation (array): The expectation values for data element
            (as computed by expectation_step()).

    Returns:
        A numpy array of absolute changes in any mu or sigma,
        that means the returned array has twice as many elements as
        the supplied list of gaussians.
    """
    changes = []

    for idx, gaussian in enumerate(gaussians):
        mu_idx_old = gaussian.mu
        mu_idx_new = 1/sum(expectation[:, idx]) * np.sum(np.multiply(expectation[:, idx], data))

        # Update the expectations here already?
        sigma_idx_old = gaussian.sigma
        sum1 = np.sum(expectation[:, idx])
        sum2 = np.sum(np.multiply(expectation[:, idx], (data - mu_idx_new)**2))
        sigma_idx_new = np.sqrt(1/sum1*sum2)

        #changes.append(np.abs([mu_idx_new-mu_idx_old, sigma_idx_new-sigma_idx_old]))
        changes.append(np.abs(mu_idx_new - mu_idx_old))
        changes.append(np.abs(sigma_idx_new - sigma_idx_old))

        gaussian.mu = mu_idx_new
        gaussian.sigma = sigma_idx_new

    return np.array(changes)

maximization_step([NormPDF(), NormPDF()], np.linspace(-2, 2, 100), expectation_step([NormPDF(), NormPDF()], np.linspace(-2, 2, 100)))


##################


import time
import itertools

import numpy as np
import matplotlib.pyplot as plt

# Sets the random seed to a fix value to make results consistent
np.random.seed(2)

colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
figure, axis = plt.subplots(1)
axis.set_xlim(-5, 5)
axis.set_ylim(-0.2, 4)
axis.set_title('Intermediate Results')
plt.figure('Final Result')


def plot_intermediate_result(gaussians, data, mapping):
    """
    Gets a list of gaussians and data input. The mapping
    parameter is a list of indices of gaussians. Each value
    corresponds to the data value at the same position and
    maps this data value to the proper gaussian.
    """
    x = np.linspace(-5, 5, 100)
    if len(axis.lines):
        for j, N in enumerate(gaussians):
            axis.lines[j * 2].set_xdata(x)
            axis.lines[j * 2].set_ydata(N(x))
            axis.lines[j * 2 + 1].set_xdata(data[mapping == j])
            axis.lines[j * 2 + 1].set_ydata([0] * len(data[mapping == j]))
    else:
        for j, N in enumerate(gaussians):
            axis.plot(x, N(x), data[mapping == j], [0] * len(data[mapping == j]), 'x', color=next(colors), markersize=5)
    figure.canvas.draw()
    time.sleep(0.5)
    plt.show()


# Perform the initialization.
data = load_data('em_normdistdata.txt')
gaussians = initialize_EM(data, 3)

# Loop until the changes are small enough.
eps = 0.05
changes = [float('inf')] * 2
while max(changes) > eps:
    # Iteratively apply the expectation step, followed by the maximization step.

    print("banana")
    expectation = expectation_step(gaussians, data)
    changes = maximization_step(gaussians, data, expectation)

    # Optional: Calculate the parameters to update the plot and call the function to do it.
    plot_intermediate_result(gaussians, data, np.argmax(expectation, 1))


# Plot your final result and print the final parameters.
# YOUR CODE HERE
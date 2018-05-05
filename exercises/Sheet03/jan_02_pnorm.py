import numpy as np


def pnorm(x, p):
    """
    Calculates the p-norm of x.

    Args:
        x (array): the vector for which the norm is to be computed.
        p (float): the p-value (a positive real number).

    Returns:
        The p-norm of x.
    """
    x = np.abs(x)
    result = np.power(np.sum(np.power(x, p)), 1/p)

    return result


# 1e-10 is 0.0000000001
assert pnorm(1, 2)      - 1          < 1e-10 , "pnorm is incorrect for x = 1, p = 2"
assert pnorm(2, 2)      - 2          < 1e-10 , "pnorm is incorrect for x = 2, p = 2"
assert pnorm([2, 1], 2) - np.sqrt(5) < 1e-10 , "pnorm is incorrect for x = [2, 1], p = 2"
assert pnorm(2, 0.5)    - 2          < 1e-10 , "pnorm is incorrect for x = 2, p = 0.5"

def pdist(x0, x1, p):
    """
    Calculates the distance between x0 and x1
    using the p-norm.

    Arguments:
        x0 (array): the first vector.
        x1 (array): the second vector.
        p (float): the p-value (a positive real number).

    Returns:
        The p-distance between x0 and x1.
    """
    result = None
    if not isinstance(x0, (int, float)):
        assert len(x0)==len(x1), "Both have to have the same length"
    x0 = np.array(x0)
    x1 = np.array(x1)

    diff_abs = np.abs(x0-x1)

    return pnorm(diff_abs, p)

# 1e-10 is 0.0000000001
assert pdist(1, 2, 2)           - 1          < 1e-10 , "pdist is incorrect for x0 = 1, x1 = 2, p = 2"
assert pdist(2, 5, 2)           - 3          < 1e-10 , "pdist is incorrect for x0 = 2, x1 = 5, p = 2"
assert pdist([2, 1], [1, 2], 2) - np.sqrt(2) < 1e-10 , "pdist is incorrect for x0 = [2, 1], x1 = [1, 2], p = 2"
assert pdist([2, 1], [0, 0], 2) - np.sqrt(5) < 1e-10 , "pdist is incorrect for x0 = [2, 1], x1 = [0, 0], p = 2"
assert pdist(2, 0, 0.5)         - 2          < 1e-10 , "pdist is incorrect for x0 = 2, x1 = 0, p = 0.5"


import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter

color = ColorConverter()
figure_norms = plt.figure('p-norm comparison')

# create the linspace vector
ls = np.linspace(start=-100, stop=100, num=50)

assert len(ls) == 50 , 'ls should be of length 50.'
assert (min(ls), max(ls)) == (-100, 100) , 'ls should range from -100 to 100, inclusively.'

for i, p in enumerate([1/8, 1/4, 1/2, 1, 1.5, 2, 4, 8, 128]):

    # Create a numpy array containing useful values instead of zeros.
    # Iterate over all values in ls for x as well as for y and compute the pnorm in the same step
    data = np.array([[x, y, pnorm([x, y], p)] for x in ls for y in ls])
    data[:, 2] = data[:, 2] / np.max(data[:, 2])

    assert all(data[:,2] <= 1), 'The third column should be normalized.'

    # Plot the data.
    colors = [color.to_rgb((1, 1-a, 1-a)) for a in data[:,2]]
    a = plt.subplot(3, 3, i + 1)
    plt.scatter(data[:,0], data[:,1], marker='.', color=colors)
    a.set_ylim([-100, 100])
    a.set_xlim([-100, 100])
    a.set_title('{:.3g}-norm'.format(p))
    a.set_aspect('equal')
    plt.tight_layout()
    figure_norms.canvas.draw()
plt.show()
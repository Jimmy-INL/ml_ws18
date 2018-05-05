import numpy as np
import matplotlib.pyplot as plt

# generate dataset
data = list(zip(np.random.uniform(size=100), np.random.normal(size=100)))
data += list(zip(np.random.uniform(size=10), np.random.normal(0, 10, size=10)))
data = np.array(data)
outliers = []

# just to check if everything is pretty
fig_rosner_data = plt.figure('The Dataset')
plt.scatter(data[:,0], data[:,1], marker='x')
plt.axis([0, 1, -20, 20])
fig_rosner_data.canvas.draw()

#Plot all the data
plt.show()

# Now find the outliers, add them to 'outliers', and remove them from 'data'.


def rosner_test(data, k_largest = 1, z_threshold=3):
    data_center = np.median(data[:, 1])
    data_std_dev = np.std(data[:, 1])

    # sort already for absolute y-coordinate to have the biggest z-values ones in the front
    # enables us to later reference the first num_outliers items
    data_sorted = sorted(data, reverse=True, key=lambda item: np.abs(item[1]))

    # compute all the z-values
    z_values_data = [[x, np.abs((y-data_center))/data_std_dev] for x, y in data_sorted]

    # Threshold all z-values by the given threshold
    z_values_thresholded = [point for point in z_values_data if point[1] > z_threshold]
    print(f'Found {len(z_values_thresholded)} outliers in the data')

    # either take the k_largest outliers, or if only fewer than that found, the number of found outliers
    num_outliers = min(k_largest, len(z_values_thresholded))

    # first num_outliers are outliers, the rest is real data
    outliers = np.array(data_sorted[:num_outliers])
    data_clean = np.array(data_sorted[num_outliers:])

    # We are done if no new outliers were found
    if num_outliers == 0:
        return np.array(data_clean), np.array([])
    else:
        # recursively call this function to start the process on remaining data
        clean_data, outliers_new = rosner_test(data_clean, k_largest=k_largest, z_threshold=z_threshold)
        return np.array(clean_data), np.array(list(outliers) + list(outliers_new))


data, outliers = rosner_test(data)

# plot results
outliers = np.array(outliers)
fig_rosner = plt.figure('Rosner Result')
plt.scatter(data[:,0], data[:,1], c='b', marker='x', label='cleared data')
plt.scatter(outliers[:,0], outliers[:,1], c='r', marker='x', label='outliers')
plt.axis([0, 1, -20, 20])
plt.legend(loc='lower right');
fig_rosner.canvas.draw()
plt.show()

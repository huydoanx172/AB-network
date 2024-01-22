import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.cluster import KMeans

# Get euclidean distance
def dist(a, b):
    return np.linalg.norm(a-b)

# Convert distance to dataframe
def get_distance(sample, centroidA, centroidB):
    distA = sample.apply(lambda x: dist(x, centroidA), axis=1)
    distB = sample.apply(lambda x: dist(x, centroidB), axis=1)
    return(pd.DataFrame({'A': distA, 
                         'B': distB}))

# Get assignment as a pd.Series
def get_assign(dist_df):
    return dist_df.idxmin(axis=1).rename('assign')

# Get new centroid coordinates
def get_new_centroids(sample, assign):
    return pd.concat([sample, assign], axis=1).groupby(['assign']).agg('mean')

def kmeans_demo():
    
    sample = pd.read_csv('clustering.csv').sample(5, random_state=26).reset_index(drop=True).round(2)

    centroidA = sample.iloc[2,:]
    centroidB = sample.iloc[4,:]
    centroids = pd.DataFrame.from_records([centroidA, centroidB], index=['A', 'B'])

    max_num_iter = 3
    num_iter = 1
    colormap = {'A': 'red', 'B': 'green'}

    for iteration in range(max_num_iter):
        display('Iteration', num_iter)

        display('Centroid at start of iteration')
        display(centroids)

        distance = get_distance(sample, centroidA, centroidB)
        display('Distance matrix')
        display(distance)

        assign = get_assign(distance)
        display('Cluster assignment')
        display(assign)

        centroids = get_new_centroids(sample, assign)
        display('Centroid at end of iteration')
        display(centroids)

        centroidA = centroids.loc['A']
        centroidB = centroids.loc['B']
        num_iter +=1

        # Plotting the obtined clusters of each iteration 

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.scatter(sample['time_on_site'], 
                   sample['quantity'], 
                   sample['movie_rating'],
                  c=assign.map(colormap))
        ax.scatter(centroids['time_on_site'], 
                   centroids['quantity'], 
                   centroids['movie_rating'],
                  marker='x', c=['red','green'])
        ax.set_ylabel('quantity')
        ax.set_xlabel('time_on_site')
        ax.set_zlabel('movie_rating')

        for i in range(5):
            ax.text(sample['time_on_site'][i], 
                    sample['quantity'][i], 
                    sample['movie_rating'][i], 
                    i)
        plt.show()

        display('-'*100)
        display('\n')

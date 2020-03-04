import sklearn
import cv2
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import Counter
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path


class ImageClassifier:

    def __init__(self, n_clusters, target):
        self._n_clusters = n_clusters
        self._colorspaces = {
            cv2.COLOR_BGR2HSV: cv2.COLOR_HSV2BGR,
            cv2.COLOR_BGR2LAB: cv2.COLOR_LAB2BGR,
            cv2.COLOR_BGR2HLS: cv2.COLOR_HLS2BGR,
        }
        self._img = cv2.imread(target)
        self._rows,self._cols,_ = self._img.shape


    def run(self, dst):
        df = self.get_dataframe(colorspace=cv2.COLOR_BGR2HSV)
        cluster_map = self.run_kmeans(df, [0])
        clusters = self.get_clusters(cluster_map)

        cmp = lambda pixel: int(pixel[0])
        clusters = self.sort_clusters(clusters, cmp, color_sort=cv2.COLOR_BGR2LAB)
        res = self.merge_clusters(clusters, lambda cluster: sum(cluster[0][0]))
        cv2.imwrite(dst, res)

    def get_dataframe(self, colorspace=None):
        """
        Function to get a dataframe from an image's data.

        Return value (pandas.DataFrame):
            dataframe with every pixel's information (3 channels).
            pixels are extracted left to right, top to bottom.

        Parameters:
            img_mat (cv2.Mat): image to extract data from (must be in BGR colorspace)
            colorspace (cv2.COLOR_BGR2*): used if you want to form dataframe from other colorspace 
        """

        data = {'val1':[], 'val2':[], 'val3':[]}
        
        img = self._img.copy()
        # Convert image to desired colorspace
        if colorspace is not None:
            img = cv2.cvtColor(img, colorspace).astype(np.uint8)

        for i in range(self._rows):
            for j in range(self._cols):
                data['val1'].append(img[i][j][0])
                data['val2'].append(img[i][j][1])
                data['val3'].append(img[i][j][2])
        df = pd.DataFrame(data=data)

        return df


    def get_optimal_n_clusters(self, dataframe, keys):
        max_n = 0
        max_score = 0
        x = dataframe.iloc[:, keys].values
        print("Finding optimal cluster count...")
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, n_jobs=-1)
            preds = kmeans.fit_predict(x)
            print("start silhouette")
            score = silhouette_score(x, preds)
            print("end silhouette")
            if (score > max_score):
                max_n = n_clusters
                max_score = score
            print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
        print("Optimal cluster count is {}".format(max_n))
        return max_n

    def run_kmeans(self, dataframe, keys):
        """
        Run kmeans from dataframe and returns clustering information.

        Return value (list):
            cluster id for each entry in the dataframe

        Parameters:
            dataframe (pandas.DataFrame): dataframe to run kmeans on
            keys (list): indexes of the dataframe's columns used to run kmeans
        """
        if self._n_clusters == -1:
            self._n_clusters = self.get_optimal_n_clusters(dataframe, keys)

        kmeans = KMeans(n_clusters=self._n_clusters, n_init=10, max_iter=300, n_jobs=-1)
        x = dataframe.iloc[:, keys].values

        y = kmeans.fit_predict(x)
        return y

    def get_clusters(self, cluster_map):
        """
        Extract clusters from image

        Return value (list):
            List containing each cluster as a list of pixels.

        Parameters:
            n_clusters (int): Number of clusters to use
            img_mat (cv2.Mat): img to extract pixels from
            cluster_map (list): list containing cluster id for each pixel of img_mat (left to right, top to bottom)
        """

        groups = [[] for i in range(self._n_clusters)]
        
        for i in range(self._rows):
            for j in range(self._cols):
                group_id = cluster_map[i * self._cols + j]
                groups[group_id].append(self._img[i][j])

        return groups


    def sort_clusters(self, clusters, comparator, color_sort=None):
        """
        Sorts each cluster with a custom comparator

        Return value (list): 
            list of sorted np.arrays

        Parameters:
            clusters (list): list of clusters to sort
            comparator (lambda x): comparator function to use to sort clusters
            colorspace: in which colorspace to be to sort the clusters
        """
        avg = [np.zeros((3), dtype=np.uint64) for i in range (self._n_clusters)]

        for i in range(len(clusters)):
            cluster = clusters[i]

            cluster = np.reshape(cluster, (1, len(cluster), 3)) # Reshape cluster so it fits cv2.Mat format, allowing to change its colorspace
            if color_sort is not None:                         # Convert cluster to desired colorspace
                cluster = cv2.cvtColor(cluster, color_sort).astype(np.uint8)

            cluster[0] = np.array(sorted(cluster[0], key=comparator)).astype(np.uint8)    # Sort cluster with specified comparator
            if color_sort is not None:                          # Convert cluster back to BGR
                cluster = cv2.cvtColor(cluster, self._colorspaces[color_sort]).astype(np.uint8)
            clusters[i] = cluster
        
        return clusters

    def merge_clusters(self, clusters, comparator):
        """
        Merges all clusters into one image. Clusters are places from left to right, top to bottom.

        Return value (cv2.Mat):
            cv2 image with merged clusters
        
        Parameters:
            clusters (list): list of clusters (np.arrays) (shape: (1, x, 3))
            shape (2 value tuple): desired image shape (rows, cols)
        """
        
        res = np.zeros((self._rows * self._cols, 3), dtype=np.uint8)
        merge_index = 0

        clusters = sorted(clusters, key=comparator)
        for cluster in clusters:
            res[merge_index:merge_index+len(cluster[0])] = cluster[0]
            merge_index = merge_index + len(cluster[0])
        
        res = np.reshape(res, (self._rows, self._cols, 3))
        return res






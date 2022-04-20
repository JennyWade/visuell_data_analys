import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn import decomposition
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh
from sklearn import metrics
import umap.umap_ as UMAP


class Kmean:
    def __init__(self):
        self.data = load_breast_cancer(as_frame=True)

    def createDataframe(self):
        self.df = pd.DataFrame(np.c_[self.data['data'], self.data['target']], columns= np.append(self.data['feature_names'], ['target']))

    def standardise(self):
        self.data = self.df[0:569]
        self.data = np.asarray(self.data)
        scaler = StandardScaler()
        scaler.fit(self.data)
        self.scaled_data = scaler.transform(self.data)

    def elbow(self):
        no_clusters = []
        K = range(1,10)
        for k in K:
            kmean = KMeans(n_clusters=k)
            kmean.fit(self.scaled_data)
            no_clusters.append(kmean.inertia_)
        
        plt.figure(figsize=(10,5))
        plt.plot(K, no_clusters)
        plt.show()

    def kmean(self):
        model = KMeans(n_clusters=2)
        model.fit(self.scaled_data)
        self.labels = model.predict(self.scaled_data)
        x1 = self.scaled_data[:,0]
        y1 = self.scaled_data[:,1]
        x2 = self.scaled_data[:,0]
        y2 = self.scaled_data[:,1]
        fig, ax = plt.subplots(1, 2, figsize=(16,8))
        ax[0].scatter(x1, y1, c=self.labels)
        ax[1].scatter(x2, y2, c=self.df['target'], cmap=plt.cm.Set1)
        ax[0].set_title('Without KMeans')
        ax[1].set_title('With KMeans')
        plt.show()

    def umap(self):
        umap = UMAP()
        umap_fitted = umap.fit_transform(self.scaled_data)
        plt.scatter(umap_fitted[:,0], umap_fitted[:,1])
        plt.show()


#class PCA:
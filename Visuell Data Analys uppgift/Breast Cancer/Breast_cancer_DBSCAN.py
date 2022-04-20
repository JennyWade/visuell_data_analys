from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_breast_cancer
from umap.umap_ import UMAP
import plotly.express as px

class DBScan:
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

    def nearestNeighbor(self):
        nn = NearestNeighbors(n_neighbors=2)
        nbrs = nn.fit(self.scaled_data)
        self.distances, self.indices = nbrs.kneighbors(self.scaled_data)

    def nearestNeighborGraph(self):
        self.distances = np.sort(self.distances, axis=0)
        self.distances = self.distances[:,1]
        plt.figure(figsize=(10,10))
        plt.plot(self.distances)
        plt.title('Graph over nearest neighbor distance')
        plt.xlabel('Data points sorted by distance')
        plt.ylabel('Epsilon')
        plt.show()   

    def dbscan(self):
        db = DBSCAN(eps = 5, min_samples = 62).fit(self.scaled_data)
        self.labels = db.labels_

    def umap(self):
        umap = UMAP(random_state=2)
        self.umap_fitted = umap.fit_transform(self.scaled_data)
        fig = px.scatter(self.umap_fitted, x=0, y=1, color=self.labels, title='UMAP')
        fig.show()

    def pca(self):
        self.pca = PCA(n_components=15)
        self.pca_data = self.pca.fit_transform(self.scaled_data)
        db = DBSCAN(eps = 5, min_samples = 62).fit(self.pca_data)
        self.pca_labels = db.labels_

    def pcaUmap(self):
        umap_pca = UMAP(random_state=0)
        self.umap_pca_fitted = umap_pca.fit_transform(self.pca_data)
        fig = px.scatter(self.umap_pca_fitted, x=0, y=1, color=self.pca_labels, title='UMAP with PCA')
        fig.show()


    def difference(self):
        x1 = self.scaled_data[:,0]
        y1 = self.scaled_data[:,1]
        x2 = self.scaled_data[:,0]
        y2 = self.scaled_data[:,1]
        x3 = self.pca_data[:,0]
        y3 = self.pca_data[:,1]
        fig, ax = plt.subplots(1, 3, figsize=(16,8))
        ax[0].scatter(x2, y2, c=self.df['target'], cmap=plt.cm.Set1)
        ax[1].scatter(x1, y1, c=self.labels)
        ax[2].scatter(x3, y3, c=self.pca_labels)
        ax[0].set_title('Without DBSCAN')
        ax[1].set_title('With DBSCAN')
        ax[2].set_title('PCA & KMeans')
        plt.show()
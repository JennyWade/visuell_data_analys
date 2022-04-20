import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from umap.umap_ import UMAP
import plotly.express as px

class Kmean:
    def __init__(self):
        self.data = load_wine(as_frame=True)

    def createDataframe(self):
        self.df = pd.DataFrame(np.c_[self.data['data'], self.data['target']], columns= np.append(self.data['feature_names'], ['target']))

    def standardise(self):
        self.data = self.df[0:178]
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
        fig = px.line(x=K, y=no_clusters, title='Elbow')
        fig.show()

    def kmean(self):
        model = KMeans(n_clusters=3)
        model.fit(self.scaled_data)
        self.labels = model.predict(self.scaled_data)

    def umap(self):
        umap = UMAP(random_state=0)
        self.umap_fitted = umap.fit_transform(self.scaled_data)
        fig = px.scatter(self.umap_fitted, x=0, y=1, color=self.labels, title='UMAP')
        fig.show()

    def pcaComponents(self):
        self.pca_full_df = PCA(n_components=14)
        self.pca_full_df.fit(self.scaled_data)
        fig = px.line(np.cumsum(self.pca_full_df.explained_variance_ratio_), title='PCA component variance ratio for dataframe')
        fig.show()

    def pca(self):
        self.pca = PCA(n_components=10)
        self.pca_data = self.pca.fit_transform(self.scaled_data)
        model = KMeans(n_clusters=3)
        model.fit(self.pca_data)
        self.pca_labels = model.predict(self.pca_data)

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
        ax[0].set_title('Without KMeans')
        ax[1].set_title('With KMeans')
        ax[2].set_title('PCA KMeans')
        plt.show()
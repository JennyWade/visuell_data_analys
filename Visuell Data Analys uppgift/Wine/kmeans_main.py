import Wine_Kmeans

if __name__ == "__main__":
    kmeans = Wine_Kmeans.Kmean()
    kmeans.createDataframe()
    kmeans.standardise()
    kmeans.elbow()
    kmeans.kmean()
    kmeans.umap()
    kmeans.pcaComponents()
    kmeans.pca()
    kmeans.pcaUmap()
    kmeans.difference()
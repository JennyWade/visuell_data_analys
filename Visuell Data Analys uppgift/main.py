import Breast_cancer_KMeans

if __name__ == "__main__":
    kmeans = Breast_cancer_KMeans.Kmean()
    kmeans.createDataframe()
    kmeans.standardise()
    kmeans.elbow()
    kmeans.kmean()
    kmeans.umap()
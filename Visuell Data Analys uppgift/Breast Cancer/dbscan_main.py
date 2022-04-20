import Breast_cancer_DBSCAN

if __name__ == "__main__":
    dbscan = Breast_cancer_DBSCAN.DBScan()
    dbscan.createDataframe()
    dbscan.standardise()
    dbscan.nearestNeighbor()
    dbscan.nearestNeighborGraph()
    dbscan.dbscan()
    dbscan.umap()
    dbscan.pca()
    dbscan.pcaUmap()
    dbscan.difference()
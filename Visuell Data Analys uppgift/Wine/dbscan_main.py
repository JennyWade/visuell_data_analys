import Wine_DBSCAN

if __name__ == "__main__":
    dbscan = Wine_DBSCAN.DBScan()
    dbscan.createDataframe()
    dbscan.standardise()
    dbscan.nearestNeighbor()
    dbscan.nearestNeighborGraph()
    dbscan.dbscan()
    dbscan.umap()
    dbscan.pca()
    dbscan.pcaUmap()
    dbscan.difference()
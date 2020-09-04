from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, homogeneity_score
from sklearn import preprocessing

from scipy.spatial.distance import pdist
from scipy.cluster.vq import vq, kmeans, whiten

from nltk.cluster.kmeans import KMeansClusterer

import fastcluster
import scipy.cluster
import numpy as np
import nltk
import replearn.b3 as b3

class Clustering(object):
    def __init__(self,
                event_log):
        self._event_log = event_log
        
        # properties
        self._pred_labels = None

    def cluster(self, vectors, clusterer='k_means', n_clusters=5, metric='cosine'):   
        """
        Clusters the given vectors
        
        :vectors: the vector representation to cluster
        :clusterer: the cluster algorithm (k_means, agglomerative)
        :n_clusters: the number of clusters to generate
        :metric: the metric for the clusterer
        """
        if clusterer == 'k_means':
            self._pred_labels = self.k_means(vectors, n_clusters, metric=metric)
        elif clusterer == 'agglomerative':
            self._pred_labels = self.agglomerative(vectors, n_clusters, metric=metric)
    
    
    def evaluate(self):
        """
        Computes the ARI, NMI, Silhouette and distribution of the computed clusters.
        """
        # distribution of clusters
        unique, counts = np.unique(self._pred_labels, return_counts=True)
        distribution = dict(zip(unique, counts))

        # evaluation
        ari = adjusted_rand_score(self._event_log.true_cluster_labels, self._pred_labels)
        nmi = normalized_mutual_info_score(self._event_log.true_cluster_labels, self._pred_labels)
        homogeneity = homogeneity_score(self._event_log.true_cluster_labels, self._pred_labels)
        
        fmeasure, precision, recall = b3.calc_b3(self._event_log.true_cluster_labels, self._pred_labels)

        return ari, nmi, fmeasure, 0, homogeneity, distribution
    
    
    # K-Means
    def k_means(self, vectors, n_clusters, metric='euclid'):
        if metric == 'cosine':
            vector_norm = preprocessing.normalize(vectors, norm='l2')
        else:
            vector_norm = vectors
            
        clustering = KMeans(n_clusters=n_clusters)
        clustering = clustering.fit(vector_norm)
        pred_labels = clustering.labels_

        return pred_labels
    
    # Agglomerative
    def agglomerative(self, vectors, n_clusters, method='ward', metric='cosine'):
        linkageMatrix = fastcluster.linkage(vectors, method=method, metric=metric)
        pred_labels = scipy.cluster.hierarchy.fcluster(linkageMatrix, n_clusters, criterion='maxclust')

        return pred_labels
    
    @property
    def pred_labels(self):
        return self._pred_labels
    
    @property
    def dist_matrix(self):
        return self._dist_matrix
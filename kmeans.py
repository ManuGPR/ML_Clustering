from random import random
from math import sqrt
import matplotlib.pyplot as plt
import numpy
import pandas as pd



class MyK_Means:
  def __init__(self, k: int, data: pd.DataFrame):
    # Número de clusters
    self.k = k

    # Datos sobre los que trabajar
    self.data = data

    # Lista con centroides, index del centroide = value del centroide
    self.centroids = []

    # Lista con los clusters, index del centroide = lista con instancias de esos clusters
    self.past_clusters = []
    self.clusters = []

    self.init_cluster()
    self.init_centroid()

  @property
  def cluster_labels(self):
    out = numpy.array([], dtype=numpy.int32)
    for row in range(len(self.data)):
      closest_centroid = self.calculate_closest_centroid(self.data.iloc[row, :])
      entero = numpy.int32(closest_centroid)
      out = numpy.append(out, [entero,])
    return out

  def init_cluster(self):
    for key in range(self.k):
      self.clusters.append(pd.DataFrame(columns=self.data.columns))

  def init_centroid(self):
    list_min = self.data.min().to_list()
    list_max = self.data.max().to_list()

    for key in range(self.k):
      centroid_list = []
      for column in range(len(self.data.columns)):
        centroid_list.append(random()*(list_max[column] - list_min[column]) + list_min[column])
      self.centroids.append(tuple(centroid_list))

  def calculate_closest_centroid(self, row):
    distances_cluster = []
    for key in range(len(self.centroids)):
      distances_cluster.append((self.calc_distance(self.centroids[key], row)))
    return distances_cluster.index(min(distances_cluster))

  def fit(self):
    it = 0
    while (True):
      it += 1
      self.past_clusters = []
      for n in range (self.k):
        self.past_clusters.append(self.clusters[n].copy(deep=True))
      self.clusters = []
      self.init_cluster()
      for row in range(len(self.data)):
        closest_centroid = self.calculate_closest_centroid(self.data.iloc[row, :])
        self.clusters[closest_centroid].loc[len(self.clusters[closest_centroid])] = self.data.iloc[row, :]
      self.recalculate_centroids()

      if (self.check_variation()):
        return it

  @staticmethod
  def calc_distance(vec1, vec2):
    return numpy.linalg.norm(vec1-vec2)

  def recalculate_centroids(self):
    self.centroids = []
    for index in range(len(self.clusters)):
      self.centroids.append(self.clusters[index].mean())

  def check_variation(self):
    for i in range(self.k):
      if not self.clusters[i].equals(self.past_clusters[i]):
        return False
    return True

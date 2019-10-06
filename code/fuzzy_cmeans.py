import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import math
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt
from iads_a import kmoyennes as km


def _memberships(dist,data,fuzz):
    u=[]
    cluster = []
    affectation = dict()
    for i in range(len(dist)):
        u.append([])
        for data in dist[i]:
            if(data == 0):
                u[i].append(1.0)
            else:
                u[i].append(1/sum([math.pow((data/x),fuzz) for x in dist[i]]))
        d=max(u[i])
        if u[i].index(d) in affectation:
            affectation[u[i].index(d)].append(i)
        else:
            affectation[u[i].index(d)] = [i]
        cluster.append(u[i].index(d))
    return np.asarray(cluster), np.asarray(u), affectation

def compute_centroids(data, u, fuzz):
    w = np.zeros(shape=(u.shape[1], data.shape[1]))
    for r in range(w.shape[0]):
        sum_t = np.zeros(shape=(1, data.shape[1]))
        for i in range(data.shape[0]):
            sum_t += (u[i][r] ** fuzz) * data.iloc[i]

        sum_b = np.zeros(shape=(1, data.shape[1]))
        for i in range(data.shape[0]):
            sum_b += u[i][r] ** fuzz
        w[r] = sum_t/ sum_b
    return w



def initialization(df,nb_clusters):
    data = df.values
    centers = data[np.random.choice(data.shape[0], size=nb_clusters, replace=False)]
    return pd.DataFrame(centers,columns=df.columns)

def fuzzy_c_means(data, nb_clusters=3, eps=1e-4, max_iter=200, fuzz=2):

    curr_centroids = initialization(data, nb_clusters)
    memberships = None
    aff = None
    best_aff_inv = None
    centroids = None
    best_loss = np.inf
    curr_memberships = None
    current_iter = 0
    losses = []
    while (current_iter < max_iter):
        curr_memberships, curr_aff, aff_b = memberships(data, centroids, fuzz)
        curr_centroids = _compute_centroids(data, memberships, fuzz)
        curr_loss = _compute_loss(data, memberships, centroids, fuzz)
        losses.append(curr_loss)
        if curr_loss < loss:
            loss = curr_loss
            memberships = curr_memberships
            centroids = curr_centroids
            aff = curr_aff
            best_aff_b = aff_b
        current_iter += 1
        if abs(losses[-2] - losses[-1]) < eps:
            memberships, centroids, np.array(losses), aff,best_aff_b
            
    return memberships, entroids, np.array(losses), aff,best_aff_b
def affiche_resultat(df,centroides, dic):
    fig0, ax0 = plt.subplots(figsize=(10,10))
    ax0.scatter(centroides['X'],centroides['Y'],color='r',marker='x')
    couleurs = ['b','r','g','y','k']
    for i, k in enumerate(dic):
        data = df.loc[dic[k]]
        ax0.scatter(data['X'],data['Y'],color=couleurs[i], marker = '.')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')


    
"""Implementation avec des fonctions utilisant le calcul matriciel (inspirée de quelques exemples trouvés sur Stack Overflow"""
def _compute_centroids(data, memberships, fuzz):
    fuzz_ = memberships ** fuzz
    sum_ = np.sum(fuzzified_memberships, axis=0)
    d_ = np.dot(data.T, fuzz_)
    l = np.divide(d_, sum_d,where=sum_ != 0).T
    return pd.DataFrame(l,columns=data.columns)

def _compute_loss(data, memberships, centroids, fuzz):
    dist_data_centroids = cdist(data, centroids, metric="euclidean") ** 2
    return ((memberships ** fuzz) * dist_data_centroids).sum()   
def memberships(data, centroids, f):
    distances = cdist(data.values, centroids.values, metric="euclidean")
    powered = np.power(distances, -2 / (f-1), where=distances != 0)
    sum_ = powered.sum(axis=1, keepdims=True)
    divided = np.divide(tmp, sum_, where=sum_ != 0)
    aff = dict()
    aff_b = dict()
    id_ = np.where(np.isclose(dist_data_centroids, 0))
    res[id_[0]] = 0
    res[id_] = 1
    res = np.fmax(res, 0.)
    d = np.argmax(res,axis=1)
    for i in range(len(d)):
        if d[i] in  aff:
            aff[d[i]].append(i)
        else:
            aff[d[i]] = [i]
        if i in aff_b:
            aff_b[i].append(d[i])
        else:
            aff_b[i] = d[i]
    return res, aff,aff_b

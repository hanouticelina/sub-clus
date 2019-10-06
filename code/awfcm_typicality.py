import numpy as np
import copy
from sklearn import preprocessing
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
import math
import time
from iads_a import kmoyennes as km
from iads_a import fuzz_clustering as fc
from datetime import datetime as dt
import matplotlib.pyplot as plt




#-------------------------------------------------------------------------------------------
"""Typicality degree, resemblance, dissimilarity"""
def distance_normalized(x,y, dft):
    all_distances = cdist(dft.values,dft.values)
    d_max = np.max(all_distances)
    euc_dist = np.linalg.norm(dft.iloc[x]-dft.iloc[y])
    min_ = np.min([euc_dist/d_max,1])
    return np.max([min_,0])

def dissimilarity(x,affectations,aff_b,dft):
    aff_inv = aff_b
    aff = copy.deepcopy(affectations)
    del aff[aff_b[x]]
    list_ = np.hstack(np.asarray(list(aff.values())))
    s = []
    for v in list_:
        s.append(distance_normalized(x,v,dft))
    return np.mean(np.asarray(s))

def resemblance(x, affectations, aff_b, dft):
    aff = copy.deepcopy(affectations)
    l = aff[aff_b[x]]
    l.remove(x)
    list_ = np.hstack(np.asarray(l))
    s = []
    for v in list_:
        s.append(1-distance_normalized(x,v,dft))
    return np.mean(np.asarray(s))
def typicality_degree(x,aff,aff_inv, data, agg):
    if(agg == 'MAX'):
        return max(dissimilarity(x,aff,aff_inv,data),resemblance(x,aff,aff_inv,data))
    if(agg == 'MIN'):
        return min(dissimilarity(x,aff,aff_inv,data),resemblance(x,aff,aff_inv,data))
    else :
        return 0.6*resemblance(x,aff,aff_inv,data)+ 0.4*dissimilarity(x,aff,aff_inv,data)

def distance_normalized_w(x,y,p,dft,distances):
    d_max = np.max(distances[p])
    euc_dist = np.linalg.norm(dft.iloc[x][dft.columns[p]]-dft.iloc[y][dft.columns[p]])
    min_ = np.min([euc_dist/d_max,1])
    return np.max([min_,0])

def dissimilarity_w(x,affectations,aff_b,dft,p, distances):
    aff_inv = aff_b
    aff = copy.deepcopy(affectations)
    del aff[aff_b[x]]
    list_ = np.hstack(np.asarray(list(aff.values())))
    s = []
    for v in list_:
        s.append(distance_normalized_w(x,v,p,dft,distances))
    return np.mean(np.asarray(s))

def resemblance_w(x, affectations, aff_b, dft,p,distances):
    aff = copy.deepcopy(affectations)
    l = aff[aff_b[x]]
    l.remove(x)
    list_ = np.hstack(np.asarray(l))
    s = []
    for v in list_:
        s.append(1-distance_normalized_w(x,v,p,dft,distances))
    return np.mean(np.asarray(s))
def typicality_degree_w(x,aff,aff_inv, data, agg,p,distances):
    if(agg == 'MAX'):
        return max(dissimilarity_w(x,aff,aff_inv,data,p,distances),resemblance_w(x,aff,aff_inv,data,p,distances))
    if(agg == 'MIN'):
        return min(dissimilarity_w(x,aff,aff_inv,data,p,distances),resemblance_w(x,aff,aff_inv,data,p,distances))
    else :
        return 0.6*resemblance_w(x,aff,aff_inv,data,p,distances)+0.4*dissimilarity_w(x,aff,aff_inv,data,p,distances)







#---------------------------------------------------------------------------------------------------------
"""Weighted fuzzy c-means"""
def compute_weights_bis(df,aff, aff_inv, centroids,distances):
    w = np.full(shape=(centroids.shape[0],df.shape[1]),fill_value=1.)
    for r in range(centroids.shape[0]):
        for p in range(df2.shape[1]):
            typicalities = [[],[]]
            for i in range(df2.shape[0]):
                typicalities[aff_inv[i]].append(typicality_degree_w(i,aff,aff_inv,df,'B',p,distances))
            w[r][p] = np.mean(typicalities[r])
    return w
def compute_memberships_w_typ(data, centroids, fuzz,w):
    u_ir = np.zeros(shape=(data.shape[0], centroids.shape[0]))
    affectation = dict()
    affectation_inv = dict()
    for i in range(data.shape[0]):
        for r in range(centroids.shape[0]):
            d_ir = np.sqrt(((w[r]**2)*((data.iloc[i] - centroids.iloc[r]) ** 2)).sum())
            if d_ir == 0:
                for s in range(centroids.shape[0]):
                    u_ir[i][s] = 0
                u_ir[i][r] = 1
                break

            sum_b = 0
            for s in range(centroids.shape[0]):
                d_is = np.sqrt(((w[s]**2)*((data.iloc[i] - centroids.iloc[s]) ** 2)).sum())
                if d_is == 0:
                   
                    continue
                sum_b += (d_ir / np.sqrt(((data.iloc[i] - centroids.iloc[s]) ** 2).sum())) ** (2 / (fuzz - 1))
            u_ir[i][r] = 1 / sum_b
        d=np.argmax(u_ir[i])
        if d in affectation:
            affectation[d].append(i)
        else:
            affectation[d] = [i]
        affectation_inv[i] = d
    return u_ir, affectation, affectation_inv

def fuzzy_c_means_w(data,distances, nb_clusters=2, eps=1e-4, max_iter=300, fuzz=2):

    curr_centroids = fc.initialization(data, nb_clusters)
    memberships = None
    aff = None
    best_aff_inv = None
    best_w = None
    centroids = None
    loss = np.inf
    curr_memberships = None
    current_iter = 0
    losses = []
    w = np.full(shape=(curr_centroids.shape[0],data.shape[1]),fill_value=1.)
    while (current_iter < max_iter):
        curr_memberships, curr_aff, aff_inv = compute_memberships_w_typ(data, centroids, fuzz,w)
        w = compute_weights_bis(data,aff,aff_inv,centroids,distances)
        curr_centroids = fc._compute_centroids(data, memberships, fuzz)
        loss = compute_loss_w(data, memberships, centroids, fuzz,w)
        losses.append(loss)
        if curr_loss < loss:
            loss = curr_loss
            memberships = curr_memberships
            centroids = curr_centroids
            best_aff = aff
            best_aff_inv = aff_inv
            best_w = w
        
	if(abs(losses[-2] - losses[-1]) < eps):
		return memberships, centroids, np.array(losses), aff, best_w, best_aff_inv
	current_iter += 1
    return memberships, centroids, np.array(losses), aff, best_w, best_aff_inv

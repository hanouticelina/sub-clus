# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: experiences.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz as gv
from . import decision_trees as dt
from . import Classifiers as cl
from . import LabeledSet as ls
from . import clustering as ct
from . import utils as ut
from itertools import combinations

path = "../Data_Mining_MovieLens/movielens/"
columns = ['score_ml', 'score_tmdb', 'score_imdb']
cmp_prop = 0.2

def read_files():
    # Lecture des fichiers
    movies_complete = pd.DataFrame(data=pd.read_csv(path + "movies_complete.csv", sep=","))
    genres_pop = pd.DataFrame(data=pd.read_csv(path + "genres_popularity.csv", sep=","))

    # Dummy-coding des genres
    movies_complete = movies_complete.join(movies_complete.genres.str.get_dummies())
    movies_complete.drop(['genres'], inplace=True, axis=1)

    # Nouvelle feature : popularité d'un film selon ses genres associés
    columns = genres_pop['genre'].values
    column_values = genres_pop.where(genres_pop.genre == columns)['popularity'].values
    genre_scores = movies_complete.apply(lambda row: row[columns] * column_values, axis=1) \
                    .sum(axis=1).rename('g_popularity')

    movies_complete = movies_complete.join(genre_scores)

    # Filtrage des films dont on connaît pas le revenu global
    movies_complete = movies_complete[movies_complete.worldwide_gross != 0.]

    # Filtrage des features inutiles
    scores = movies_complete[['score_ml', 'score_tmdb', 'score_imdb']]
    scores = round(scores, 1)
    movies_complete.drop(np.hstack((['tmdbId', 'original_language', 'original_title', 'popularity', \
                          'release_date', 'title', 'score_tmdb', 'votes_tmdb', 'movieId', \
                          'imdbId', 'year', 'company', 'country', 'director', 'score_imdb', \
                          'star', 'votes_imdb', 'writer', 'votes_ml', 'score_ml'], columns)),
                         inplace=True, axis=1)
    return movies_complete, scores

def to_df(matrix, score_ser):
    dataset = ls.LabeledSet(5)
    for row, value in zip(matrix, score_ser):
        dataset.addExample(row, value)
    return dataset

class LinearKernel:
    dim = 6
    def transform(self, x):
        return np.hstack(([1] , x))

class BilinearKernel:
    dim = 16
    def transform(self, x):
        features = [a * b for a, b in combinations(x, 2)]
        return np.hstack(([1] , x, features))

class SquareKernel:
    dim = 21
    def transform(self, x):
        features = [a * b for a, b in combinations(x, 2)]
        return np.hstack(([1] , x, features, [k**2 for k in x]))

class PowerKernel:
    dim = 21
    def transform(self, x):
        powers = np.array([[b ** e for b in x] for e in range(1, 5)]).reshape(-1)
        return np.hstack(([1] , *np.split(powers, 5)))

def batch_kernels(movies, scores, col_num=0):
    col = columns[col_num]
    score_ser = scores[col]
    matrix = ct.normalisation(movies)
    ds = to_df(matrix, score_ser)
    kernel = SquareKernel()
    print("Square kernel")

    eps = 0.1
    iters = [10, 50, 100, 200, 500, 1000]
    loss_iters = []
    for it in iters:
        classifier = cl.ClassifierBatchGradDescentKernel(kernel.dim, eps, kernel, it)
        classifier.train(ds)
        loss_mean = classifier.loss_values[-int(cmp_prop*it):].mean()
        loss_iters.append((it, loss_mean))
    loss_iters.sort(key=lambda x: x[1])
    print("epsilon = 0.1 :", loss_iters)

    num_iters = loss_iters[0][0]
    epsilons = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.]
    loss_eps = []
    for eps in epsilons:
        classifier = cl.ClassifierBatchGradDescentKernel(kernel.dim, eps, kernel, num_iters)
        classifier.train(ds)
        loss_mean = classifier.loss_values[-int(cmp_prop*num_iters):].mean()
        loss_eps.append((eps, loss_mean))
    loss_eps.sort(key=lambda x: x[1])
    print("num_iters =", num_iters, ":", loss_eps)

    return loss_iters, loss_eps

def k_nn_performance(dataset, k_min, k_max, k_step=1, prop=0.7):
    training_set, test_set = ut.split(dataset, prop)

    ks = np.arange(k_min, k_max+1, k_step)
    weighted = np.zeros_like(ks, dtype=np.float64)
    non_weighted = np.zeros_like(ks, dtype=np.float64)
    inp_dim = dataset.getInputDimension()
    for i_k, k in enumerate(ks):
        _cl = cl.ClassifierKNN(inp_dim, k, weighted=True)
        _cl.train(training_set)
        weighted[i_k] += _cl.loss(test_set)

        _cl = cl.ClassifierKNN(inp_dim, k, weighted=False)
        _cl.train(training_set)
        non_weighted[i_k] += _cl.loss(test_set)
    print(training_set.size(), test_set.size())
    plt.xlabel('Nombre de voisins')
    plt.ylabel(f'Loss')
    plt.plot(ks, weighted)
    plt.plot(ks, non_weighted)
    plt.legend(['Somme pondérée', 'Somme simple'])
    plt.show()

    return weighted, non_weighted

def compare_batch_knn(movies, scores, prop=0.7):
    legends = ['MovieLens', 'TMDb', 'IMDb']
    matrix = ct.normalisation(movies)
    datasets = [to_df(matrix, scores[col]) for col in columns]
    div_datasets = [ut.split(ds, prop) for ds in datasets]

    # Perceptron batch kernelisé
    kernel = SquareKernel()
    eps = 0.1
    iters = 1000

    # K-NN
    k = 19
    inp_dim = datasets[0].getInputDimension()

    batch_loss = np.empty((len(datasets), iters), dtype=np.float)
    final_loss = np.empty((len(datasets), 2, 2), dtype=np.float) # pb / algo / set

    for ind, (tr_set, test_set) in enumerate(div_datasets):
        classifier = cl.ClassifierBatchGradDescentKernel(kernel.dim, eps, kernel, iters)
        classifier.train(tr_set)
        batch_loss[ind] = classifier.loss_values
        final_loss[ind][0][0] = classifier.loss_values[-1]
        final_loss[ind][0][1] = classifier.loss(test_set)

        classifier = cl.ClassifierKNN(inp_dim, k, weighted=True)
        classifier.train(tr_set)
        final_loss[ind][1][0] = classifier.loss(tr_set)
        final_loss[ind][1][1] = classifier.loss(test_set)

    iter_array = 1 + np.arange(iters)

    plt.xlabel('Itération')
    plt.ylabel(f'Loss')
    for bl in batch_loss:
        plt.plot(iter_array, bl)
    plt.legend(legends)
    plt.show()

    return batch_loss, final_loss

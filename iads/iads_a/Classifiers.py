# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import graphviz as gv
from . import decision_trees as dt

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """

        raise NotImplementedError("Please Implement this method")

    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système
        """
        tp_preds = np.apply_along_axis(self.predict, 1, dataset.x).reshape(-1,1)
        return np.where(tp_preds * dataset.y > 0)[0].size / dataset.size()

    def loss(self, dataset):
        """ rend la fonction de coût C(X) = (1/2m)*||Y - k(X)*w||²,
        où m est le nombre d'exemples, i.e. m = n_rows(X)
        """
        X = dataset.x
        Y = dataset.y
        f_X = np.apply_along_axis(lambda x: self.predict(x), 1, X).reshape(-1, 1)
        return ((Y - f_X) ** 2).sum() / (2 * dataset.size())



# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.w = np.random.uniform(low=-1.,high=1.,size=input_dimension)

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        return np.dot(self.w, x)

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        print("This is not a learning classifier")

# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k, weighted=True):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        self.weighted = weighted

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        dists = np.apply_along_axis(lambda b: np.linalg.norm(b - x), 1, \
                self.training_set.x)
        ind_sort = np.argsort(dists)[:self.k]
        nn_dists = 1 - dists[ind_sort] / dists[ind_sort].sum()
        knn = self.training_set.y[ind_sort].reshape(-1)
        if self.weighted is True:
            return np.dot(knn, nn_dists).sum() / (self.k - 1)
        return knn.sum() / self.k

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.training_set = labeledSet

# ---------------------------
class ClassfierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.lr = learning_rate
        #self.w = np.zeros(input_dimension) # no preconception
        self.w = np.random.uniform(low=-self.lr,high=self.lr,size=input_dimension) # random

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if z > 0:
            return +1
        else:
            return -1

    def loss(self,labeledSet):
        """ rend la fonction de coût C(X) = sum(alpha(y * predict(x)), où x = X[i]
        et alpha(x) = 1 si x < 0,
                      0 sinon
        """
        X = labeledSet.x
        Y = labeledSet.y
        Yp = np.apply_along_axis(lambda x: self.predict(x), 1, X).reshape(-1,1)
        return np.where(Y*Yp < 0,1,0).sum()

    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.accuracy_array = np.zeros(labeledSet.size())
        self.loss_array = np.zeros_like(self.accuracy_array)
        indices = np.arange(labeledSet.size())
        np.random.shuffle(indices)
        for i in range(len(indices)):
            x = labeledSet.getX(indices[i])
            y = labeledSet.getY(indices[i])
            yp = self.predict(x)
            self.w += self.lr * (y - yp) * x
            self.accuracy_array[i] = self.accuracy(labeledSet)
            self.loss_array[i] = self.loss(labeledSet)

# ---------------------------
class ClassfierStochasticGradDescent(Classifier):
    """ Classifieur par descente de gradient stochastique
    La fonction de coût est C(X) = sum((y - <w,x>)²), où x = X[i]
    Sa dérivée partielle par rapport à w_i est -2 * sum((y - <w, x>) * x_i)
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.lr = learning_rate
        #self.w = np.zeros(input_dimension) # no preconception
        self.w = np.random.uniform(low=-self.lr,high=self.lr,size=input_dimension) # random

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if z > 0:
            return +1
        else:
            return -1

    def loss(self,labeledSet):
        """ rend la fonction de coût C(X) = ||Y - X*w||²
        """
        X = labeledSet.x
        Y = labeledSet.y
        f_X = np.dot(X, self.w).reshape(-1,1)
        return ((Y - f_X) ** 2).sum()

    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        Ce classifieur n'utilise qu'un exemple par itération
        (mise à jour de w)
        """
        self.accuracy_array = np.zeros(labeledSet.size())
        self.loss_array = np.zeros_like(self.accuracy_array)
        indices = np.arange(labeledSet.size())
        np.random.shuffle(indices)
        for i in range(len(indices)):
            x = labeledSet.getX(indices[i])
            y = labeledSet.getY(indices[i])
            f_x = np.dot(self.w, x)
            self.w += self.lr * (y - f_x) * x
            self.accuracy_array[i] = self.accuracy(labeledSet)
            self.loss_array[i] = self.loss(labeledSet)

# ---------------------------
class ClassfierBatchGradDescent(Classifier):
    """ Classifieur par descente de gradient en batch
    La fonction de coût est C(X) = (1/2m)*sum((y - <w,x>)²), où x = X[i]
    Sa dérivée partielle par rapport à w_i est -2 * sum((y - <w, x>) * x_i)
    Le gradient vaut ainsi -X^ * (Y - X*w), où X^ est la transposée de X
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.lr = learning_rate
        #self.w = np.zeros(input_dimension) # no preconception
        self.w = np.random.uniform(low=-self.lr,high=self.lr,size=input_dimension) # random

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if z > 0:
            return +1
        else:
            return -1

    def loss(self,labeledSet):
        """ rend la fonction de coût C(X) = (1/2m)*||Y - X*w||²,
        où m est le nombre d'exemples, i.e. m = n_rows(X)
        """
        X = labeledSet.x
        Y = labeledSet.y
        f_X = np.dot(X, self.w).reshape(-1,1)
        return ((Y - f_X) ** 2).sum() / (2 * labeledSet.size())

    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        Ce classifieur utilise tous les exemples pour la mise à
        jour de w
        """
        X = labeledSet.x
        Y = labeledSet.y
        f_X = np.dot(X, self.w).reshape(-1,1)
        self.w += self.lr * (X * (Y - f_X)).sum(axis=0) / labeledSet.size()
        self.accuracy_value = self.accuracy(labeledSet)
        self.loss_value = self.loss(labeledSet)

# ---------------------------
class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.lr = learning_rate
        #self.w = np.zeros(input_dimension) # no preconception
        self.w = np.random.uniform(low=-self.lr,high=self.lr,size=dimension_kernel) # random
        self.kernel = kernel


    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(self.kernel.transform(x), self.w)
        if z > 0:
            return +1
        else:
            return -1


    def loss(self,labeledSet):
        """ rend la fonction de coût C(X) = sum(alpha(y * predict(x))), où x = X[i]
        et alpha(x) = 1 si x < 0,
                      0 sinon
        """
        X = labeledSet.x
        Y = labeledSet.y
        Yp = np.apply_along_axis(lambda x: self.predict(x), 1, X).reshape(-1,1)
        return np.where(Y*Yp < 0,1,0).sum()

    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.accuracy_array = np.zeros(labeledSet.size())
        self.loss_array = np.zeros_like(self.accuracy_array)
        indices = np.arange(labeledSet.size())
        np.random.shuffle(indices)
        for i in range(len(indices)):
            x = labeledSet.getX(indices[i])
            k_x = self.kernel.transform(x)
            y = labeledSet.getY(indices[i])
            yp = self.predict(x)
            self.w += self.lr * (y - yp) * k_x
            self.accuracy_array[i] = self.accuracy(labeledSet)
            self.loss_array[i] = self.loss(labeledSet)

# ---------------------------
class ClassifierStochasticGradDescentKernel(Classifier):
    """ Classifieur par descente de gradient stochastique kernelisé
    La fonction de coût est C(X) = (1/2m)*sum((y - <w,k(x)>)²), où x = X[i]
    Sa dérivée partielle par rapport à w_i est -2 * sum((y - <w, k(x)>) * k(x)_i)
    """
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.lr = learning_rate
        #self.w = np.zeros(input_dimension) # no preconception
        self.w = np.random.uniform(low=-self.lr,high=self.lr,size=dimension_kernel) # random
        self.kernel = kernel


    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(self.kernel.transform(x), self.w)
        if z > 0:
            return +1
        else:
            return -1


    def loss(self,labeledSet):
        """ rend la fonction de coût C(X) = ||Y - k(X)*w||²
        """
        X = labeledSet.x
        Y = labeledSet.y
        k_X = np.apply_along_axis(lambda x: self.kernel.transform(x), 1, X)
        f_X = np.dot(k_X, self.w).reshape(-1,1)
        return ((Y - f_X) ** 2).sum()

    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        Ce classifieur n'utilise qu'un exemple par itération
        (mise à jour de w)
        """
        self.accuracy_array = np.zeros(labeledSet.size())
        self.loss_array = np.zeros_like(self.accuracy_array)
        indices = np.arange(labeledSet.size())
        np.random.shuffle(indices)
        for i in range(len(indices)):
            x = labeledSet.getX(indices[i])
            k_x = self.kernel.transform(x)
            y = labeledSet.getY(indices[i])
            f_x = np.dot(self.w, k_x)
            self.w += self.lr * (y - f_x) * k_x
            self.accuracy_array[i] = self.accuracy(labeledSet)
            self.loss_array[i] = self.loss(labeledSet)

# ---------------------------
class ClassifierBatchGradDescentKernel(Classifier):
    """ Classifieur par descente de gradient en batch kernelisé
    La fonction de coût est C(X) = (1/2m)*sum((y - <w,k(x)>)²), où x = X[i]
    Sa dérivée partielle par rapport à w_i est -2 * sum((y - <w, k(x)>) * k(x)_i)
    Le gradient vaut ainsi -k(X)^ * (Y - k(X)*w), où X^ est la transposée de X
    """
    def __init__(self,dimension_kernel,learning_rate,kernel, num_iters):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.lr = learning_rate
        #self.w = np.zeros(input_dimension) # no preconception
        self.w = np.zeros(dimension_kernel) # np.random.uniform(low=-bound,high=bound,size=dimension_kernel) # random
        self.kernel = kernel
        self.num_iters = num_iters
        self.loss_values = np.empty(num_iters, dtype=np.float)


    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        return np.dot(self.kernel.transform(x), self.w)


    def loss(self,labeledSet):
        """ rend la fonction de coût C(X) = (1/2m)*||Y - k(X)*w||²,
        où m est le nombre d'exemples, i.e. m = n_rows(X)
        """
        X = labeledSet.x
        Y = labeledSet.y
        k_X = np.apply_along_axis(lambda x: self.kernel.transform(x), 1, X)
        f_X = np.dot(k_X, self.w).reshape(-1,1)
        return ((Y - f_X) ** 2).sum() / (2 * labeledSet.size())

    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        Ce classifieur n'utilise qu'un exemple par itération
        (mise à jour de w)
        """
        for i in range(self.num_iters):
            X = labeledSet.x
            Y = labeledSet.y
            k_X = np.apply_along_axis(lambda x: self.kernel.transform(x), 1, X)
            f_X = np.dot(k_X, self.w).reshape(-1,1)
            self.w += self.lr * (k_X * (Y - f_X)).sum(axis=0) / labeledSet.size()
            # self.accuracy_value = self.accuracy(labeledSet)
            self.loss_values[i] = self.loss(labeledSet)

# ---------------------------
class ArbreDecision(Classifier):
    # Constructeur
    def __init__(self,epsilon):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None

    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self, x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)

    # Permet d'entrainer le modele sur un ensemble de données
    def train(self, set):
        # construction de l'arbre de décision
        self.set = set
        self.racine = dt.construit_AD_GI(set,self.epsilon)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)

# ---------------------------
class ClassifierBaggingTree(Classifier):
    # Constructeur
    def __init__(self, B, m_percent, epsilon, replacement):
        self.B = B
        self.m_percent = m_percent
        self.replacement = replacement
        self.epsilon= epsilon # valeur seuil d'entropie pour arrêter la construction
        self.forest = [] # fôrét d'arbres de décision

    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend -1 (classe -1) ou 1 (classe 1)
        votes = sum([tree.predict(x) for tree in self.forest])
        return np.sign(votes)

    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        m = round(self.m_percent * set.size())
        # construction des arbres de décision
        for _ in range(self.B):
            tree = dt.ArbreDecision(self.epsilon)
            sample = dt.echantillonLS(set, m, self.replacement)
            tree.train(sample)
            self.forest.append(tree)

# ---------------------------
class ClassifierBaggingTreeOOB(ClassifierBaggingTree):
    # Constructeur
    def __init__(self, B, m_percent, epsilon, replacement):
        ClassifierBaggingTree.__init__(self, B, m_percent, epsilon, replacement)
        self.oob = []

    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        m = round(self.m_percent * set.size())
        # construction des arbres de décision
        for _ in range(self.B):
            tree = dt.ArbreDecision(self.epsilon)
            sample_indices = dt.tirage(np.arange(set.size()), m, self.replacement)
            sample = dt.sampleLS(set, sample_indices)
            oob_indices = dt.complement(set, sample_indices)
            oob = dt.sampleLS(set, oob_indices)
            self.oob.append(oob)
            tree.train(sample)
            self.forest.append(tree)

    def oob_accuracy(self):
        ts = [self.forest[i].accuracy(self.oob[i]) for i in range(self.B)]
        return sum(ts) / self.B

# ---------------------------
class ArbreDecisionAleatoire(ArbreDecision):
    # Constructeur
    def __init__(self,epsilon, nbatt):
        ArbreDecision.__init__(self, epsilon)
        self.nbatt = nbatt # nombre d'attributs à utiliser pour la création d'un niveau

    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision
        self.set=set
        self.racine = dt.construit_AD_aleatoire(set,self.epsilon, self.nbatt)

# ---------------------------
class ArbreDecisionGen(ArbreDecision):
    # Constructeur
    def __init__(self,epsilon):
        ArbreDecision.__init__(self, epsilon)

    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision
        self.set=set
        self.racine = dt.construit_ADGen(set,self.epsilon)

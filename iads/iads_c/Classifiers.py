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
from . import LabeledSet as ls

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
        ok = 0
        taille_data = dataset.size()
        for i in range(taille_data):
            if (dataset.getY(i) == self.predict(dataset.getX(i))):
            #On dit qu'une prédiction est correcte si elle a le même signe que
            #la valeur correspondante à cette donnée dans le dataset.
                ok+=1
        return (ok/taille_data)*100
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

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k

    """def predict(self, x):
        rend la prediction sur x (-1 ou +1)
        
        dists = np.apply_along_axis(lambda b: np.sum((b - x) ** 2) , 1, self.training_set.x)
        ind_sort = np.argsort(dists)
        return self.training_set.y[ind_sort[:self.k]].sum() / self.k"""
    def predict(self,x):
        n=[]
        for i in range(self.training_set.size()): #calculer les k plus proche voisins
            distance = np.linalg.norm(np.array(x)-np.array(self.training_set.getX(i)))
            n.append(distance)
        indsorted = np.argsort(n)  # on a les k plus proche voisins triés
        ind = indsorted[:self.k]
        classes = [0,0,0,0]
        for i in range(len(ind)):
            classes[int(self.training_set.getY(ind[i])[0])] +=n[ind[i]]
        return np.argmax(classes)
    def accuracy2(self,dataset):
        ok = 0
        taille_data = dataset.size()
        for i in range(taille_data):
            if (dataset.getY(i)[0] == self.predict(dataset.getX(i))):
            #On dit qu'une prédiction est correcte si elle a le même signe que
            #la valeur correspondante à cette donnée dans le dataset.
                ok+=1
        return (ok/taille_data)*100
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
        """if z > 0:
            return +1
        else:
            return -1"""
        return z

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
        """if z > 0:
            return +1
        else:
            return -1"""
        return z

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
        """if z > 0:
            return +1
        else:
            return -1"""
        return z

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
        X = labeledSet.x
        Y = labeledSet.y
        k_X = np.apply_along_axis(lambda x: self.kernel.transform(x), 1, X)
        f_X = np.dot(k_X, self.w).reshape(-1,1)
        self.w += self.lr * (k_X * (Y - f_X)).sum(axis=0) / labeledSet.size()
        self.accuracy_value = self.accuracy(labeledSet)
        self.loss_value = self.loss(labeledSet)



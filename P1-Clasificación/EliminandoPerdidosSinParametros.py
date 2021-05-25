#!/usr/bin/env python
# coding: utf-8



#Todas las librerías para los distintos algoritmos
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from statistics import mode
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn import impute
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
 
#Primero con esto y luego con la media para ver si la mejora
mamografias= pd.read_csv("./mamografias.csv",na_values=["?"])
mamografias=mamografias.dropna() 

le = pp.LabelEncoder()
columna_codificada=le.fit_transform(mamografias['Shape'])
mamografias['Shape']=le.fit_transform(mamografias['Shape'])
mamografias['Severity']=le.fit_transform(mamografias['Severity'])
atributos=mamografias[['BI-RADS','Age','Shape','Margin','Density']]
target=mamografias['Severity']
data_train, data_test, target_train, target_test = train_test_split(atributos ,target, test_size = 0.8, random_state = 5)


#Definición de la función de la matriz
def matrizCruzada(prediccion):
    m = confusion_matrix(target_test, prediccion, normalize="all")
    tn,fp,fn,tp=m.ravel();
    print("---------------------------------")
    print("TN ",tn*100)
    print("FP ",fp*100)
    print("FN ",fn*100)
    print("TP ",tp*100)
    print("FP-FN ",(fp-fn)*100)
    return m


#Primer algoritmo Nayve-Bayes

#Nayve-Bayes Gaussian
gnb = GaussianNB()
modeloNBgau = gnb.fit(data_train, target_train)
predNBgau = modeloNBgau.predict(data_test)
scoresGau = cross_val_score(modeloNBgau, atributos, target, cv=5, scoring='accuracy')

#Nayve-Bayes Complement
cnb = ComplementNB()
modeloNBcom = cnb.fit(data_train, target_train)
predNBcom = modeloNBcom.predict(data_test)
scoresCom = cross_val_score(modeloNBcom, atributos, target, cv=5, scoring='accuracy')

#Nayve-Bayes Bernoulli
bnb = BernoulliNB()
modelNBBer = bnb.fit(data_train, target_train)
predNBber = modelNBBer.predict(data_test)
scoresBer = cross_val_score(modelNBBer, atributos, target, cv=5, scoring='accuracy')

#Nayve-Bayes Multinominal
mnb = MultinomialNB()
modelNBMul = mnb.fit(data_train, target_train)
predNBmul = modelNBMul.predict(data_test)
scoresMul = cross_val_score(modelNBMul, atributos, target, cv=5, scoring='accuracy')

#Porcentajes de acierto
print("Usando NB Gaussian se tiene una tasa de acierto del ",np.mean(scoresGau)*100)
print("Usando NB Complement se tiene una tasa de acierto del ",np.mean(scoresCom)*100)
print("Usando NB Bernoulli se tiene una tasa de acierto del ",np.mean(scoresBer)*100)
print("Usando NB Multinominal se tiene una tasa de acierto del ",np.mean(scoresMul)*100)


#Matrices de validación
print("Matriz Gaussian: ", matrizCruzada(predNBgau))
print("Matriz Complement: ", matrizCruzada(predNBcom))
print("Matriz Bernoulli: ", matrizCruzada(predNBber))
print("Matriz Multinominal: ", matrizCruzada(predNBmul))


#ÁRBOLES DE DECISIÓN


#Segundo algoritmo Árboles de decisión

#Árbol de decisión normal
arbNor = tree.DecisionTreeClassifier()
arbNor = arbNor.fit(data_train, target_train)
predADnor = arbNor.predict(data_test)
scoresADnor = cross_val_score(arbNor, atributos, target, cv=5, scoring='accuracy')

#Árbol de decisión extra
arbEx = tree.ExtraTreeClassifier()
arbEx = arbEx.fit(data_train, target_train)
predADex = arbEx.predict(data_test)
scoresADex = cross_val_score(arbEx, atributos, target, cv=5, scoring='accuracy')

#Porcentajes de acierto
print("Usando AD normal se tiene una tasa de acierto del ",np.mean(scoresADnor)*100)
print("Usando AD extra se tiene una tasa de acierto del ",np.mean(scoresADex)*100)

#Matrices de validación
print("Matriz ArbDec Normal: ",matrizCruzada(predADnor))
print("Matriz ArbDec Extra: ",matrizCruzada(predADex))


#Pintamos los árboles
tree.plot_tree(arbNor)
tree.plot_tree(arbEx)

#Tercer algoritmo SUPPORT VECTOR MACHINE 

#SVM - NuSVC
svr_nu = NuSVC()
svr_nu.fit(data_train, target_train)
predsvNu = svr_nu.predict(data_test)
scoresNu = cross_val_score(svr_nu, atributos, target, cv=5, scoring='accuracy')

#SVM - SVC
svr_svc = SVC()
svr_svc.fit(data_train, target_train)
predsvSvc = svr_svc.predict(data_test)
scoresSvc = cross_val_score(svr_svc, atributos, target, cv=5, scoring='accuracy')

#Porcentajes de acierto
print("Usando NuSVC se tiene una tasa de acierto del ",np.mean(scoresNu)*100)
print("Usando SVC se tiene una tasa de acierto del ",np.mean(scoresSvc)*100)

#Matrices de validación
print("Matriz SVM - Nu: ",matrizCruzada(predsvNu))
print("Matriz SVM - SVC: ",matrizCruzada(predsvSvc))


#Cuarto algoritmo ENSEMBLED METHODS

#Bagging meta-estimator
bagging = BaggingClassifier()
bagging.fit(data_train, target_train)
preBag = bagging.predict(data_test)
scoresBag = cross_val_score(bagging, atributos, target, cv=5, scoring='accuracy')

#Random Forests
forests = RandomForestClassifier()
forests.fit(data_train, target_train)
preFo = forests.predict(data_test)
scoresFo = cross_val_score(forests, atributos, target, cv=5, scoring='accuracy')

#Porcentajes de acierto
print("Usando EM meta-estimator se tiene una tasa de acierto del ",np.mean(scoresBag)*100)
print("Usando EM Random Forests se tiene una tasa de acierto del ",np.mean(scoresFo)*100)

#Matrices de validación
print("Matriz EM - Nmeta-estimatoru: ",matrizCruzada(preBag))
print("Matriz EM - Random Forests: ",matrizCruzada(preFo))


#Quinto algoritmo Redes neuronales

#MLPClassifier
modelMLP = MLPClassifier()
modelMLP.fit(data_test, target_test)
preMLP=modelMLP.predict(data_test)
scoreMLP = cross_val_score(modelMLP, atributos, target, cv=5, scoring='accuracy')

#KNC
KNC = KNeighborsClassifier()
KNC.fit(data_test,target_test)
preKNC=KNC.predict(data_test)
scoreKNC = cross_val_score(KNC, atributos, target, cv=5, scoring='accuracy')

#Porcentajes de acierto
print("Usando RN MLPClassifier se tiene una tasa de acierto del ",np.mean(scoreMLP)*100)
print("Usando RN KNC se tiene una tasa de acierto del ",np.mean(scoreKNC)*100)


#Matrices de validación
print("Matriz RN MLPClassifier: ",matrizCruzada(preMLP))
print("Matriz RN KNC: ",matrizCruzada(preKNC))


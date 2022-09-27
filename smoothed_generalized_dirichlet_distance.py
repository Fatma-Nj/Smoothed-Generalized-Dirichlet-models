# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 2020

@author: Fatma Najar
--Generalized smoothed dirichlet mixture models 
--Clustering count data using geometrical distance:KL, fisher, bhattacharya
"""
###################################
#Imports libraries
###################################
from wordcloud import WordCloud, STOPWORDS
import matplotlib 
import matplotlib.pyplot as plt
from time import time
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score # for classification
from coclust.evaluation.external import accuracy   #for clustering
from statistics import stdev
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score
import seaborn as sns; sns.set()
from sklearn.cluster import AgglomerativeClustering
##################################
# Defining functions
##################################


""" 
Initilize parameters
"""

def initialize_parameters(data):
    D = data.shape
    D = int(D[0])
    alpha =  np.zeros((D-1,1))
    beta =   np.zeros((D-1,1))
    A = np.ones((D-1,1))
    B = np.ones((D-1,1))
    a = np.zeros((D-1,1))
    mu = np.mean(data,axis=0) + 1e-10
    
    
    S = np.std(data,axis=0) + 1e-10
    a[0] = 1
    B[0] = A[0] = 1
    alpha [0] = np.divide( mu ** 2 - mu  * (S  + mu  **2), (S + mu  **2) -  mu )
    beta [0]  = np.divide(alpha[0] * (1 - mu ), mu )
    eps = 1e-10
    
    for d in range(1,D-1):
        a[d] = mu / ( B[d-1] + eps)
        alpha [d] = np.divide(a[d] * mu * B[d-1] - mu * (S + mu **2), (A[d-1] * (S + mu **2) - a[d] * mu * B[d-1])+ 1e-10)
        beta [d]  = np.divide(alpha[d] * (A[d-1] - mu), mu)
        
        A[d] = A[d-1] * (beta[d] / (alpha[d] + beta[d]))
        B[d] = B[d-1] * (np.divide(beta[d] * (beta[d]+1), (alpha[d]+beta[d]) * (alpha[d] + beta[d]+1)))
    
    return abs(alpha), abs(beta) 


def generalized_smoothed_dirichlet(p, alpha, beta):
    
   
    " get the size of words"
    D = p.shape
    D = int(D[0])
    beta_s = 0.01
    lamda = 0.3 #smoothing parameter

    steps = 1
    " smoothing the words"
    
    #x_u = p / (np.sum(p,axis=1) + 1e-10)
    x_G = np.divide(np.sum(p,axis=0) + beta_s , np.sum(p) + D * beta_s )
    x = p * lamda + x_G * (1 - lamda)

    for iter in range(steps):
        
        "calculate new mixing weight"
          
        " calcualte new alpha parameter"
        alpha_new = np.zeros((D-1,1))
        for d in range(D-1):
                alpha_new[d] = np.sum(  (np.divide(x[d],1-x[d])) * beta[d]) 
                alpha_new[d] = abs(alpha_new[d])
        " calcualte new beta parameter" 
        beta_new = np.zeros((D-1,1))
        
        delta = np.zeros((D-1,1))
         
        for d in range(D-1):
            sum_x = np.sum(x[0:d]) 
            beta_new[d] = np.sum( (np.divide(abs(1 - sum_x), 1e-10 + abs(sum_x))) * alpha[d]) 
            beta_new[d] = abs(beta_new[d])
                # if np.isnan(beta_new[d,j]):
                #     beta_new[d,j] = 1e10
       
        """ 
         Update parameters 
        """
      

        alpha = alpha_new
        beta = beta_new

        
        
        
    
    return alpha_new, beta_new
    
def KL_SD (alpha1, beta1, alpha2, beta2):
    
    sum_alpha1 = np.sum(alpha1 * np.log(alpha1))
    sum_alpha2 = np.sum(alpha2 * np.log(alpha2))
    
    sum_beta1 = np.sum(beta1 * np.log(beta1))
    sum_beta2 = np.sum(beta2 * np.log(beta2))
    
    sum_1 = np.sum((alpha1 + beta1) * np.log(alpha1+ beta1))
    sum_2 = np.sum((alpha2 + beta2) * np.log(alpha2 + beta2))
    
    KL = sum_1 - sum_2 + sum_alpha2 + sum_beta2 - sum_alpha1 - sum_beta1 \
       + np.sum((alpha1 - alpha2) * (np.log(alpha1 / (alpha1+beta1))) \
       +     (beta1 - beta2) * (np.log(beta1 / (alpha1+beta1))) )
    #- np.log(np.sum(alpha1)))
    return KL

def Bhatt(alpha1, beta1, alpha2, beta2):
    sum_alpha1 = np.sum(alpha1 * np.log(alpha1))
    sum_alpha2 = np.sum(alpha2 * np.log(alpha2))
    
    sum_beta1 = np.sum(beta1 * np.log(beta1))
    sum_beta2 = np.sum(beta2 * np.log(beta2))
    
    sum_1 = np.sum((alpha1 + beta1) * np.log(alpha1+ beta1))
    sum_2 = np.sum((alpha2 + beta2) * np.log(alpha2 + beta2))
    
    BH = 0.5 * (sum_1 + sum_2 - sum_alpha1 - sum_alpha2 - sum_beta1 - sum_beta2) \
        + np.sum(.5 * (alpha1 + beta1 + alpha2 + beta2) * np.log((alpha1 + beta1 + alpha2 + beta2)/2) \
        - (.5 * (alpha1 + alpha2) * np.log((alpha1+ alpha2)/2)) - (.5 * (beta1 + beta2) * np.log((beta1+ beta2)/2)))    
    return  BH
##########################################
    # Main algorithm
##########################################


""" 

Algorihtm: Hieracrchical Clustering based on generalized smoothed Dirichlet mixture models

"""


##################################
#Initial parameters
##################################

   
print("Loading dataset...")

data = pd.read_csv("2013_Pakistan_eq_CF_labeled_data.tsv",sep="\t")
data_samples = data['tweet_text']         
label_gd = data['label']



# First, we extract the count data,

vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=50)
x_fits = vectorizer.fit_transform(data_samples)

x_counts = vectorizer.transform(data_samples).toarray()

# Next, we set a TfIdf Transformer, and transform the counts with the model.

transformer = TfidfTransformer(smooth_idf=False)
x_tf = transformer.fit_transform(x_counts) # matrix of size (documents x features)
x_tfidf = transformer.transform(x_tf).toarray()


print(x_tfidf.shape[0]) ## the number of documents

print(x_tfidf.shape[1]) ## the number of features

             
# initialize alpha and beta as the initialization of generalized Dirichlet
alpha_data = []
beta_data = []
N, D = x_counts.shape
t0 = time()
for i in range(N):
    "initialize alpha parameters for each cluster"

    alpha_i, beta_i =  initialize_parameters(x_tfidf[i])

    " training the data with SGD mixture models for each text/image"

    alpha_gSD, beta_gSD = generalized_smoothed_dirichlet(x_tfidf[i], alpha_i, beta_i)
    " normalizing alpha and beta parameters"
    alpha_gSD = np.divide(alpha_gSD, np.sum(alpha_gSD, axis=0)) 
    beta_gSD  = np.divide(beta_gSD, np.sum(beta_gSD, axis=0))       

    #saving parameters for each image
    alpha_data.append(alpha_gSD)
    beta_data.append(beta_gSD)
    
      

" Hierarchical clustering using KL "

KL_distance = np.zeros((N,N))
Fisher_distance = np.zeros((N,N))
Bhatt_distance = np.zeros((N,N))
for i in range(N):
    for k in range(N):
        KL_distance[i,k] = KL_SD(alpha_data[i], beta_data[i], alpha_data[k], beta_data[k])  
        Fisher_distance[i,k] = math.sqrt(2) * math.sqrt(abs(KL_distance[i,k])) 
        Bhatt_distance[i,k] =  Bhatt(alpha_data[i], beta_data[i], alpha_data[k], beta_data[k]) 
# Perform agglomerative clustering.
# The affinity is precomputed (since the distance are precalculated).
agg = AgglomerativeClustering(n_clusters=9, affinity='precomputed', linkage='average')

# Use the distance matrix directly.
predict_KL = agg.fit_predict(KL_distance)  
predict_Fisher = agg.fit_predict(Fisher_distance)
predict_Bhatt = agg.fit_predict(Bhatt_distance)  
print("done in %0.3fs." % (time() - t0))
label = []  

for i in range(len(label_gd)):
    if (label_gd[i]=="injured_or_dead_people"):
        label.append(0)
    if (label_gd[i]=="missing_trapped_or_found_people"):
        label.append(1)
    if (label_gd[i]=="displaced_people_and_evacuations"):
        label.append(2)
    if (label_gd[i]=="infrastructure_and_utilities_damage"):
        label.append(3)
    if (label_gd[i]=="donation_needs_or_offers_or_volunteering_services"):
        label.append(4)
    if (label_gd[i]=="caution_and_advice"):
        label.append(5)
    if (label_gd[i]=="sympathy_and_emotional_support"):
        label.append(6)
    if (label_gd[i]=="other_useful_information"):
        label.append(7)
    if (label_gd[i]=="not_related_or_irrelevant"):
        label.append(8)
        
        
 
" Evaluation metrics "
"Accuracy"
accuracy_KL = accuracy(label, predict_KL)
accuracy_Fisher = accuracy(label, predict_Fisher)
accuracy_Bhatt =  accuracy(label, predict_Bhatt)
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:22:03 2020

@author: Fatma Najar
--smoothed dirichlet multinomial mixture models
Clustering count data 
"""
###################################
#Imports libraries
###################################
import numpy as np
import pandas as pd
import scipy.special as sc
from coclust.evaluation.external import accuracy   #for clustering
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score,  balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import precision_recall_curve, f1_score
from guppy import hpy
##################################
# Defining functions
##################################


""" 
Initilize parameters
"""

def initialize_alpha_parameters(data):
    N,D = data.shape
    alpha = np.zeros((D,1))
    X = np.sum(data,axis=0) / N
    X_1 = np.sum(data[:,0]) /N
    X_2 = np.sum(data[:,0]**2 ) / N
    
    X_s = np.sum(X) - X[D-1]
#
    
    for d in range(D-2):
        alpha [d]= ((X_1 - X_2) * (X[d] + 1e-10 )) / ( X_2 - X_1 **2 +1e-10)   
    alpha[D-1] = ((X_1 - X_2) * (1- X_s + 1e-10)) / ( X_2 - X_1 **2 + 1e-10) 
    alpha = np.absolute(alpha)
    # alpha = np.repeat(alpha,K)  
    # alpha = np.reshape(alpha, (D,K))
    
    return alpha
 
def pdf_smoothed_dir_multi(x,alpha):
    N,D = x.shape
    _,K = alpha.shape
    """
           probability distribution function for each data point 
                       SD mixture model
    """
 

    n = np.sum(x)
        
    pdf = np.zeros((N,K))    
    logl = sc.gammaln(n+1) - np.sum(sc.gammaln(x+1),axis=1)          
    
    S = np.sum(alpha,axis=0)
    for i in range(N):
        for j in range(K):  
            for d in range(D-1):
                sum_x = np.sum(x, axis=1)
                pdf[i,j] = pdf[i,j] + (S[j]) * np.log(S[j])\
                    + ((alpha[d,j]+x[i,d]) * np.log(alpha[d,j]+x[i,d] + 1e-10)) \
                    - alpha[d,j] * np.log(alpha[d,j]+1e-10)   + (S[j] + sum_x[i]) * np.log(S[j] + sum_x[i]) 
            if np.isnan(pdf[i,j]):
                pdf[i,j] = 1e10
            pdf[i,j] += (logl[i])
    return pdf


def smoothed_dir_multinomial(x, pis, alpha, tol):
    
   
    " get the size of words"
    N,D = x.shape
    K=len(pis)
    log_like = []

    steps = 1
    conv = 10


    for iter in range (steps):
    #while(conv > tol):   
        "calculate the pdf for smoothed dirichlet"
        pdf = pdf_smoothed_dir_multi(x,alpha)        
        "calclate the log-likelihood" 
        
        log_like = np.sum(np.log(np.sum(np.log(pis) + pdf,axis=1)+1e-10),axis=0)
        r = np.zeros((N,K))
   
        """
          E-step: calculate the responsibilites
        """
        for i in range(len(r)):
            for j in range(len(r[i])):
                r[i,j] = np.divide(pis[j] * pdf[i,j], np.sum(np.multiply(pdf, pis),axis=1)[i] )
  
   
        """
          M-step: maxmize parameters
        """
        
        "calculate new mixing weight"
        pis_new =[]
        m = np.sum(r,axis=0)
        
        pis_new = m/N # for each cluster, calcualte the fraction of poinst which belongs to cluster c  
        
        " calcualte new alpha parameter using Newton-raphson"
        alpha_new = np.zeros((D,K))
   
        
        S = np.sum(alpha,axis=0)
        "Inverse of Hessian matrix"
        for j in range(len(pis)):
            for d in range(D-1):
                sum_x = np.sum(x, axis=1)
                f_1 = np.sum(r[:,j] * np.log(S[j]) - np.log(alpha[d,j]+1e-10) \
                    - np.divide(S[j] + np.sum(x), alpha[d,j]+x[:,d]+1e-10))
                    
                f_2 = np.sum(r[:,j] *np.divide(alpha[d,j] - S[j] - sum_x, (alpha[d,j] +1e-10) * (S[j] + sum_x)) \
                    -  np.divide(S[j] + np.sum(x), (alpha[d,j]+x[:,d] +1e-10) ** 2))
                # f_2 = np.sum(r[:,j] * (np.divide(1,S[j]) - np.divide(1, alpha[d,j] + 1e-10) \
                #     + np.divide(1,alpha[d,j]+x[:,d]+1e-10) - np.divide(1, S[j]+ sum_x + 1e-10))  )
                alpha_new[d,j] = alpha[d,j] - np.divide(f_1,f_2)
        """
           Evaluate the log-likelihood
        """
      
        new_log_like = np.sum(np.log(np.sum(pis_new * pdf,axis=1)),axis=0)
      
      
        """ 
          Check convergence
        """
      
        change = abs(new_log_like - log_like) 
        
        alpha = alpha_new
        pis = pis_new
        log_like = new_log_like
        conv = .1
        
    
    return abs(alpha_new), pis_new
    

##########################################
    # Main algorithm
##########################################


""" 

Algorihtm: Clustering count data based on smoothed Dirichlet multinomial

"""


##################################
#Initial parameters
##################################

   
print("Loading train dataset...")


data = pd.read_csv("2013_Pakistan_eq_CF_labeled_data.tsv",sep="\t")
data_samples = data['tweet_text'] 
label_gd = data['label']

"######################################"

        

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




# First, we extract the count data,

vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=100)
x_fits = vectorizer.fit_transform(data_samples)

x_counts = vectorizer.transform(data_samples).toarray()

# Next, we set a TfIdf Transformer, and transform the counts with the model.

transformer = TfidfTransformer(smooth_idf=False)
x_tf = transformer.fit_transform(x_counts) # matrix of size (documents x features)
x_tfidf = transformer.transform(x_tf).toarray()


print(x_tfidf.shape[0]) ## the number of documents

print(x_tfidf.shape[1]) ## the number of features

K = 8  # number of mixture components
N, D = x_counts.shape
"initialize mixing weight"
pis = np.ones (K) / K
"K-means clustering"
t0 = time()
kmeans = KMeans(n_clusters=K).fit(x_counts)
index  = kmeans.labels_

"initialize alpha parameters for each cluster"
alpha_i = np.zeros((D,0))


for j in range(K):
    alpha =  initialize_alpha_parameters(x_counts[index==j,:])
    alpha_i = np.append(alpha_i, alpha, axis=1)

" training the data with SGD mixture models"
    
alpha_SDM, pis_SDM = smoothed_dir_multinomial(x_counts, pis, alpha_i, tol=0.9)
" normalizing alpha and beta parameters"
alpha_SDM = np.divide(alpha_SDM, np.sum(alpha_SDM, axis=0))   
time_elapsed = time() - t0
print("done in %0.3fs." % (time() - t0))
h=hpy()
h.heap()
" clustering using Bayes rule"


pdf_SDM = pdf_smoothed_dir_multi(x_counts,alpha_SDM) 

posterior = pdf_SDM 
#+ np.log(pis_SDM)



label_predicted = np.zeros((N,))
for i in range(N):
    (m,label_predicted[i]) = max((v,index) for index,v in enumerate(posterior[i,:]))



  
" Evaluation metrics "
"average precision score"
from sklearn.metrics import average_precision_score
mAP = average_precision_score(label_predicted,label)
"Accuracy"
accuracy_SDM = accuracy(label, label_predicted)





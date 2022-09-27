# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:22:03 2020

@author: Fatma Najar
--Generalized smoothed dirichlet multinomial mixture models
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
import scipy.io
##################################
# Defining functions
##################################


""" 
Initilize parameters
"""

def initialize_parameters(data):
    N,D = data.shape
    alpha =  np.zeros((D-1,1))
    beta =   np.zeros((D-1,1))
    A = np.ones((D-1,1))
    B = np.ones((D-1,1))
    a = np.zeros((D-1,1))
    mu = np.mean(data,axis=0) + 1e-10
    
    
    S = np.std(data,axis=0) + 1e-10
    a[0] = 1
    B[0] = A[0] = 1
    alpha [0] = np.divide( mu[0] ** 2 - mu[0]  * (S[0]  + mu[0]  **2), (S[0]  + mu[0]  **2) -  mu[0] )
    beta [0]  = np.divide(alpha[0] * (1 - mu[0] ), mu[0] )
    eps = 1e-10
    
    for d in range(1,D-1):
        a[d] = mu[d] / ( B[d-1] + eps)
        alpha [d] = np.divide(a[d] * mu[d] * B[d-1] - mu[d] * (S[d] + mu[d] **2), (A[d-1] * (S[d] + mu[d] **2) - a[d] * mu[d] * B[d-1])+ 1e-10)
        beta [d]  = np.divide(alpha[d] * (A[d-1] - mu[d]), mu[d])
        
        A[d] = A[d-1] * (beta[d] / (alpha[d] + beta[d]))
        B[d] = B[d-1] * (np.divide(beta[d] * (beta[d]+1), (alpha[d]+beta[d]) * (alpha[d] + beta[d]+1)))
    
    return abs(alpha), abs(beta) 
def pdf_gen_smoothed_multi(x,alpha,beta):
    N,D = x.shape
    _,K = alpha.shape
    """
           probability distribution function for each data point 
                       GSDM mixture model
    """
 

    n = np.sum(x)
        
    pdf = np.zeros((N,K))    
    logl = sc.gammaln(n+1) - np.sum(sc.gammaln(x+1),axis=1)          
    
    
    delta = np.zeros((D-1,K))
    for i in range(N):
        for j in range(K):  
            for d in range(D-1):
                sum_x = np.sum(x[:,d+1:D], axis=1)
                # pdf[i,j] = pdf[i,j] * (  ((alpha[d,j] + beta[d,j]) ** (alpha[d,j] + beta[d,j])) \
                #     * ((alpha[d,j]+x[i,d]) ** (alpha[d,j]+x[i,d])) * ((beta[d,j]+sum_x[i]) ** (beta[d,j]+sum_x[i])))\
                #     / ( (alpha[d,j] ** alpha[d,j]) * (beta[d,j] ** beta[d,j]) * (alpha[d,j]+beta[d,j]+x[i,d]\
                #     + sum_x[i]) ** (alpha[d,j]+beta[d,j]+x[i,d]+ sum_x[i]) )
                pdf[i,j] = pdf[i,j] + (alpha[d,j] + beta[d,j]) * np.log((alpha[d,j] + beta[d,j]))\
                    + ((alpha[d,j]+x[i,d]) * np.log(alpha[d,j]+x[i,d])) + (beta[d,j]+sum_x[i]) * np.log( beta[d,j]+sum_x[i])\
                    - alpha[d,j] * np.log(alpha[d,j]) - beta[d,j] * np.log(beta[d,j]) - (alpha[d,j]+beta[d,j]+x[i,d]\
                    + sum_x[i]) * np.log(alpha[d,j]+beta[d,j]+x[i,d]+ sum_x[i]) 
            if np.isnan(pdf[i,j]):
                pdf[i,j] = 1e10
        #pdf[i,j] += (logl[i])
    return pdf


def generalized_smoothed_multinomial(x, pis, alpha, beta, tol):
    
   
    " get the size of words"
    N,D = x.shape
    K=len(pis)
    log_like = []
 
    steps = 1
    conv = 10


    for iter in range (steps):
    #while(conv > tol):   
        "calculate the pdf for smoothed dirichlet"
        pdf = pdf_gen_smoothed_multi(x,alpha,beta)        
        "calclate the log-likelihood" 
        pdf = np.exp(pdf)      
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
        
        " calcualte new alpha and beta parameter using Newton-raphson"
        alpha_new = np.zeros((D-1,K))
        beta_new = np.zeros((D-1,K))
        
        "Inverse of Hessian matrix"
        for j in range(len(pis)):
            for d in range(D-1):
                D_jd = np.diag([1/(np.sum(r[:,j] /(alpha[d,j] + x[:,d])) \
                    - np.sum(r[:,j] /alpha[d,j])) , 1/ (np.sum(r[:,j] /(beta[d,j] + np.sum(x[:,d+1]))) \
                    - np.sum(r[:,j] /beta[d,j]))])
                H_jd = D_jd + np.diag(D_jd) * (np.sum(r[:,j] /(alpha[d,j] + x[:,d] + beta[d,j] + np.sum(x[:,d+1:D]))) \
                    - np.sum(r[:,j]) /(alpha[d,j] + beta[d,j])) * ( 1/  1 + ( (np.sum(r[:,j]) /(alpha[d,j]+beta[d,j]) \
                    - np.sum(r[:,j] /(alpha[d,j] + x[:,d] + beta[d,j] + np.sum(x[:,d+1:D]))))/ (np.sum(r[:,j] /(alpha[d,j] + x[:,d])) \
                    - np.sum(r[:,j]) /(alpha[d,j]))  ) + ( (np.sum(r[:,j]) /(alpha[d,j]+beta[d,j]) \
                    - np.sum(r[:,j] /(alpha[d,j] + x[:,d] + beta[d,j] + np.sum(x[:,d+1:D]))) )/ ( np.sum(r[:,j]) /(beta[d,j] + np.sum(x[:,d+1:D])) \
                    - np.sum(r[:,j]) /(beta[d,j])) ) ) 
                                                                                                                 
                Gradient_jd = np.matrix([(np.log(alpha[d,j] + beta[d,j]) - np.log(alpha[d,j])) * np.sum(r[:,j]) \
                            + np.sum(r[:,j] * (np.log(alpha[d,j] + x[:,d])) - np.log(alpha[d,j] + x[:,d] \
                            + beta[d,j] + np.sum(x[:,d+1:D]))), (np.log(alpha[d,j] + beta[d,j]) - np.log(beta[d,j])) * np.sum(r[:,j]) \
                            + np.sum(r[:,j] * (np.log(beta[d,j] + np.sum(x[:,d+1:D])) - np.log(alpha[d,j] + x[:,d] \
                            + beta[d,j] + np.sum(x[:,d+1:D]))))]) 
                
                Theta = np.subtract(np.matrix([alpha[d,j], beta[d,j]]), np.dot(Gradient_jd,H_jd))
  
                alpha_new[d,j] = Theta[0,0]
                beta_new[d,j] = Theta[0,1]
        """
           Evaluate the log-likelihood
        """
      
        new_log_like = np.sum(np.log(np.sum(pis_new * pdf,axis=1)),axis=0)
      
      
        """ 
          Check convergence
        """
      
        change = abs(new_log_like - log_like) 
        
        alpha = alpha_new
        beta = beta_new
        pis = pis_new
        log_like = new_log_like
        conv = change
        
    
    return abs(alpha_new), abs(beta_new), pis_new
    

##########################################
    # Main algorithm
##########################################


""" 

Algorihtm: Clustering count data based on generalized smoothed Dirichlet mixture models

"""


##################################
#Initial parameters
##################################

   
print("Loading dataset...")
##########################################################


data = pd.read_csv("2013_Pakistan_eq_CF_labeled_data.tsv",sep="\t")
data_samples = data['tweet_text']         
label_gd = data['label']

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

             
# initialize alpha and beta as the initialization of generalized Dirichlet
K = 8  # number of mixture components
N, D = x_counts.shape
"initialize mixing weight"
pis = np.ones (K) / K
"K-means clustering"
t0 = time()
kmeans = KMeans(n_clusters=K).fit(x_counts)
index  = kmeans.labels_

"initialize alpha parameters for each cluster"
alpha_i = np.zeros((D-1,0))
beta_i  = np.zeros((D-1,0))

for j in range(K):
    alpha, beta =  initialize_parameters(x_counts[index==j,:])
    alpha_i = np.append(alpha_i, alpha, axis=1)
    beta_i = np.append(beta_i, beta, axis=1)

" training the data with SGD mixture models"
    
alpha_SGDM, beta_SGDM, pis_SGDM = generalized_smoothed_multinomial(x_counts, pis, alpha_i, beta_i, tol=0.9)
" normalizing alpha and beta parameters"
alpha_SGDM = np.divide(alpha_SGDM, np.sum(alpha_SGDM, axis=0)) 
beta_SGDM  = np.divide(beta_SGDM, np.sum(beta_SGDM, axis=0))       

" clustering using Bayes rule"



pdf_SGDM = pdf_gen_smoothed_multi(x_counts,alpha_SGDM,beta_SGDM) 

posterior = pdf_SGDM + np.log(pis_SGDM)


print("done in %0.3fs." % (time() - t0))
label_predicted = np.zeros((N,))
for i in range(N):
    (m,label_predicted[i]) = max((v,index) for index,v in enumerate(posterior[i,:]))


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
accuracy_SGDM = accuracy(label, label_predicted)
"Weighted accuracy"
weight_acc_gsd =  balanced_accuracy_score(label, label_predicted)
" Area Under ROC Curve"
#roc_gsd =  roc_auc_score(label, label_predicted, multi_class = 'ovo')
"confusion matrix"
cm = confusion_matrix(label, label_predicted)






# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 2020

@author: Fatma Najar

--Generalized smoothed dirichlet mixture models

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
def pdf_gen_smoothed_dir(x,alpha,beta):
    N,D = x.shape
    _,K = alpha.shape
    """
           probability distribution function for each data point 
                       GSD mixture model
    """
 

             
    pdf = np.ones((N,K))
    P = np.ones((N,K))
    delta = np.zeros((D-1,K))
    for j in range(K):  
        ## gamma parameter
        for d in range(D-2):
            delta[d,j] = beta[d,j] - alpha[d+1,j] - beta[d+1,j]
        delta[D-2,j] = beta[D-2,j] -1
        for i in range(N):
            for d in range(D-1):
                sum_x = np.sum(x[:,0:d], axis=1)
                P[i,j] =  (x[i,d] ** (alpha[d,j] - 1)) * ((abs( 1- sum_x[i]) ** delta[d,j]))
                #P[i,j] =  (alpha[d,j] - 1) * np.log(x[i,d]+ 1e-10) +  delta[d,j] * np.log( 1- sum_x[i])
                pdf[i,j] = pdf[i,j] * ( ((alpha[d,j] + beta[d,j]) ** (alpha[d,j] + beta[d,j])) /( (alpha[d,j] ** alpha[d,j]) * (beta[d,j] ** beta[d,j]) )) * P[i,j]
                #pdf[i,j] = pdf[i,j] + P[i,j] + (alpha[d,j] + beta[d,j]) * np.log((alpha[d,j] + beta[d,j])) - (alpha[d,j]) * np.log(alpha[d,j]) - (beta[d,j]) * np.log( beta[d,j])
            if np.isnan(pdf[i,j]):
                pdf[i,j] = 1e10
            
    return pdf


def generalized_smoothed_dirichlet(p, pis, alpha, beta, tol):
    
   
    " get the size of words"
    N,D = p.shape
    K=len(pis)
    log_like = []
    beta_s = 0.01
    lamda = 0.9 #smoothing parameter
    conv = 1
    steps = 1
    " smoothing the words"
    
    #x_u = p / (np.sum(p,axis=1) + 1e-10)
    x_G = np.divide(np.sum(p,axis=0) + beta_s , np.sum(p) + D * beta_s )
    x = p * lamda + x_G * (1 - lamda)

    for iter in range(steps):
        
        "calculate the pdf for smoothed dirichlet"
        pdf = pdf_gen_smoothed_dir(x,alpha,beta)        
        "calclate the log-likelihood" 
                  
        """
          E-step: calculate the responsibilites
        """
        r = np.zeros((N,K))
        for i in range(len(r)):
            for j in range(len(r[i])):
                r[i,j] = np.divide(pis[j] * pdf[i,j], np.sum(np.multiply(pdf, pis),axis=1)[i] )
  
   
        """
          M-step: maxmize parameters
        """
        
        "calculate new mixing weight"
        pis_new =[]
        m = np.sum(r,axis=0)
        
        pis_new = m/N # for each cluster, calcualte the fraction of points which belongs to cluster c  
        """
         maxmize parameters
        """
     
        " calcualte new alpha parameter"
        alpha_new = np.zeros((D-1,K))
        for j in range(len(pis)):
            for d in range(D-1):
                alpha_new[d,j] = np.sum(  (np.divide(x[:,d],1-x[:,d])) * beta[d,j]) 
      
        " calcualte new beta parameter" 
        beta_new = np.zeros((D-1,K))
        
        delta = np.zeros((D-1,K))
        for j in range(len(pis)):
            
            # ## delta parameter
            # for d in range(D-2):
            #    delta[d,j] = beta[d,j] - alpha[d+1,j] - beta[d+1,j]
            # delta[D-2,j] = beta[D-2,j] -1
            
            
            for d in range(D-1):
                sum_x = np.sum(x[:,0:d],axis=1) 
                beta_new[d,j] = np.sum( (np.divide(abs(1 - sum_x), 1e-10 + abs(sum_x))) * alpha[d,j]) 
                beta_new[d,j] = abs(beta_new[d,j])
                # if np.isnan(beta_new[d,j]):
                #     beta_new[d,j] = 1e10
       
        """ 
         Update parameters 
        """
      

        alpha = alpha_new
        beta = beta_new
        pis = pis_new
        
        
        
    
    return alpha_new, beta_new, pis_new
    

##########################################
    # Main algorithm
##########################################


""" 

Algorihtm: Clustering count data based on generalized smoothed Dirichlet mixture models

"""


   
print("Loading dataset...")
t0 = time()




data = pd.read_csv("2014_India_floods_CF_labeled_data.tsv",sep="\t")
data_samples = data['tweet_text'] 
label_gd = data['label']


        
"########################################"



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

vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=10)
x_fits = vectorizer.fit_transform(data_samples)

x_counts = vectorizer.transform(data_samples).toarray()

# Next, we set a TfIdf Transformer, and transform the counts with the model.

transformer = TfidfTransformer(smooth_idf=False)
x = transformer.fit_transform(x_counts) # matrix of size (documents x features)
x_tfidf = transformer.transform(x).toarray()



##################################
#Initial parameters
##################################
            
# initialize alpha and beta as the initialization of generalized Dirichlet
K = 9  # number of mixture components





N, D = x_tfidf.shape
"initialize mixing weight"
pis = np.ones (K) / K
"K-means clustering"
kmeans = KMeans(n_clusters=K).fit(x_tfidf)
index  = kmeans.labels_

"initialize alpha parameters for each cluster"
alpha_i = np.zeros((D-1,0))
beta_i  = np.zeros((D-1,0))

for j in range(K):
    alpha, beta =  initialize_parameters(x_tfidf[index==j,:])
    alpha_i = np.append(alpha_i, alpha, axis=1)
    beta_i = np.append(beta_i, beta, axis=1)

" training the data with SGD mixture models"
    
alpha_gSD, beta_gSD, pis_gSD= generalized_smoothed_dirichlet(x_tfidf, pis, alpha_i, beta_i, tol=0.9)
" normalizing alpha and beta parameters"
alpha_gSD = np.divide(alpha_gSD, np.sum(alpha_gSD, axis=0)) 
beta_gSD  = np.divide(beta_gSD, np.sum(beta_gSD, axis=0))       

" clustering "

" smoothing the words"
beta_s = 0.01
lamda = 0.9 #smoothing parameter
    
 
x_G = np.divide(np.sum(x_tfidf,axis=0) + beta_s , np.sum(x_tfidf) + D * beta_s )
x = x_tfidf * lamda + x_G * (1 - lamda)
pdf = np.ones((N,K))
P = np.ones((N,K))
delta = np.zeros((D-1,K))
for j in range(K):  
        ## gamma parameter
    for d in range(D-2):
        delta[d,j] = beta_gSD[d,j] - alpha_gSD[d+1,j] - beta_gSD[d+1,j]
    delta[D-2,j] = beta_gSD[D-2,j] -1
    for i in range(N):
        for d in range(D-1):
            sum_x = np.sum(x[:,0:d], axis=1)
            P[i,j] =  (alpha_gSD[d,j] - 1) * np.log(x[i,d]+ 1e-10) +  delta[d,j] * np.log((1- sum_x[i]) + 1e-10)
            pdf[i,j] = pdf[i,j] + P[i,j]  + (alpha_gSD[d,j] + beta_gSD[d,j]) * np.log((alpha_gSD[d,j] + beta_gSD[d,j]))- (alpha_gSD[d,j]) * np.log(alpha_gSD[d,j]) - (beta_gSD[d,j]) * np.log(beta_gSD[d,j])
# 

posterior = pdf + np.log(pis_gSD) 
#posterior = np.divide(posterior, np.sum(posterior))

label_predicted = np.zeros((N,))
for i in range(N):
    (m,label_predicted[i]) = max((v,index) for index,v in enumerate(posterior[i,:]))




  
" Evaluation metrics "
"Accuracy"
accuracy_gsd = accuracy(label, label_predicted)
"confusion matrix"
cm = confusion_matrix(label, label_predicted)

from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

indexes = linear_assignment(_make_cost_m(cm))
js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
cm2 = cm[:, js]

ax = sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")


"""
Bag-of-words######################################
"""


emotion_wc = WordCloud(background_color="white",width = 512,height = 512, collocations=False)
emotion_wc.generate(str(vectorizer.vocabulary_ ))
plt.figure(figsize = (10, 8))
#, facecolor = 'k'
plt.imshow(emotion_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:39:25 2020

@author: toztel17
"""
import pystan
import pandas as pd
import os
import numpy as np

os.chdir('C:\\Users\\toztel17\\Desktop\\python hm3') # change current working directory
data = pd.read_csv('trend2.csv') # load data as a dataframe

dv = np.array(data['church2'])
iv = np.array(data['gini_net'])
control = np.array(data['rgdpl'])
country = np.array(data['country'])
year = np.array(data['year'])

all_data_comb = np.vstack([dv,iv,control,year,country]).T
all_data_del = pd.DataFrame(all_data_comb).dropna(axis=0,how='any') # listwise deletion performed
all_data_del.columns = ['dv','iv','control',
                     'year','country']


 
country = np.array(all_data_del['country']) #otherwise, the indexing is weird
year = np.array(all_data_del['year'])
nrows = len(year)

ones_col = np.ones((nrows,1)).T

# combining two IVs in one matrix
exp_data_mat = np.vstack([ones_col,all_data_del['iv'],all_data_del['control']]).T #first column is iv, sec is control

exp_data_mat = pd.DataFrame(exp_data_mat)
exp_data_mat.columns=['ones_col','iv','control']

nUnique_country = max(pd.Categorical(country).codes)
country_labels = pd.Categorical(country).codes
nUnique_year = max(pd.Categorical(year).codes)
year_labels = pd.Categorical(year).codes

hierarchical_intercept = """
data {
  int<lower=0> nUnique_country; // this is J1
  int<lower=0> nUnique_year; // this is J2
  int<lower=0> N; //observations
  int<lower=1,upper=nUnique_country+1> nCountry[N]; //group sizes
  int<lower=1,upper=nUnique_year+1> nYear[N]; //
  matrix[N,3] x;
  vector[N] y;
} 
parameters {
  vector[nUnique_country] a1;
  vector[nUnique_year] a2;
  vector[3] b;
  real mu_a1;
  real mu_a2;
  real<lower=0> sigma_a1;
  real<lower=0> sigma_a2;
  real<lower=0> sigma_y; // ,upper=100> kısmını sildim
} 
transformed parameters {
  vector[N] y_hat;
  vector[N] m;

  for (i in 1:N) {
    m[i] <- a1[nCountry[i]] + a2[nYear[i]] + x[i,1] * b[1];
    y_hat[i] <- m[i] + x[i,2] * b[2];
  }
}
model {
  mu_a1 ~ normal(0,1); #1 yerine daha büyük bir sayı yazabilirsin uninformative olması için
  mu_a2 ~ normal(0,1);
  sigma_a1 ~ inv_gamma(1,.5);
  sigma_a2 ~ inv_gamma(1,.5);
  a1 ~ normal(mu_a1, sigma_a1);
  a2 ~ normal(mu_a2,sigma_a2);
  b ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
"""


# country and year should be random intercepts


dv = np.array(all_data_del['dv'])
country= np.array(all_data_del['country'])
year = list(all_data_del['year'])



hierarchical_intercept_data = {'N': len(dv),
                          'nUnique_country': nUnique_country+1, #len(country), #unique
                          'nUnique_year' : nUnique_year+1,
                          'nCountry': country_labels+1, # Stan counts starting at 1
                          'nYear': year_labels+1, #random intercept
                          'x': np.array(exp_data_mat[['ones_col','iv','control']], dtype='float'),
                          'y': np.array(dv, dtype = 'float')}

hierarchical_intercept_fit = pystan.stan(model_code=hierarchical_intercept, 
                                         data=hierarchical_intercept_data, 
                                         iter=1000, chains=2)

#b = hierarchical_intercept_fit['b'].mean(axis=0)


hierarchical_intercept_fit #print the whole summary, print prints the whole
 

# b2 mean = 0.57 , se =8.5e-3, sd = 0.07 we are interested in this variable

#hierarchical_intercept_fit2 = pystan.StanModel(model_code=hierarchical_intercept)

# posterior_sampling = hierarchical_intercept_fit2.sampling(data=hierarchical_intercept_data, iter=1000, chains=4);

# print(posterior_sampling) # give us the posterior summary


#%% rerun the model w/ informative priors

hierarchical_intercept_informative = """
data {
  int<lower=0> nUnique_country; // this is J1
  int<lower=0> nUnique_year; // this is J2
  int<lower=0> N; //observations
  int<lower=1,upper=nUnique_country+1> nCountry[N]; //group sizes
  int<lower=1,upper=nUnique_year+1> nYear[N]; //
  matrix[N,3] x;
  vector[N] y;
} 
parameters {
  vector[nUnique_country] a1;
  vector[nUnique_year] a2;
  vector[2] b;
  real mu_a1;
  real mu_a2;
  real<lower=0> sigma_a1;
  real<lower=0> sigma_a2;
  real<lower=0> sigma_y; // ,upper=100> kısmını sildim
} 
transformed parameters {
  vector[N] y_hat;
  vector[N] m;

  for (i in 1:N) {
    m[i] <- a1[nCountry[i]] + a2[nYear[i]] + x[i,1] * b[1];
    y_hat[i] <- m[i] + x[i,2] * b[2];
  }
}
model {
  mu_a1 ~ normal(0,1); #1 yerine daha büyük bir sayı yazabilirsin uninformative olması için
  mu_a2 ~ normal(0,1);
  sigma_a1 ~ inv_gamma(1,.5);
  sigma_a2 ~ inv_gamma(1,.5);
  a1 ~ normal(mu_a1, sigma_a1);
  a2 ~ normal(mu_a2,sigma_a2);
  b ~ normal(5.7, 1); // changed the beta priors
  y ~ normal(y_hat, sigma_y);
}
"""



hierarchical_intercept_data_informative = {'N': len(dv),
                          'nUnique_country': nUnique_country+1, #len(country), #unique
                          'nUnique_year' : nUnique_year+1,
                          'nCountry': country_labels+1, # Stan counts starting at 1
                          'nYear': year_labels+1, #random intercept
                          'x': np.array(exp_data_mat[['ones_col','iv','control']], dtype='float'),
                          'y': np.array(dv, dtype = 'float')}

hierarchical_intercept_fit_informative = pystan.stan(model_code=hierarchical_intercept, 
                                         data=hierarchical_intercept_data, 
                                         iter=1000, chains=2)

print(hierarchical_intercept_fit_informative) 


# b[2] mean = 0.6, se = 0.01, sd = 0.07

#%%
# answer to the question : the posteriors of mean beta estimate for the informative and uninformative models
# did not change remarkably (b[2] for the uninformative model = 0.57 whereas 0.6 for informative model).
# However, the posterior standard errors are very different in the two models (b[2] se = 8.5e-3 for the
# uninformative model, and b[2] se = 0.01 for the informative model). This discrepancy between the two models
# is due to the fact that the beta prior for the informative model is 10 times larger than the beta posterior
# for the uninformative model.
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:05:21 2024

@author: Vetle
"""

import numpy as np
import pymc as pm
import pandas as pd
import arviz as az
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from scipy.stats import norm, skew




histdata = pd.read_csv('fx_spx_data.csv',sep = ';').values[850:]

sigma_spx = np.std(histdata[:,1])# * np.sqrt(365)
mu_spx = 0.0
sigma_usdnok = np.std(histdata[:,0])# * np.sqrt(365)
mu_usdnok = 0.0



mean = np.array((mu_usdnok, mu_spx))
cov = np.cov(histdata.T)
#cov = np.array(((sigma_spx**2,-0.03),(-0.03,sigma_usdnok**2)))
skewness = 10.85

p_mu_spx=0.0
p_mu_usdnok=0.0
p_s_spx=0.011# * np.sqrt(365)
p_s_usdnok=0.007# * np.sqrt(365)

with pm.Model() as model:
    # Priors for the marginals
    p_usdnok = pm.HalfNormal("p_s_usdnok", sigma = 5.0)
    p_spx = pm.HalfNormal("p_s_spx", sigma = 5.0)
    #exchange_rate = pm.SkewNormal("exchange_rate", 
    #                              mu=p_mu_usdnok - np.sqrt(2/3.14) * p_usdnok * (4.8/(np.sqrt(1 + 4.8**2))) , 
    #                              sigma=p_s_usdnok / (np.sqrt(1 - (2/3.14) * (4.8/(np.sqrt(1 + 4.8**2)) ** 2))) , alpha = 4.8)
    #stock_index = pm.Normal("stock_index", mu=p_mu_spx, sigma=p_s_spx)
    #stock_index = pm.SkewNormal("stock_index", mu=p_mu_spx - np.sqrt(2/3.14) * p_spx * (-0.9/(np.sqrt(1 - 0.9**2))), 
    #                            sigma=p_s_spx / (np.sqrt(1 - (2/3.14) * (-0.9/(np.sqrt(1 - 0.9**2)) ** 2))), alpha = -0.9)
    # Cholesky decomposition to represent correlation
    
    exchange_rate = pm.ExGaussian("exchange_rate", mu = 0.0, sigma = p_usdnok, nu = 2.0)
    stock_index = pm.ExGaussian("stock_index", mu = 0.0, sigma = p_spx, nu = 0.5)
    
    rho = pm.Uniform("rho", lower=-1, upper=1)

    

    chol, cov, _ = pm.LKJCholeskyCov("chol", n=2, eta=0.5, sd_dist=pm.Exponential.dist(2.0))
    joint = pm.MvNormal("joint", mu=[exchange_rate, stock_index], chol=chol, observed=histdata)
    
    trace = pm.sample(10000,cores = 1, chains = 1 ,return_inferencedata=True)
    
    
az.summary(trace)
import seaborn as sns
import scipy.stats.exponnorm

# Extract Cholesky components
chol_samples = trace.posterior["chol"].values  # Shape: (chains, draws, chol_dim_0)

# Compute the covariance matrices from Cholesky decomposition
covariance_matrices = np.array([np.dot(chol.T, chol) for chol in chol_samples.reshape(-1, 2, 2)])

mean_usdnok = np.mean(trace.posterior["exchange_rate"].values)
mean_spx = np.mean(trace.posterior["stock_index"].values)

# Generate joint samples
num_samples = len(covariance_matrices)
joint_samples = np.array([
    np.random.multivariate_normal(
        mean=[mean_usdnok, mean_spx], 
        cov=covariance_matrices[i]
    ) for i in range(num_samples)
])

usdnok_joint_samples = -joint_samples[:, 0]
spx_joint_samples = joint_samples[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(usdnok_joint_samples, spx_joint_samples, alpha=0.5, s=10)
plt.title("Joint Posterior Distribution (with Covariance): SPX vs USDNOK")
plt.xlabel("USDNOK")
plt.ylabel("SPX")
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
sns.kdeplot(x=usdnok_joint_samples, y=spx_joint_samples, cmap="Blues", fill=True)
plt.title("Joint Posterior Contour (with Covariance): SPX vs USDNOK")
plt.xlabel("USDNOK")
plt.ylabel("SPX")
plt.grid()
plt.show()

posterior_correlation = np.corrcoef(usdnok_joint_samples, spx_joint_samples)[0, 1]
print(f"Posterior Correlation (with Covariance): {posterior_correlation:.2f}")

# plot inferences in red
from scipy.stats import skewnorm
from  scipy.stats import exponnorm




# Parameters for the SkewNormal distributions
mu_exchange = p_mu_usdnok  # Exchange rate location
sigma_exchange = p_s_usdnok  # Exchange rate scale
alpha_exchange = 4.8  # Exchange rate skewness

mu_stock = p_mu_spx  # Stock index location
sigma_stock = p_s_spx  # Stock index scale
alpha_stock = 5  # Stock index skewness

# Generate grid for the SkewNormal marginals
x = np.linspace(-3, 3, 500)  # Adjust range as needed
y = np.linspace(-3, 3, 500)

# SkewNormal PDF for marginals
"""
fx_sample = skewnorm.rvs(alpha_exchange, loc=p_mu_usdnok - np.sqrt(2/3.14) * p_s_usdnok * (4.8/(np.sqrt(1 + 4.8**2))) 
                         , scale=sigma_exchange, size = 10000)
spx_sample = skewnorm.rvs(-alpha_exchange, loc=p_mu_spx - np.sqrt(2/3.14) * p_s_spx * (-0.9/(np.sqrt(1 - 0.9**2)))
                          , scale=sigma_stock, size = 10000)
"""
fx_sample = exponnorm.rvs(1/1.4, loc = 0.0, scale = p_s_usdnok, size = 10000)
spx_sample = exponnorm.rvs(1/0.8, loc = 0.0, scale = p_s_spx, size = 10000)

ax = az.plot_pair(
    {"a": histdata[:,0], "b": histdata[:,1]},
    marginals=True,figsize=(16, 12),
    # kind=["kde", "scatter"],
    kind="kde",
    scatter_kwargs={"alpha": 0.1},
    kde_kwargs=dict(
        contour_kwargs=dict(colors="k", linestyles="--"), contourf_kwargs=dict(alpha=0)
    ),
    marginal_kwargs=dict(color="k", plot_kwargs=dict(ls="--")),
)

axz = az.plot_pair(
    {"a": fx_sample, "b": spx_sample},
    marginals=True,
    # kind=["kde", "scatter"],
    kind="kde",figsize=(16, 12),
    scatter_kwargs={"alpha": 0.1},
    kde_kwargs=dict(
        contour_kwargs=dict(colors="b", linestyles="-"), contourf_kwargs=dict(alpha=0)
    ),
    marginal_kwargs=dict(color="b", plot_kwargs=dict(ls="--")),ax=ax


)

axs = az.plot_pair(
    {"exchange_rate": spx_joint_samples, "stock_index": usdnok_joint_samples},
    marginals=True,
    # kind=["kde", "scatter"],
    kind="kde",figsize=(16, 12),
    scatter_kwargs={"alpha": 0.01},
    kde_kwargs=dict(contour_kwargs=dict(colors="r", linestyles="-"), contourf_kwargs=dict(alpha=0)),
    marginal_kwargs=dict(color="r"),ax=ax

    
);


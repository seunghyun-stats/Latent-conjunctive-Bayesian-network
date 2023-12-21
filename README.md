# Latent conjunctive Bayesian network
This repository contains MATLAB codes for the paper "Latent Conjunctive Bayesian Network: Unify Attribute Hierarchy and Bayesian Network for Cognitive Diagnosis"

The DINA (or GDINA) folder contains the sample code for estimating the DINA (or GDINA)-based LCBN. We present the DINA/GDINA-versions of the main algorithm in Section 4, and also the code used to generate Tables 3 and 4.
- "main_DINA/GDINA.m" is the main script used to generate Tables 3 and 4 (and additional results in the Supplementary Material)
- There are two main functions "get_DINA/GDINA_PEM.m" and "get_CBN_DINA/GDINA_EM.m", which correspond to the two steps (Algorithm 1 and 2) in our estimation procedure
- One may change the attribute hierarchy and the proportion/item parameters accordingly by modifying the initialization part in the main script

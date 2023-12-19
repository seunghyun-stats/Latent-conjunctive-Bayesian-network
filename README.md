# Latent conjunctive Bayesian network
This repository contains MATLAB codes for the paper "Latent Conjunctive Bayesian Network: Unify Attribute Hierarchy and Bayesian Network for Cognitive Diagnosis"

The DINA folder contains the sample code for estimating the DINA-based LCBN. We present the DINA-version of the main algorithms in Section 4, and also the code used to generate Table 3.
- "main_DINA.m" is the main script used to generate Table 3 (and additional results in the Supplementary Material)
- There are two main functions "get_DINA_PEM.m" and "get_CBN_DINA_EM.m", which correspond to the two steps (Algorithm 1 and 2) in our estimation procedure
- One may change the attribute hierarchy and the proportion/item parameters accordingly by modifying the initialization part in the main script

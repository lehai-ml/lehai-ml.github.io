---
layout: post
title: "Latent Class Growth Analysis and Growth Mixture Modelling"
date: 2025-03-19
excerpt: ""
permalink: /blogs/latent-class-growth-analysis-and-growth-mixture-modelling/
tags: [statistics, r, clustering]
---

# Introduction

Clustering is the task of grouping set of data into subgroups, such that data points within a subgroup (a cluster) is similar to each other but is different from other subgroups. However, clustering algorithm such as K-means clustering does not necessarily consider the temporal aspect of the data.

Growth mixture modelling (GMM) and Latent Class Growth Analysis (LCGA) are two types of longitudional modelling techniques that identify *homogenous* subpopulations based on growth trajectories. In medical research, the tools can be applied to study the different developmental trajectories within a population.

<br>
<br>

>> *A useful framework for beginning to understand latent class analysis and growth mixture modelling is the distinction between person-centred and variable-centred approaches. Variable-centred approaches such as regression [...] focuses on describing the relationships among variables. The goal is to identify significant predictors of outcomes, and describe how dependent and independent variables are related. Person-centred approaches, on the other hand, include methods such as cluster analysis, latent class analysis, and finite mixture modelling. The focus is on the relationships among indiviudals, and the goal is to classify individuals into distinct groups or categories based on individual response patterns*[[1]](#1).

<br>
<br>

One example that I am very interested in was this paper by Bandoli et al. [[2]](#2), where the authors have examine whether patterns of prenatal alcohol exposure differentially affect dysphormic features in infants.

>![](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/cda1/7722075/69e1f92bc02a/nihms-1629488-f0001.jpg)

Here, using longitudinal modelling, the authors found 5 distinct trajectories of development, which corresponded to high sustaiend, moderate/high, low/moderate sustained, low/moderate and minimal/no prenatal alcohol exposure. Dysmorphology score was then calculated and examined for association with trajectory of prenatal alcohol exposure.

## Implementation

The following two tutorials explain quite clearly how to carry out LCGA and GMM in R [[3,4]](#3). In a nutshell, imagine you have a set of 

<!-- 
Suppose you have selected a 

Unsupervised models:
- K means longitudal
    - create clusters
    - calculate mean trjaectory within each cluster
    - assign each individual to the cluster with the nearest mean trajectory, i.e., distance between individual's mean and trajectory mean minimised
    - repeat steps 1-3 until there are no more changes in the cluster asignment
    - solution do not always converge
- Group-based trajectory models
    - Analyst specifies polynomial shapes of trajectories and number of possible groups
    - GBTM simultaneously estiamtes
        - A multinomial model for group-assignment probabilities
        - models estimating longitudinal trajectories using polynomial functions of time, e.g., quadratic, cubic
        - Individuals are assigned to the trajectory group to which they had the highest membership probability
- Latent class analysis (LCA)
- Latent class growth mixture modeling
    - Similar to GBTM, but allows for variations in individual trajectories within the same group.
- Hiearchical Cluster analysis


Group-based trajectory modeling is a statistical method to idenfity groups of individuals following a similar trajectory over time based on a single outcome [[1]](#1).    -->



# References
<a id="1"></a> [1] [Jung and Wickrama. An Introduction to Latent Class Growth analysis and Growth Mixture Modeling](https://www.statmodel.com/download/JungWickramaLCGALGMM.pdf)

<a id="2"></a> [2] [Bandoli *et al*, 2020. Patterns of Prenatal Alcohol Exposure and Alcohol-Related Dysmorphic Features](https://pubmed.ncbi.nlm.nih.gov/32772389/) 

<a id="3"></a> [3] [https://www.youtube.com/watch?v=cqnpN1k1mPk&ab_channel=RegorzStatistik](
https://www.youtube.com/watch?v=cqnpN1k1mPk&ab_channel=RegorzStatistik)

<a id="4"></a> [4] [https://www.youtube.com/watch?v=sQfIeOh3rJQ&ab_channel=RegorzStatistik](
https://www.youtube.com/watch?v=sQfIeOh3rJQ&ab_channel=RegorzStatistik)

<a id="1"></a> [1] [Nagin *et al*., 2016. Group-based multi-trajectory modeling](https://www.andrew.cmu.edu/user/bjones/refpdf/multtraj.pdf)




# Opt4MLProject

This is the code repository for my semester project at Machine Learning and Optimization lab, EPFL, during my master's study at EPFL. Our work led to a publication at NeurIPS workshop: Optmization in Machine Learning, [\<**Decentralized Stochastic Optimization with Client Sampling**\>](https://opt-ml.org/papers/2022/paper72.pdf), me as the first author. 

In the project, we proposed a novel algorithmic framework to capture the **convergence characteristics** of the realistic scenario of **Decentralized Stochastic Gradient Descent (D-SGD) with partial node participation**, covering a large variety of D-SGD methods developed in different communities, and enabling **training in a distributed fashion**.

Specifically, we derived unified theorem of theoretical convergence rate under the proposed algorithmic framework, quantifying the effect of **node sampling** and choice of the **topology** on the convergence rate. We also empirically **justified the tightness** of these theoretical results. 

The theoretical part was done by me under the supervision of my Ph.D. advisor, Anastasia Koloskova, while the experimental part was done in collaboration with two of my classmates in CS439 - Optimization for machine learning. 

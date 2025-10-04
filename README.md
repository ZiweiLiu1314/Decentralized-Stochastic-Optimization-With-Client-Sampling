# Decentralized Stochastic Optimization with Client Sampling

[![Paper](http://img.shields.io/badge/Paper-NeurIPS%20OPT--ML%202022-blue)](https://opt-ml.org/papers/2022/paper72.pdf)

This repository contains the code for the paper **"Decentralized Stochastic Optimization with Client Sampling"**.

> **Note:** This is a migrated repository. The original was hosted on a now-expired EPFL student account ([original link](https://github.com/Ziwei-Liu3/Opt4MLProject)).


## 📖 Overview

This work proposes a novel algorithmic framework to capture the **convergence characteristics** of **Decentralized Stochastic Gradient Descent (D-SGD) with partial node participation**. Our framework unifies a large variety of D-SGD methods and enables a theoretical analysis that quantifies the effect of **node sampling** and **network topology** on the convergence rate.

Our key contributions are:
1.  A unified theorem for the theoretical convergence rate under the proposed framework.
2.  Empirical validation justifying the tightness of our theoretical results.

This project was conducted as a semester project at the [Machine Learning and Optimization Lab (MLO), EPFL](http://mlo.epfl.ch/), and led to a 1st-author publication at the NeurIPS 2022 Workshop on Optimization for Machine Learning.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   [PyTorch](https://pytorch.org/) (Version 1.x. Check `requirements.txt` for details)
*   [MPI4Py](https://mpi4py.readthedocs.io/) (for distributed communication)

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/ZiweiLiu1314/Decentralized-Stochastic-Optimization-With-Client-Sampling.git
    cd Decentralized-Stochastic-Optimization-With-Client-Sampling
    ```
2.  (Recommended) Create a virtual environment and install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🧪 Usage

The main experiments can be run using the provided scripts.

**Example: Running a decentralized training experiment on CIFAR-10**
```bash
python main.py --dataset cifar10 --model resnet20 --graph ring --participation 0.5
```

---
layout: default     # use your main layout
title: 3. Model formulas         # page title
---

# 3. Model formulas 

### My goal for this section

My goal for this section seems simple enough. For each of the models we've identified as useful for fraud detection, I want to do two seemingly simple things:  1) give the formula for the model's predicted fraud rates as a function of the feature vector and any model parameters, and 2) write down the function that is optimized in fitting the model parameters, given a set of tuned hyperparameters. In particular, I want to write out the optimization function in terms of the model parameters $\mathbf{w}$ and hyperparameters $\mathbf{\lambda}$, rather than just hiding them in the $f_{\mathbf{w}}$

For instance, if we were talking (in a non-fraud scenario) about ordinary least squares linear regression on a single feature with an L2 regularization penalty, this is straightforward. Let's say we use mean-square error for the loss and validation loss here, and we have decided on a search space $\Lambda$ of positive numbers in which to search for the L2 hyperparameter. Then, remembering our training and validation folds $(T_1, V_1),\ldots, (T_k, V_k)$, and our full training set $T$, we have: 

<div style="margin-left: 30px;">
<strong>Formula for predictions</strong>: $y=w_0 + w_1 x$, where $w_0, w_1\in \mathbb{R}$ are the model parameters. 
<br><br>
<strong>Regularization</strong>: A common regularization would be L2 regularization with $\Omega(\lambda):=\lambda w_1^2$. (The intercept term typically isn't regularized.)
<br><br>
<strong>Optimization for the model parameters $w_0, w_1$ given a tuned hyperparameter $\lambda$</strong>: The model parameters are determined by minimizing 

$$\lambda w_1^2 + \sum_{i\in T} (y_i - w_0 - w_1 x_i)^2$$  
</div>

Why do I want to do this? It helps me understand the models better and the process of fitting them.  And it helps me figure out how to modify the models and/or process to accommodate special circumstances in the data or business objectives.  A secondary objective is that because class imbalance is sometimes dealt with in fraud detection with class weighting, I wanted to see where the sample weights appear in these formulations.

Why am I writing down the explicit function being optimized in the final model fitting, but not those for hyperparameter tuning? A few reasons: First, as indicated in the last section, the optimization of the regularized log-loss in tuning hyperparamters is essentially the same problem as final model fitting, just on different sets of data. Second, hyperparameter tuning usually involves a brute-force search of a small search space, so there's no real benefit in seeing the functional form of the validation loss as a function of the model parameters. And third, I'm not completely nuts, although I don't entirely regret the Sunday afternoon I spent writing this out for many of the models. 


## 3.1 Additional setup

### 3.1.1 Random variables
We will use $Y, X_1,..., X_m$ to denote the random variables from which the data $\mathcal{D}$ was generated, so that $P(Y=1 \ \mid \mathbf{x}\in\mathcal{X})$ refers to the distribution of $Y$ conditioned on the features $X_1=x_1,\ldots , X_m=x_m$.

### 3.1.2 Class weights

Fraud data is sometimes adjusted by class weights to address the fact that we're most interested in the fraud signal and fraud is rare.  We'll use $s_1, ..., s_n \geq 0$ to denote the class weights, sometimes referring to them via $\mathbf{s}:=(s_1, ..., s_n) \in \mathbb{R}^n$.  (Fraud data rarely seems to involve probabilistic sampling. But if it does, include the sample weights in the $s_i$ too.)

### 3.1.3 Link function

We'll be transitioning back and forth between probabilities and their log-odds, so let $\sigma:\mathbb{R}\rightarrow (0,1)$ denote the sigmoid link function 
$$\sigma(x)=\frac{1}{1+\exp(-x)}$$ So, $\sigma$ is invertible with inverse 
$$\sigma^{-1} (x):=\ln \frac{x}{1-x}$$

### 3.1.4 Our usual loss function

With the exception of support vector machines (which use hinge loss), I'll use the log-loss (aka cross-entropy, aka negative log-likelihood) for both the training loss and the validation loss.  This seems to be a common choice for fraud detection.

To review, the log-loss of a single predicted probability $z$ of a label $y$ is 
$$L(y,z) = -y\ln z - (1-y) \ln(1-z)$$ 

Now, suppose we are considering models $f_{\mathbf{w}}:\mathcal{X}\rightarrow [0,1]$ from a given family (e.g. neural networks). The log-loss of the model using parameters $\mathbf{w}$ on a subset 
$S\subseteq\{1,\ldots,n\}$ is the class-weighted average of the log-losses of the predictions on the subset of $\mathcal{D}$ indexed by $S$:

$$LogLoss(\mathbf{w}, S):=\frac{1}{\sum_{i\in S} s_i} \sum_{i\in S} s_i L(y_i, f_{\mathbf{w}}(\mathbf{x}_i)) = \frac{-1}{\sum_{i\in S} s_i} \sum_{i\in S} s_i 
\left(y_i \ln f_{\mathbf{w}}(\mathbf{x}_i) + (1 - y_i) \ln(1 - f_{\mathbf{w}}(\mathbf{x}_i))\right)$$

The log-loss is familiar to statisticians as the negative log-likelihood applied to a binomial distribution.  If the class weights were all 1, then $-LogLoss(\mathbf{w}, \{1,\ldots,n\})$ would be the chance of seeing the data 
$y_1,..., y_n$ if we flipped $n$ coins (independently) with $P(\text{heads})$ for the 
$i^{\text{th}}$ coin being $f_{\mathbf{w}}(\mathbf{x}_i)$, and wrote down a list of the results, with 1 for every heads and 0 for every tails.

### 3.1.5 Optimization with log-loss

With log-loss as our training loss function, the regularized loss on our training data $T$ becomes:
$$ \Omega (\mathbf{w},\mathbf{\lambda}^*) - \frac{1}{\sum_{i\in T} s_i} \sum_{i\in T}  s_i \left( y_i \ln f_{\mathbf{w}}(\mathbf{x}_i) + (1 - y_i) \ln(1 - f_{\mathbf{w}}(\mathbf{x}_i)) \right)$$

This is the function we minimize to fit the final model parameters $\mathbf{w}$ from tuned hyperparameters $\mathbf{\lambda}$.

So, now, let's hit the models.

## 3.2 Logistic regression

$\textbf{Model form}$: 
$$P(Y=1 \ | \ \mathbf{x}\in\mathcal{X}) = \sigma(\mathbf{w}^t \mathbf{x} + w_0) = \frac{1}{1 + \exp(-\sum_{i=1}^m w_i x_i  - w_0)}$$
where $w_0, w_1,..., w_m \in \mathbb{R}$ are the model parameters. 

$\textbf{Regularization}$: We'll use an L2 penalty, so $\Omega (\mathbf{w},\lambda) := \lambda \sum_{i=1}^m w_i^2$. (Again, the bias term $w_0$ typically isn't regularized.)

$\textbf{Optimization for the model parameters} \ \mathbf{w}$: Note that for each $1\leq i\leq n$, we have the following simplification in our log-loss function:
$$\begin{align*}- s_i \left( y_i \ln f_{\mathbf{w}}(\mathbf{x}_i) + (1 - y_i) \ln (1 - f_{\mathbf{w}}(\mathbf{x}_i))\right)
& = - s_i \left( y_i \ln \frac{1}{1 + \exp(-\sum_{i=1}^m w_i x_i  - w_0)} + (1 - y_i) \ln\left(1 - \frac{1}{1 + \exp(-\sum_{i=1}^m w_i x_i  - w_0)}\right)\right) \\
& = s_i \left( y_i \ln \left(1 + \exp\left(-\sum_{i=1}^m w_i x_i  - w_0\right)\right) + (1 - y_i) \ln\left(1 + \exp\left(\sum_{i=1}^m w_i x_i  + w_0\right)\right)\right)\end{align*}$$

So using a tuned hyperparameter $\lambda$, the model parameters $\mathbf{w}$ are determined by minimizing:

$$\lambda \sum_{i=1}^m w_i^2 + \frac{1}{\sum_{i\in T} s_i} s_i \left( y_i \ln \left(1 + \exp\left(-\sum_{i=1}^m w_i x_i  - w_0\right)\right) + (1 - y_i) \ln\left(1 + \exp\left(\sum_{i=1}^m w_i x_i  + w_0\right)\right)\right)$$

As noted earlier, this minimization can be approximated using, e.g., gradient descent.    

### Some notes on logistic regression

- Logistic models are easy to interpret.

- They are susceptible to redundant features, such as multicollinear ones.

- They are susceptible to features with wildly different means and variances, so it's important to standardize numeric features before fitting a logistic model (or tuning its hyperparameter).


## 3.3 Decision trees

Trees are formed by repeatedly partitioning the feature space into half spaces 

$$\{ \mathbf{x}\in\mathbb{R}^m: x_i \leq c_i\} \text{ and } \{ \mathbf{x}\in\mathbb{R}^m : x_i > c_i \}$$ 

where $1\leq i\leq m$ and $c_i\in\mathbb{R}$. This gives rise to leaves of the form 
$$\prod_{j\in J} \{a_j < x_j \leq b_j\}$$ where 
$J\subseteq \{1,...,m\}$ and $a_j, b_j \in [ -\infty, +\infty ] \ \forall j\in J$. (The interval $(a_j, b_j]$ can be bounded at both ends if the $j$th feature is visited multiple times in the tree.) Geometrically, the leaves are rectanguloids. 

Decision trees give constant predictions on the leaves (so these models are locally constant). It is a simple exercise to see that the constant prediction that minimizes the log-loss is the the class-weighted average incidence of fraud in the data.  Similarly, the prediction on each leaf is the class-weighted fraud incidence.  These observations gives us the model form.

$\textbf{Model form}$: A decision tree with $V$ leaves predicts as follows: $$P(Y=1 \mid \mathbf{x}\in\mathcal{X}) = \sum_{v=1}^V r_t \ \mathbb{I}(\mathbf{x}\in L_v)$$ 

where $\mathbb{I}(.)$ is the boolean indicator function (taking the value 1 if its argument is true and 0 otherwise), the leaves 
$L_1,... L_V$ are rectanguloids partitioning the feature space, and the "leaf weights" $r_1,\ldots, r_V$ are the class-weighted fraud incidences on the leaves: 

$$r_v:= \frac{\sum_{i\in L_v} s_i y_i}{\sum_{i\in L_v} s_i}, \forall 1\leq v\leq V$$  

Writing each leaf $L_v$ as 

$$L_v = \prod_{j\in J_v} (a_{vj}, b_{vj}]$$ 

where 
$J_v\subseteq \{1,\ldots, m \}$ and 
$a_{vj}, b_{vj} \in [ -\infty, +\infty ] \ \forall j\in J_v$, we have 

$$P(Y=1 \mid \mathbf{x}\in\mathcal{X}) = \sum_{v=1}^V r_v \ \mathbb{I}(\mathbf{x}\in \prod_{j\in J_v} (a_{vj}, b_{vj}]) 
= \sum_{v=1}^V r_v \  \prod_{j\in J_v} \mathbb{I}(a_{vj}<x_j\leq b_{vj})$$

That is, a decision tree simply partitions the feature space into rectanguloids and predicts the chance of fraud in each rectanguloid to be the fraud incidence for the portion of the data $\mathcal{D}$ that falls into the rectanguloid.

$\textbf{Model parameters}$: So what are the model parameters for a decision tree? Well, to make a decision tree (as I have formulated them), you need to specify: 
- the number $V$ of leaves, 
- $J_v\subseteq \{1,\ldots, m \}$ for all $1\leq v\leq V$, and 
- $a_{vj}, b_{vj} \in [ -\infty, +\infty ] \ \forall j\in J_v$

and the $J_v, a_{vj}, b_{vj}$ are constrained in that the resulting $L_v = \prod_{j\in J_v} (a_{vj}, b_{vj}]$ for $1\leq v\leq V$ collectively have to partition the feature space.  

The number of leaves $V$ is usually treated as a hyperparameter. So the model parameters would be the 
$J_v\subseteq \{1,\ldots,m\}$ and the 
$a_{vj}, b_{vj} \in [ -\infty, +\infty ]$. 

$\textbf{Regularization}$: I'll just regularize the number $V$ of leaves, so $\Omega (\mathbf{w},\lambda) := \lambda V$. But you can also regularize other aspects, like the maximum depth of the tree and the number of samples per leaf.

$\textbf{Optimization for the model parameters}$: The regularized log-loss on the training data is:

$$\lambda V - \frac{1}{\sum_{i\in T} s_i} \sum_{i\in T} s_i \left(y_i \ln \left(\sum_{v=1}^V r_v \  \prod_{j\in J_v} \mathbb{I}(a_{vj}<x_{ij}\leq b_{vj})\right) + (1 - y_i) \ln\left(1 - \sum_{v=1}^V r_v \  \prod_{j\in J_v} \mathbb{I}(a_{vj}<x_{ij}\leq b_{vj})\right)\right)$$

This is what I think would be minimized to fit the model parameters. But this minimization isn't easy. Even setting aside the non-numeric parameters $J_v$, the log-loss doesn't look continuous in, e.g., $a_{11}$ (assuming for simplicity that $1\in J_1$).  
But this doesn't mean that the optimal model parameters and hyperparameters don't exist. It just means we can't use techniques like gradient descent to find them.  

Various sources, such as Wikipedia, say that under very general conditions on the loss function, finding the optimal decision tree is an NP-hard problem. [^2]  I imagine that log-loss would meet these general conditions. That said, I wasn't able to find a source saying, e.g., that the objective in fitting a decision tree (or random forest or gradient boosted trees) from already-tuned hyperparameters is to minimize the regularized loss.  Instead, the sources I consulted present the "greedy" algorithm that optimizes each step in a tree-building and tree-pruning process, and (sometimes) acknowledge that the result is "suboptimal" without seeming to specify the optimization that would make a tree "optimal".  [^3]  [^4]  [^5]  [^6]  [^7] 
So, I can't say definitively that the objective is to minimize the regularized training loss, and the greedy algorithms provide a good-enough solution to that objective. At the same time, I don't see why tree-based models would be an exception to what otherwise seems to be a general principle. 

$\mathbf{Notes}$:

- Decision trees aren't susceptible to features with wildly different means and variances, so one needn't standardize numeric features before fitting them.

- The effect of redundant features on decision trees is a little nuanced. Trees aren't susceptible to redundant features in the sense that if two features are highly correlated, a decision tree might use one feature or the other as it makes its splits. The tree will still be built and approximately minimize the regularized log-loss. What will be lost though, if you're not aware of which features are mutually redundant, is the magnitude of feature importance (which might be divided among the redundant features).

- Decision trees are easily interpreted

- But if they're not sufficiently pruned, then they tend to overfit the data (low bias, high variance). 


## 3.4 Random forests

Random forests address the high variance of decision trees. It does this by averaging the results of several trees, each built on a randomized pertubation of the training data $T$ and a random subset of the features.  

The randomized pertubation is accomplished by bootstrapping the data.  That is, by replacing $T$ with a simple random sample with replacement of the same size $\vert T\vert$.

$\textbf{Model form}$: 
$$P(Y=1 \mid \mathbf{x}\in\mathcal{X}) = \frac{1}{K} \sum_{k=1}^K P_k (Y=1 \mid \mathbf{x})$$ 

where $P_k (Y=1 \mid \mathbf{x})$, for each $1\leq k\leq K$, is the prediction from a decision tree 
trained on a bootstrap sample of size $\vert T\vert$ from the training data $T$ and from a simple random sample of 
$F$ features, for some $F\geq 1$. (The same value of $F$ is used for each tree.)

So the random forest predicts the chance of fraud given $\mathbf{x}\in\mathcal{X}$ to be the average of the class-weighted fraud incidence in the 
$K$ leaves to which $\mathbf{x}$ belongs.

$\textbf{Optimization (my take)}$: The same sources cited for decision trees seem to suggest, but not state explicitly, that the model parameters for each tree in the forest should be designed to minimize the regularized loss function.  If we regularize the numbers of leaves in the trees, the regularized log-loss would be the average of the regularized log-losses for the individual trees.

The general technique of taking a class of models (like decision trees), and forming several models from the class by bootstrapping the data is $\textit{bagging}$. So a random forest is a collection of bagged decision trees. Bagging might or might not involve using random subsets of features for the models (but for random forests, it does).

Random forests typically have lots of trees.  The default in scikit-learn is 100 trees.  As each tree is based on different data and different features, the trees can have different depths and numbers of leaves.

The number of features subsetted for each tree is typically $\sqrt{m}$ (rounded down) for classification trees and 
$m/3$ (rounded down) for regression, provided there are at least 5 samples per node. [^8]

$\textbf{Notes}$:

- The random perturbations to the training data and features considered make random forests outperform decision trees. In fact, random forests are described in the Handbook as having state-of-the-art performance for fraud detection. 

- Like decision trees, random forests aren't susceptible to features with wildly different means and variances, so one needn't standardize numeric features before fitting them.

- Random forests are less susceptible to feature redundancy than individual trees are. [^9] 

- Random forests are harder to interpret than decision trees.  For any given feature vector, you can certainly identify the leaves it falls in, and their associated class-weighted fraud incidence as indicated by the data $\mathcal{D}$.  But with often 100 trees, this information is probably too complex to be helpful.

## 3.5 Gradient-boosted trees

Having looked at one type of ensemble (bagging), we now look at another (boosting).  While the decision trees in a random forest can be built in parallel, those in gradient boosting are built sequentially. And rather than simply averaging the tree predictions like in random forests, gradient boosted trees predict the log-odds as a linear combination of the log-odds predicted from the trees. I will only look at XGBoost, which seems popular.

$\textbf{Model form}$: 
$$P(Y=1 \mid \mathbf{x}\in\mathcal{X}) = \sigma(b+\eta \sum_{k=1}^K f_k(\mathbf{x})) = \frac{1}{1+ \exp(-b-\eta \sum_{k=1}^K f_k(\mathbf{x}))}$$ 

where $b$ is the log-odds of the fraud rate in the training data $T$, 
$0<\eta<1$ is a hyperparameter (the "learning rate"), $K\geq 1$, and $f_1(\mathbf{x}),..., f_K(\mathbf{x})$ are the predictions from the decision trees determined by the boosting algorithm.  (Although scikit-learn accepts learning rates larger than 1, it seems to make most sense to limit to smaller learning rates.) 

$\textbf{Regularization}$: 
XGBoost regularizes the numbers $V_1,\ldots, V_K$ of leaves in each tree and puts L1 and L2 penalties on the leaf predictions.  If we denote the vector of leaf predictions from the $k$th tree by 
$\mathbf{r}_k$, then the regularization is 
$\sum_{k=1}^K \left( \gamma V_k + \lambda ||r_k||^2 + \alpha |r_k| \right)$ 

$\textbf{Optimization (my take)}$: For given values of the hyperparameters $\eta, \gamma, \lambda, \alpha>0$, and taking 
$$b:= \frac{\sum_{i=1}^n s_i y_i }{\sum_{i=1}^n s_i(1-y_i)},$$ 
the log-odds trees $f_1,..., f_K$ should again attempt to minimize the regularized log-loss of the predictions, which is:
$$\sum_{k=1}^K \left( \gamma T_k + \lambda ||r_k||^2 + \alpha |r_k| \right) + \frac{1}{\sum_{i\in T} s_i} s_i \left( y_i \ln \left(1 + \exp\left(-b-\eta \sum_{k=1}^K f_k(\mathbf{x})\right)\right) + (1 - y_i) \ln\left(1 + \exp\left(b+\eta \sum_{k=1}^K f_k(\mathbf{x})\right)\right)\right) $$

As with random forests, the resulting trees can have different depths and numbers of leaves.

## 3.6 Support vector machines

Support Vector Machine Classifiers are an outlier in two respects. First, they don't use the same loss function: Instead of log-loss, they use hinge loss.
And second they predict a classification of fraud-vs-legit (separating the two via a hyperplane in a high-dimensional space defined by the kernel), not a formula for $P(Y=1 \mid \mathbf{x}\in\mathcal{X})$. You can generate probabilities via "Platt scaling", which fits a sigmoid function to the predicted class using validation. We'll give both versions of the model form:

$\textbf{Model form for class prediction}$: 
$$Y = 0.5 + 0.5 \ sgn \left( \sum_{i=1}^n w_i (2y_i - 1) K(\mathbf{x}_i, \mathbf{x})+b \right)$$

where 
$K:\mathbb{R}^m \times \mathbb{R}^m\rightarrow \mathbb{R}$ is a user-specified kernel.

$\textbf{Model form for probabilities}$: 
$$P(Y=1 \mid \mathbf{x}\in\mathcal{X}) = \sigma \left( A \left( \sum_{i=1}^n w_i (2y_i - 1) K(\mathbf{x}_i, \mathbf{x})+b \right) + B \right)$$

where $K:\mathbb{R}^m \times \mathbb{R}^m\rightarrow \mathbb{R}$ is a user-specified kernel.

The model parameters are $w_1,\ldots, w_n\geq 0$, $b\in\mathbb{R}$, plus any parameters involved in the kernel 
$K$. The parameters $A$ and $B$ are considered to be hyperparameters as they are usually fit using validation. The $2y_i - 1$ term simply translates our {0,1} encoding of legit-vs-fraud to a {-1,1} encoding, and the 0.5 terms in the class predictions translate them back.  The sgn() function returns the sign of its argument (+1, -1, or 0). The support vectors are those $\mathbf{X}_i$ for which $w_i>0$.

We will focus on the "linear kernel", which is simply the dot product, becuase this turns out to be optimal for the Handbook data. In this case, $\sum_{i=1}^n w_i (2y_i - 1) K(\mathbf{X}_i, \mathbf{x})+b$ is just a linear combination of the entries of $\mathbf{x}$.

$\textbf{Optimization}$: Given $\lambda > 0$ and a kernel $K:\mathbb{R}^m \times \mathbb{R}^m\rightarrow \mathbb{R}$, the model parameters $\mathbf{w}, b, \mathbf{\xi}$ are to satisfy: 

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \;
\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n w_i w_j \,\tilde{y}_i \tilde{y}_j \, K(\mathbf{x}_i, \mathbf{x}_j)
\;+\; \frac{1}{\lambda} \sum_{i=1}^n s_i \,\xi_i
$$

$$
\text{subject to} \quad
\tilde{y}_i \left( \sum_{j=1}^n w_j \tilde{y}_j K(\mathbf{x}_j, \mathbf{x}_i) + b \right) \geq 1 - \xi_i,
\quad \xi_i \geq 0, \quad \forall i \in \{1, \ldots, n\}
$$

where $\tilde{y}_i = 2y_i - 1.$ [^10] [^11] 

## 3.7 K-nearest neighbors

$\textbf{Model form}$:    
$$P(Y=1 \mid \mathbf{x}\in\mathcal{X}) = \frac{\sum_{i\in N_k(\mathbf{x})} s_i y_i}{\sum_{i\in N_k(\mathbf{x})} s_i}$$ 

where $k\geq 1$ is user-specified (or tuned), $N_k(\mathbf{x})$ is the set of indices of the 
$k$ samples in the training data $T$ with the 
$k$ smallest values of $||\mathbf{x}_i - \mathbf{x}||$. 

$\textbf{Optimization}$: The parameter $k$ is normally considered a hyperparameter (i.e. not learned from the data). So, the model has no model parameters, and no optimization.  

Also, I'm using Euclidean distance, although other choices are possible.

## 3.8 Neural networks

For now at least, I'm just going to consider feedforward nerual networks (aka multilayer perceptrons) with ReLU activations on all hidden layers and a sigmoid output layer.  This seems to be a standard deep neural network architecture for fraud detection

$\textbf{Model form}$: 
$$P(Y=1 \mid \mathbf{x}\in\mathcal{X}) = \sigma(W_L a_{L-1} + b_L)$$ 

where $L\geq 1$ is user-specified, $a_0:=\mathbf{x}$ and for each 
$1\leq k\leq L-1$, $a_k:=ReLU(W_k a_{k-1} + b_k)$. So the model parameters are the 
$m\times m$ matrices $W_k$ and the vectors $b_k\in\mathbb{R}^m$.

$\textbf{Optimization}$: Put all the parameters from the $W_k$ and $b_k$ into one long parameter vector $\mathbf{w}\in\mathbb{R}^{Lm(m+1)}$. For a given value of the L2 hyperparameter $\lambda>0$, the model parameters $w$ are determined by minimizing the regularized log-loss: 
$$\lambda \|w\|^2 - \frac{1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i L(y_i, P(Y=1 \mid \mathbf{x}_i))$$

noting that each $P(Y=1 \mid \mathbf{x}_i)$ is a function of the the parameters $\mathbf{w}$. 

## Up next

Let's take a break from math and look at the data.

[^1]: Franceschi, L., Frasconi, P., Salzo, S., Grazzi, R., & Pontil, M. (2018). Bilevel programming for hyperparameter optimization and meta-learning. Proceedings of the 35th International Conference on Machine Learning (ICML), 80, 1568–1577. https://proceedings.mlr.press/v80/franceschi18a.html

[^2]: https://en.wikipedia.org/wiki/Decision_tree_learning#cite_note-35

[^3]: Nielsen, D. (2016). Tree boosting with XGBoost: Why does XGBoost win "every" machine learning competition? [Master’s thesis, Norwegian University of Science and Technology]. NTNU Open. https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf?sequence=1&isAllowed=y. 


[^4]: Konstantinov, A. V., & Utkin, L. V. (2025). A novel gradient-based method for decision trees optimizing arbitrary differentiable loss functions. arXiv preprint arXiv:2503.17855. https://arxiv.org/abs/2503.17855

[^5]: Kohler, H., Akrour, R., & Preux, P. (2025). Breiman meets Bellman: Non-Greedy Decision Trees with MDPs. arXiv. https://arxiv.org/html/2309.12701v5#S3

[^6]: Bongiorno, D., D’Onofrio, A., & Triki, C. (2024). Loss-optimal classification trees: A generalized framework and the logistic case. TOP, 32(2), 409–446. https://doi.org/10.1007/s11750-024-00674-y

[^7]: van der Linden, J. G. M., Vos, D., de Weerdt, M., Verwer, S., & Demirović, E. (2025). Optimal or greedy decision trees? Revisiting their objectives, tuning, and performance. arXiv. https://arxiv.org/abs/2409.12788

[^8]: See for instance: GeeksforGeeks. (2025, July 23). Solving the multicollinearity problem with decision tree. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/solving-the-multicollinearity-problem-with-decision-tree/

[^9]: Hastie, Trevor; Tibshirani, Robert; Friedman, Jerome (2008). The Elements of Statistical Learning (2nd ed.). Springer. ISBN 0-387-95284-5.

[^10]: Ding, Y., & Huang, S. (2024). A generalized framework with adaptive weighted soft-margin for imbalanced SVM classification. arXiv. https://arxiv.org/abs/2403.08378

[^11]: Hastie, T., Rosset, S., Tibshirani, R., & Zhu, J. (2004). The entire regularization path for the support vector machine. Journal of Machine Learning Research, 5, 1391–1415. http://www.jmlr.org/papers/volume5/hastie04a/hastie04a.pdf



<table width="100%">
  <tr>
    <td align="left">
      <a href="2-whats-the-same-and-not.html">← Previous: 2. What's the same and what's different</a>
    </td>
    <td align="right">
      <a href="4-the-data-we-use.html">Next: 4. The data we use →</a>
    </td>
  </tr>
</table>
---
layout: default     # use your main layout
title: 2. Model formulas         # page title
---

# 2. Model formulas 

### My goal for this section

My goal for this section seems simple enough. For each of the models we've identified as useful for fraud detection, I want to do two seemingly simple things:  1) give the formula for the model's predicted fraud rates as a function of the feature vector and any necessary parameters, and 2) give the function that is optimized in fitting the model parameters for a fixed set of hyperparameters.  

For instance, if we were talking (in a non-fraud scenario) about ordinary least squares linear regression on a single feature with an L2 regularization penalty, this is easy: 

<div style="margin-left: 30px;">
$\textbf{Formula for predictions}$: $y=w_0 + w_1 x$ where $w_0, w_1\in \mathbb{R}$ are the model parameters. 
</div>

<div style="margin-left: 30px;">
$\textbf{Optimization}$: Fixing $\lambda>0$, the model parameters are determined by minimizing $\sum_{i=1}^n (y_i - w_0 - w_1 x_i)^2 + \lambda (w_0^2 + w_1^2)$ where $\{(x_1,y_1),..., (x_n, y_n)\}\subseteq \mathbb{R}^2$ is the data to which we are fitting the model. 
</div>


Why do I want to do this? It helps me understand the models better.  And it helps me figure out how to modify them if they need to accommodate some special circumstance.  A secondary objective is that because class imbalance is sometimes dealt with in fraud detection with class weighting, I wanted to see where the class weights appeared in these formulations.
    
So how hard can this be...?

## 2.1 Setup

We use the following notation for the our labeled input data $\mathcal{D}$:

$$\mathcal{D}:=\{(\mathbf{X}_1, y_1),\ldots, (\mathbf{X}_n, y_n)\}\subseteq \mathcal{X} \times \{ 0,1 \}$$ 

where each $1\leq i\leq n$ represents a transaction, $\mathcal{X}\subseteq \mathbb{R}^m$ denotes the feature space, and the target class $y=1$ denotes fraudulent transactions.  We'll assume that any feature engineering has already taken place. (So the $m$ features include all engineered features.)  We also assume that any categorical features have already been numerically encoded in some fashion. We will use boldface type to indicate vectors, e.g. $\mathbf{y}:=(y_1,\ldots, y_n)$.

I assume the reader is familiar with the concepts of training, validation, cross-validation, test data, and tuning hyperparameters. I'll be pretty loose about referring to the entire dataset versus the training data, assuming the reader can infer the choice from context (e.g. $\mathcal{D}$ should refer to the training data when training a model, versus the entire dataset when fitting a final model using tuned hyperparameters). 

{% raw %}

As usual, $\mathbf{X}$ will denote the 
$m \times n$ matrix whose columns are 
$\mathbf{X}_1, \ldots, \mathbf{X}_n$. The $(i,j)$th entry 
$X_{ij}$ 
of 
$\mathbf{X}$ 
is the value of the $j$th feature in the $i$th sample. We will sometimes use 
$y$ 
to denote the random variable from which the class data was generated, e.g. 
$P(y=1 \mid \mathbf{x}\in\mathcal{X})$. 

{% endraw %}

### Models

Of course, we are interested in modeling $P(y=1 \mid \mathbf{x}\in\mathcal{X})$, the likelihood of fraud conditioned on the values of the features. Any given model of this probability specified parameters $\mathbf{w}\in\mathbb{R}^W$ for some $W\geq 1$ will be a function 
$$f_{\mathbf{w}}:\mathcal{X}\rightarrow [0,1]$$ 
with $f_{\mathbf{w}}(\mathbf{x})$ denoting the model's predicted probabilty of fraud at feature vector $\mathbf{x}$.  

Sometimes the model parameters $\mathbf{w}$ will simply consistent of coefficients for the 
$m$ features plus a bias term, as with logisitic regression.  Sometimes, the model parameters will look different, as with decision trees. 

Note that for fixed $\mathbf{w}\in\mathbb{R}^W$, $f_{\mathbf{w}}(\mathbf{x})$ can be viewed as a function of $\mathbf{x}\in\mathcal{X}$. And for fixed 
$\mathbf{x}\in\mathcal{X}$, 
$f_{\mathbf{w}}(\mathbf{x})$ can be viewed as a function of $\mathbf{w}\in\mathbb{R}^W$. We will need each viewpoint at various times. 

### Class weights

Fraud data is often be adjusted by class weights to address the fact that we're most interested in the fraud signal and fraud is rare.  We'll use $s_1, ..., s_n \geq 0$ to denote the class weights, sometimes referring to them via $\mathbf{s}:=(s_1, ..., s_n) \in \mathbb{R}_n$.  (Fraud data rarely seems to involve probalistic sampling. But if it does, include the sample weights in the $s_i$ too.)

### Link function

We'll be transitioning back and forth between probabilities and their log-odds, so let $\sigma:\mathbb{R}\rightarrow (0,1)$ denote the sigmoid link function 
$$\sigma(x)=\frac{1}{1+\exp(-x)}$$ So, $\sigma$ is invertible with inverse 
$$\sigma^{-1} (x):=\ln \frac{x}{1-x}$$

### Loss function

With the exception of support vector machines (which use hinge loss), I'll just use the log-loss (aka cross-entropy, aka negative log-likelihood) function, which seems to be the most often used for fraud. If we had a single sample $(X, y)$ in our data and a single prediction 
$z$ for $P(y=1 \mid X)$, the log-loss would be 

$$L(y,z) = -y\ln z - (1-y) \ln(1-z)$$ 

Given a model $f_{\mathbf{w}}:\mathcal{X}\rightarrow [0,1]$ for fraud, the log-loss of the model is the class-weighted average of the log-losses of its predictions on $\mathcal{D}$:

$$\text{LogLoss}(f_{\mathbf{w}}):=\frac{1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i L(y_i, f_{\mathbf{w}}(\mathbf{X}_i)) = \frac{-1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i 
\left(y_i \ln f_{\mathbf{w}}(\mathbf{X}_i) + (1 - y_i) \ln(1 - f_{\mathbf{w}}(\mathbf{X}_i))\right)$$

To statisticians, this is familiar as the negative log-likelihood applied to a binomial distribution.  If the class weights were all 1, then $-\text{LogLoss}(f_{\mathbf{w}})$ would be the chance of seeing the data 
$y_1,..., y_n$ if we flipped $n$ coins (independently) with $P(\text{heads})$ for the 
$i^{\text{th}}$ coin being $f_{\mathbf{w}}(\mathbf{X}_i)$, and wrote down a list of the results, with 1 for every heads and 0 for every tails.

### Regularization

Regularization, along with cross-validation, is one of the ways to keep models from being too complex (i.e. controlling the model variance). When applicable, we will pretty much stick to L2-regularization, which seems to be commonly used.  In this case, the regularization function for a model with model parameters $\mathbf{w}$ is $$\Omega(\mathbf{w}), \lambda := \lambda \|w\|^2$$
where $\lambda>0$ is a hyperparameter (the "regularization strength"). (In scikit-learn, you specify the "inverse regularization strength $C$, defined as $C:=1/(2\lambda)$.) For some models, like decision trees where we want to control things like tree depth and the number of leaves, regularization will look a little different.

Whatever form the regularization $\Omega (\mathbf{w}, \mathbf{\lambda})$ takes (allowing for multiple hyperparameters $\mathbf{\lambda}$), The regularized log-loss is 

$$\text{RegLogLoss}(f_{\mathbf{w}}, \mathbf{\lambda}) :=\text{LogLoss}(f_{\mathbf{w}}) + \Omega (\mathbf{w},\mathbf{\lambda})$$

For given values of the hyperparameters $\mathbf{\lambda}$, I think this is what models in general aim to minimize. When $\Omega()$ is convex and twice-differentiable in $\textbf{w}$, you can minimize it, for fixed hyperparameters, by gradient descent or Newton's method.  When it's not, as will happen for tree-based models, I think what happens is that the optimization becomes NP-hard, essentially requiring an exhaustive search of a large parameter space. I think that's why in tree-based methods, "greedy" algorithms are employed that solve an analogous optimization in steps.

At this point, you might wonder why we don't simultaneously optimize the model parameters and hyperparameter(s). Simulataneous optimization would essentially undermine the role of regularization. E.g. for L2-regularized logistic regression, fitting both the model parameters $\mathbf{w}$ and $\lambda$ to any given dataset $-$ whether the full data or the training data $-$ could very well give an overfit $\mathbf{w}$ with $\lambda\approx 0$. No regularization there...  That's why the model parameters and hyperparameters are tuned separately, often in a tri-level optimization framework using different subsets of the full dataset, like this:

1. Optimize $\mathbf{w}$ on the training data, fixing the $\mathbf{\lambda}$: Set the values of $\mathbf{\lambda}$ to some default or user-informed initial values  
$\mathbf{\lambda}^*$
, and minimize $\text{RegLogLoss}(f_{\mathbf{w}}, \mathbf{\lambda}^*)$ on the training data. Say the min value occurs at $\mathbf{w}^*$.

2. Optimize $\mathbf{\lambda}$ on the validation data (or cross-validation), fixing the $\mathbf{w}$: Minimize $\text{RegLogLoss}(f_{\mathbf{w}^*}, \mathbf{\lambda})$ on this data. Say the min value occurs at $\mathbf{\lambda}^*$.

3. Re-optimize $\mathbf{w}$ on the training (or training + validation) data, fixing the $\mathbf{\lambda}$: Get the final model parameters by minimizing $\text{RegLogLoss}(f_{\mathbf{w}}, \mathbf{\lambda}^*)$ on this data.

## 2.2 Logistic regression

$\textbf{Model form}$: $$P(y=1 \ | \ \mathbf{x}\in\mathcal{X}) = \sigma(\mathbf{w}^t \mathbf{x} + b) = \frac{1}{1 + \exp(-\sum_{i=1}^n w_i x_i  - b)}$$
 where $b, w_1,..., w_m \in \mathbb{R}$ are the model parameters. 

$\textbf{Optimization}$: The model parameters $b,\mathbf{w}$ are determined by minimizing the regularized loss function for a given value of the hyperparameter $\lambda$:

$$\text{RegLogLoss}(f_{\mathbf{w}}, \mathbf{\lambda}) = \frac{1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i L(y_i, \sigma(\mathbf{w}^t \mathbf{X_i} + b))+ \lambda (\|w\|^2 + |b|^2)$$

Aside from hyperparameter tuning, the problem is now straightforward, as indicated earlier: For any given $\lambda$, you can minimize the regularized log-loss by standard numerical methods like gradient descent or Newton's method.   

Notes:

•	Logistic models are easy to interpret.

•	Susceptible to redundant features, e.g. multicollinearity

•	Susceptible to features with wildly different means & variances, so standardize numeric features before applying


## 2.3 Decision trees

Trees are formed by repeatedly partitioning the feature space into half spaces $\{ \mathbf{x}\in\mathbb{R}^m: x_i \leq c_i\}$ and $\{ \mathbf{x}\in\mathbb{R}^m : x_i > c_i \}$ where $1\leq i\leq m$ and $c_i\in\mathbb{R}$. This gives rise to leaves of the form 
$\prod_{j\in J} \{a_j < x_j \leq b_j\}$ where 
$J\subseteq \{1,...,m\}$ and $a_j, b_j \in [ -\infty, +\infty ] \ \forall j\in J$. 
(The interval $(a_j, b_j]$ can be bounded at both ends if the $j$th feature is visited multiple times in the tree.) Geometrically, the leaves are rectanguloids. 

Decision trees give constant predictions on the leaves (so these models are locally constant). It is a simple exercise to see that the constant prediction that minimizes the log-loss is the the class-weighted average incidence of fraud in the data.  Similarly, the prediction on each leaf is the class-weighted fraud incidence.  These observations gives us the model form.

$\textbf{Model form}$: $$P(y=1 \mid \mathbf{x}\in\mathcal{X}) = \sum_{t=1}^T r_t \ \mathbb{I}(\mathbf{x}\in L_t)$$ where $\mathbb{I}(.)$ is the boolean indicator function (taking the value 1 if its argument is true and 0 otherwise), the leaves $L_1,... L_T$ are rectanguloids partitioning the feature space, and the "leaf weights" $r_1,\ldots, r_T$ are the class-weighted fraud incidences on the leaves: 
$$r_t:= \frac{\sum_{i\in L_t} s_i y_i}{\sum_{i\in L_t} s_i}, \forall 1\leq t\leq T$$  Writing each leaf $L_t$ as 
$$L_t = \prod_{j\in J_t} (a_{tj}, b_{tj}]$$ where $J_t\subseteq \{1,...,m\}$ and 
$a_{tj}, b_{tj} \in [ -\infty, +\infty ] \ \forall j\in J_t$, we have 
$$P(y=1 \mid \mathbf{x}\in\mathcal{X}) = \sum_{t=1}^T r_t \ \mathbb{I}(\mathbf{x}\in \prod_{j\in J_t} (a_{tj}, b_{tj}]) 
= \sum_{t=1}^T r_t \  \prod_{j\in J_t} \mathbb{I}(a_{tj}<x_j\leq b_{tj})$$

That is a decision tree simply partitions the feature space into rectanguloids and predicts the chance of fraud in each rectanguloid to be the fraud incidence for the portion of the data $\mathcal{D}$ that falls into the rectanguloid.

Regarding optimization, optimization for non-tree-based models is a walk in the park.  It is easy to find sources that state the function that for a given set of hyperparameters is to be optimized to get the model parameters. For tree-based methods, this seems much harder to find.  Various sources such as Wikipedia, X and X say that under very general conditions regarding the loss function finding the optimal decision tree is an NP-hard problem. [^1]  I assume that attempting to minimize the log-loss or regularized log-loss would fit such criteria. But I wasn't able to find a source saying e.g. that the objective in fitting a decision tree (or random forest or gradient boosted trees) is to minimize the regularized log-loss.  Instead, sources present the "greedy" algorithm that optimizes each step in a tree-building and tree-pruning process, and (sometimes) acknowledge that the result is "suboptimal" without specifyig the optimization that would make a tree "optimal". (Or maybe I didn't read the papers carefully enough?)    [^2]  [^3]  [^4]  [^5]  [^6] 

So rather than presenting a definitive-sounding "Optimization" section, I present an "Optimization (my take)" reflecting my reading between the lines. 

$\textbf{Optimization (my take)}$: Having determined the leaf weights, all that remains to build the tree is determining the leaves themselves $-$ that is, the partitioning of the feature space into 
$L_1,..., L_T$ for some $T>0$. For given $\alpha>0$, the partition $L_1,..., L_T$ should minimize the regularized log-loss:
$$\text{RegLogLoss}(f_{\mathbf{L}}, \mathbf{\alpha}) = \frac{1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i L(y_i, \sum_{t=1}^T r_t \ \mathbb{I}(\mathbf{X}_i\in L_t)) + \alpha T$$ 

Of course, the model parameters $L_1,..., L_T$ aren't real numbers, so we can't even talk about $\text{RegLogLoss}(f_{\mathbf{L}}, \mathbf{\alpha})$ being differentiable in $\mathbf{L}$. So unlike logistic regression, we can't minimize the regularized log-loss through numerical methods. Nielsen points out that the optimization is NP-hard, requiring essentially an exhaustive search of the possible solutions. Knowing each 
$L_t$ has the form $L_t = \prod_{j\in J_t} (a_{tj}, b_{tj}]$, you could express the goal as a constrained optimization problem in which you are to find the 
$J_1,..., J_T \subseteq \{1,..., m\}$ and $\{a_{tj}, b_{tj}: 1\leq t\leq T, j\in J_t\}\subseteq [ -\infty, +\infty ]$ that minimize this regularized log-loss, subject to the constraint that the 
$L_t = \prod_{j\in J_t} (a_{tj}, b_{tj}]$ partition the feature space $\mathcal{X}$. But part of the search space (for 
$J_1,..., J_T\subseteq \{1,..., m\}$ is discrete, hence the need for an exhaustive search. So the problem is NP-hard unless you have a small number of features (and who has that?). 

So instead of trying to solve the optimization, there are "greedy" algorithms for approximate solutions that do just fine, thank you very much. I won't describe them here.

Finally, we note that cost-complexity pruning determines its optimal subtrees using this regularized log-loss. 

$\mathbf{Notes}$:

•	Decision trees aren't susceptible to features with wildly different means & variances, so one needn't standardize numeric features before applying

•	The effect of redundant features on decision trees is a little nuanced. Trees aren't susceptible to redundant features, e.g. multicollinearity, in the sense that, e.g. if two features are highly correlated, a decision tree might use one feature or the other as it makes its splits.  But the tree will still be built and approximately solve the regularized log-loss. What will be lost though, if you're not aware of the correlation (or in general, which features are mutually redundant), is the magnitude of feature importance (which might be divided among the redundant features).

•	And they're easily interpreted

•	But if they're not sufficiently pruned, then they tend to overfit the data (low bias, high variance) 


## 2.4 Random forests

Random forests address the high variance of decision trees. It does this by averaging the results of several trees, each built on a randomized pertubation of the data $\mathcal{D}$ and a random subset of the features.  

The randomized pertubation is accomplished by bootstrapping the data.  That is, by replacing $\mathcal{D}$ with a simple random sample with replacement of the same size $|\mathcal{D}|$.

$\textbf{Model form}$: 
$$P(y=1 \mid \mathbf{x}\in\mathcal{X}) = \frac{1}{K} \sum_{k=1}^K P_k (y=1 \mid \mathbf{x})$$ where $P_k (y=1 \mid \mathbf{x})$ is the prediction from a decision tree 
$T_k$ trained on a bootstrap sample of size $|\mathcal{D}|$ from the data $\mathcal{D}$ and from a simple random sample of 
$F$ features, for some $F\geq 1$. (The same value of $F$ is used for each tree.)

So the random forest predicts the chance of fraud given $\mathbf{x}\in\mathcal{X}$ to be the average of the class-weighted fraud incidence in the 
$K$ leaves to which $\mathbf{x}$ belongs.

$\textbf{Optimization}$: The same sources cited for decision trees seem to suggest, but not state explicitly, that the model parameters for each tree in the forest should be designed to minimize the regularized loss function 
$$\text{RegLogLoss}(f_{\mathbf{L}}, \mathbf{\alpha}) = \frac{1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i L(y_i, \sum_{t=1}^T r_t \ \mathbb{I}(\mathbf{X}_i\in L_t)) + \alpha T$$ 

The general technique of taking a class of models (like decision trees), and forming several models from the class by bootstrapping the data is called $\textit{bagging}$. So a random forest is a collection of bagged decision trees. Bagging might or might not involve using random subsets of features for the models (but for random forests, it does).

Random forests typically have lots of trees.  The default in scikit-learn is 100 trees.  As each tree is based on different data and different features, the trees can have different depths and numbers of nodes.

Hastie, Trevor; Tibshirani, Robert; Friedman, Jerome (2008). The Elements of Statistical Learning (2nd ed.). Springer. ISBN 0-387-95284-5. reports that the number of features subsetted for each tree is typically $\sqrt{m}$ (rounded down) for classification trees and 
$m/3$ (rounded down) for regression, provided there are at least 5 samples per node.

$\textbf{Notes}$:

•	The random perturbations to the training data and features considered make random forests outperform decision trees. In fact, random forests are described in the Handbook as having state-of-the-art performance for fraud detection. 

•	Like decision trees, random forests aren't susceptible to features with wildly different means & variances, so one needn't standardize numeric features before applying.

•	Random forests are less susceptible to feature redundancy than individual trees are. [^7] 

•	Random forests are harder to interpret than decision trees.  For any given feature vector, you can certainly identify the leaves it falls in, and their associated class-weighted fraud incidence as indicated by the data $\mathcal{D}$.  But with often 100 trees, this information is probably too complex to be helpful.

## 2.5 Gradient boosted trees

Having looked at one type of ensemble (bagging), we now look at another (boosting).  While the decision trees in a random forest can be built in parallel, those in gradient boosting are built sequentially. And rather than simply averaging the tree predictions like in random forests, gradient boosted trees predict the log-odds as a linear combination of the log-odds predicted from the trees.  I will only look at XGBoost, which seems popular.

$\textbf{Model form}$: 
$$P(y=1 \mid \mathbf{x}\in\mathcal{X}) = \sigma(b+\eta \sum_{k=1}^K f_k(\mathbf{x})) = \frac{1}{1+ \exp(-b-\eta \sum_{k=1}^K f_k(\mathbf{x}))}$$ where $b$ is the log-odds of the fraud rate in 
$\mathcal{D}$, $0<\eta<1$ is a hyperparameter (the "learning rate"), $K\geq 1$, and $f_1(\mathbf{x}),..., f_K(\mathbf{x})$ are the predictions from decision trees determined by the boosting algorithm.  (Although scikit-learn accepts learning rates larger than 1, it seems to make most sense to limit to smaller learning rates.) 

$\textbf{Optimization}$: For given values of the hyperparameters $\eta, \lambda, \alpha>0$, and taking 
$$b:= \frac{\sum_{i=1}^n s_i y_i }{\sum_{i=1}^n s_i(1-y_i)}$$ the log-odds trees $f_1,..., f_K$ should attempt to minimize 
$$\text{RegLogLoss}= L\left(y_i, \sigma(b+\eta \sum_{k=1}^K f_k(\mathbf{X}_i))) \right) + \sum_{k=1}^K \left( \gamma T_k + \lambda ||r_k||^2 + \alpha |r_k| \right)$$

As with random forests, the resulting trees can have different depths and numbers of leaves.

## 2.6 Support vector machines

SVMs are an outlier in two respects: They don't use the same loss function: Instead of log-loss, they use hinge loss.
And they predict a classification of fraud-vs-legit (separating the two via a hyperplane in a high-dimensional space defined by the kernel), not a formula for $P(y=1 \mid \mathbf{x}\in\mathcal{X})$. You can generate probabilities via "Platt scaling", which fits a sigmoid function to the predicted class using validation. We'll give both versions of the model form:

$\textbf{Model form for class prediction}$: Given a kernel 
$K:\mathbb{R}^m \times \mathbb{R}^m\rightarrow \mathbb{R}$, 
$$y = 0.5 + 0.5 \ sgn \left( \sum_{i=1}^n w_i (2y_i - 1) K(\mathbf{X}_i, \mathbf{x})+b \right)$$

$\textbf{Model form for probabilities}$: Given a kernel $K:\mathbb{R}^m \times \mathbb{R}^m\rightarrow \mathbb{R}$, 
$$P(y=1 \mid \mathbf{x}\in\mathcal{X}) = \sigma \left( A \left( \sum_{i=1}^n w_i (2y_i - 1) K(\mathbf{X}_i, \mathbf{x})+b \right) + B \right)$$

The model parameters are $w_1,\ldots, w_n\geq 0$, $b\in\mathbb{R}$, plus any parameters involved in the kernel 
$K$. The parameters $A$ and $B$ are considered to be hyperparameters as they are usually fit using validation. The $2y_i - 1$ term simply translates our {0,1} encoding of legit-vs-fraud to a {-1,1} encoding, and the 0.5 terms in the class predictions translate them back.  The sgn() function returns the sign of its argument (+1, -1, or 0). The support vectors are those $\mathbf{X}_i$ for which $w_i>0$.

We will focus on the "linear kernel", which is simply the dot product, becuase this turns out to be optimal for the Handbook data. In this case, $\sum_{i=1}^n w_i (2y_i - 1) K(\mathbf{X}_i, \mathbf{x})+b$ is just a linear combination of the entries of $\mathbf{x}$.

$\textbf{Optimization}$: Given $\lambda > 0$ and a kernel $K:\mathbb{R}^m \times \mathbb{R}^m\rightarrow \mathbb{R}$, the model parameters $\mathbf{w}, b, \mathbf{\xi}$ are to satisfy: 

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \;
\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n w_i w_j \,\tilde{y}_i \tilde{y}_j \, K(\mathbf{X}_i, \mathbf{X}_j)
\;+\; \frac{1}{\lambda} \sum_{i=1}^n s_i \,\xi_i
$$

$$
\text{subject to} \quad
\tilde{y}_i \left( \sum_{j=1}^n w_j \tilde{y}_j K(\mathbf{X}_j, \mathbf{X}_i) + b \right) \geq 1 - \xi_i,
\quad \xi_i \geq 0, \quad \forall i \in \{1, \ldots, n\}
$$

where $\tilde{y}_i = 2y_i - 1.$ [^8] [^9] 

## 2.7 K-nearest neighbors

$\textbf{Model form}$: Given $k\geq 1$, estimate  
$$P(y=1 \mid \mathbf{x}\in\mathcal{X}) = \frac{\sum_{i\in N_k(\mathbf{x})} s_i y_i}{\sum_{i\in N_k(\mathbf{x})} s_i}$$ where $N_k(\mathbf{x})$ is the set of indices of the 
$k$ samples in $\mathcal{D}$ with the 
$k$ smallest values of $||\mathbf{X}_i - \mathbf{x}||$. 

$\textbf{Optimization}$: The parameter $k$ is normally considered a hyperparameter (i.e. not learned from the data). So, the model has no model parameters, and no optimization.  

Also, I'm using Euclidean distance, although other choices are possible.

## 2.8 Neural networks

For now at least, I'm just going to consider feedforward nerual networks (aka multilayer perceptrons) with ReLU activations on all hidden layers and a sigmoid output layer.  This seems to be a standard deep neural network architecture for fraud detection

$\textbf{Model form}$: Given $L\geq 1$, $$P(y=1 \mid \mathbf{x}\in\mathcal{X}) = \sigma(W_L a_{L-1} + b_L)$$ where
$a_0:=\mathbf{x}$ and for each $1\leq k\leq L-1$, $a_k:=ReLU(W_k a_{k-1} + b_k)$. So the model parameters are the 
$m\times m$ matrices $W_k$ and the vectors $b_k\in\mathbb{R}^m$.

$\textbf{Optimization}$: Put all the parameters from the $W_k$ and $b_k$ into one long parameter vector $\mathbf{w}\in\mathbb{R}^{Lm(m+1)}$. For a given value of the L2 hyperparameter $\lambda>0$, the model parameters $w$ are determined by minimizing the regularized log-loss: $$\text{RegLogLoss}(f_{\mathbf{w}}, \mathbf{\lambda}) = \frac{1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i L(y_i, P(y=1 \mid \mathbf{X}_i)) + \lambda \|w\|^2$$
noting that each $P(y=1 \mid \mathbf{X}_i)$ is a function of the the parameters $\mathbf{w}$. 

[^1]: https://en.wikipedia.org/wiki/Decision_tree_learning#cite_note-35

[^2]: Nielsen, D. (2016). Tree boosting with XGBoost: Why does XGBoost win "every" machine learning competition? [Master’s thesis, Norwegian University of Science and Technology]. NTNU Open. https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf?sequence=1&isAllowed=y. 


[^3]: Konstantinov, A. V., & Utkin, L. V. (2025). A novel gradient-based method for decision trees optimizing arbitrary differentiable loss functions. arXiv preprint arXiv:2503.17855. https://arxiv.org/abs/2503.17855

[^4]: Kohler, H., Akrour, R., & Preux, P. (2025). Breiman meets Bellman: Non-Greedy Decision Trees with MDPs. arXiv. https://arxiv.org/html/2309.12701v5#S3

[^5]: Bongiorno, D., D’Onofrio, A., & Triki, C. (2024). Loss-optimal classification trees: A generalized framework and the logistic case. TOP, 32(2), 409–446. https://doi.org/10.1007/s11750-024-00674-y

[^6]: van der Linden, J. G. M., Vos, D., de Weerdt, M., Verwer, S., & Demirović, E. (2025). Optimal or greedy decision trees? Revisiting their objectives, tuning, and performance. arXiv. https://arxiv.org/abs/2409.12788

[^7]: See for instance: GeeksforGeeks. (2025, July 23). Solving the multicollinearity problem with decision tree. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/solving-the-multicollinearity-problem-with-decision-tree/

[^8]: Ding, Y., & Huang, S. (2024). A generalized framework with adaptive weighted soft-margin for imbalanced SVM classification. arXiv. https://arxiv.org/abs/2403.08378

[^9]: Hastie, T., Rosset, S., Tibshirani, R., & Zhu, J. (2004). The entire regularization path for the support vector machine. Journal of Machine Learning Research, 5, 1391–1415. http://www.jmlr.org/papers/volume5/hastie04a/hastie04a.pdf



<table width="100%">
  <tr>
    <td align="left">
      <a href="1-commonly-used-models.html">← Previous: 1. Commonly used models</a>
    </td>
    <td align="right">
      <a href="3-the-data-we-use.html">Next: 3. The data we use →</a>
    </td>
  </tr>
</table>
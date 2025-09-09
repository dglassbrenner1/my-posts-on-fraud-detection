---
layout: default     # use your main layout
title: 2. What's the same and what's different         # page title
use_math: true
---

# 2. What's the same and what's different 

Let's talk about what aspects of model training and validation are the same as for general binary classification problems, and what aspects differ.

## 2.1 Setup

### Data and features
We use the following notation for the our labeled input data $\mathcal{D}$:

$$\mathcal{D}:=\{(x_{11},\ldots, x_{1m}, y_1),\ldots, (x_{n1},\ldots,x_{nm}, y_n)\}\subseteq \mathcal{X} \times \{ 0,1 \}$$ 

where eachof the $n$ vectors in $\mathcal{D}$ represents a transaction, $\mathcal{X}\subseteq \mathbb{R}^m$ denotes the feature space, and the target class $y=1$ denotes fraudulent transactions.  We'll assume that any feature engineering has already taken place. (So the $m$ features include all engineered features.)  We also assume that any categorical features have already been numerically encoded in some fashion. We will use boldface type to indicate vectors, e.g., $\mathbf{y}:=(y_1,\ldots, y_n)$.  For the doubly-indexed 
$x_{ij}$, we let $\mathbf{x}_i := (x_{i1},\ldots,x_{im})$, the features for the $i$th transaction, for 
$1\leq i\leq n$.

### Models

Consider a randomly selected transaction $(\mathbf{x}, y)\in\mathcal{X}\times \{ 0,1\}$. We are interested in modeling $P(y=1 \mid \mathbf{x}\in\mathcal{X})$, the likelihood of fraud conditioned on the values of the features. Any given model of this probability specified by parameters $\mathbf{w}\in\mathbb{R}^W$ for some $W\geq 1$ will be a function 
$$f_{\mathbf{w}}:\mathcal{X}\rightarrow [0,1]$$ 
with $f_{\mathbf{w}}(\mathbf{x})$ denoting the model's predicted probabilty of fraud $P(y=1 \mid \mathbf{x}\in\mathcal{X})$ at feature vector $\mathbf{x}$.  

So in this setup, we have our data $\mathcal{D}$, fully feature-engineered and numerically encoded.  Let's say we are considering a class of models (e.g. gradient-boosted trees or neural networks). Let's talk about what's the same and different for finding the "best" model (i.e., the best $\mathbf{w}$) to predict fraud, compared to other binary classification problems.

## 2.2 What's the same

The goal and general framework for accomplishing them remain the same.  

### The goal in general terms 
We're still trying to fit a model $f_{\mathbf{w}}$ that performs well not just on the data it was trained on, but more importantly, on data the model hasn't seen yet (generalizability). 

### How we try to accomplish the goal
We still try to accomplish the goal (finding the $\mathbf{w}$ for which $f_{\mathbf{w}}$ best predicts future fraud) through a process like the following.  

1. **Set up your training/validation framework**: Choose indices for either training data 
$$T_1\subseteq \{1, \ldots,n \}$$ and validation data 
$$V_1\subseteq\{1, \ldots,n\}$$, or a $k$-fold cross-validation setup 
$(T_1, V_1), \ldots, (T_k, V_k)$ for $k\geq 1$. (I'm identifying members 
$$(\mathbf{x}_i, y_i)$$ of 
$\mathcal{D}$ with their indices 
$$i\in \{ 1, \ldots, n \}$$
.) Set $k:=1$ in the former case and 
$$T:=\cup_{i=1}^k T_i$$ in either case (so $T$ corresponds to the full training dataset).  Also select a test dataset for final model assessment.
   
2. **Specify a loss function (or two)**: Specify two loss function $$Loss:\mathbb{R}^W\times\mathcal{P}(\{1,\ldots, n\})\rightarrow\mathbb{R}$$ and $$ValLoss:\mathbb{R}^W\times\mathcal{P}(\{1,\ldots, n\})\rightarrow\mathbb{R}$$ each of which assesses, for a given set of model parameters $\mathbf{w}\in\mathbb{R}^W$ and 
$S\subseteq \{1, \ldots, n \}$, how close the predicted values $$\{ f_{\mathbf{w}} (\mathbf{x}_i): i\in S \}$$ are to the actual values 
$\{ y_i: i\in S \}$.  
$Loss$ will be the training loss function and $ValLoss$ will be the validation loss. (They can be the same function, if desired.) 
   
3. **Specify a regularization function**: Define a regularization function $$\Omega:\mathbb{R}^W\times\Lambda\rightarrow\mathbb{R}$$ that penalizes each of the ways your model can be complex (deeper trees, smaller leaves, etc) to varying degrees. The degrees of penalization are specified by hyperparameters $\mathbf{\lambda}$ that take values in a hyperparameter space $\Lambda\subseteq\mathbb{R}^H$ where $H$ is the number of hyperparameters.
   
4. **Tune the hyperparameters**: Determine values for the hyperparameters that minimize (or seek to minimize) the validation loss on the validation data (or the average loss on the validation folds) when applied to the model parameters $\mathbf{w}_{\mathbf{\lambda}}$ that minimize the regularized loss on the training data (or the average regularized loss on the training folds) for $\lambda$. That is, set the tuned hyperparameters to be: 
$$\mathbf{\lambda}^*:= \argmin_{\mathbf{\lambda} \in \Lambda} \left( \sum_{i=1}^k ValLoss\left( \argmin_{\mathbf{w}\in\mathbb{R}^W} \left(\Omega(\mathbf{w}, \mathbf{\lambda}) + \frac{1}{k}\sum_{j=1}^k Loss(\mathbf{w}, T_j) \right), V_i\right) \right)$$ (I'm omitting the multiplicative constant $1/k$ from the $ValLoss$ term.  Also, note that the validation loss isn't regularized. The purpose of regularization is to constrain the model parameters, not the hyperparameters.  The hyperparameters are constrained by the choice of $\Lambda$.) 
   
5. **Train the final model**: Determine the model parameters $\mathbf{w}\in\mathbb{R}^W$ that minimize the regularized log-loss using the tuned hyperparameters $\mathbf{\lambda}^*$ on the full training data $T$.  That is, set the model parameters to be: 
$$\mathbf{w}^*:= \argmin_{\mathbf{w}\in\mathbb{R}^W} \left( \Omega(\mathbf{w}, \mathbf{\lambda}^*) + Loss(\mathbf{w}, T) \right)$$ 
   
6. **Assess the final model**: Assess the final model's capabilities on unseen data by computing any desired performance metrics on the test data.

And we can still apply additional techniques to reduce or prevent overfitting, such as dropping neurons in neural networks or stopping training when the validation loss starts to increase.

### A side note on where these six steps came from

Given the variety of problems that models are designed to address, it's probably not surprising that there is not a one-size-fits-all approach to how to build the "best" model to solve any given problem. The six steps above are based loosely on [^1] and [^2].  

I was interested in a formulation that specifies the optimizations involved. This helps me not only better conceptualize the process. It also helps me think about how to modify it to address particular circumstances - like handling unsual characteristics present in the data or solving particular business objectives. The bi-level optimization in Step 4 is based largely on [^3].

Although Step 4 might look like complicated, it's not, at least for our models that don't involve trees. In practice, hyperparameter tuning seems to usually be done by randomly or exhaustively searching a small search space $\Lambda$ of hyperparameter combinations (via, e.g., scikit-learn's RandomizedSearchCV or GridSearchCV) and computing each validation log-loss. And the optimizations of the regularized log-loss in Steps 4 and 5 are essentially the same problem on different sets of data (the training folds and the full training data). For our non-tree-based models, the regularized log-loss will be twice differentiable and convex in the $\mathbf{w}$ and so the solution can be approximated by gradient descent or Newton's method. 

For tree-based models, the regularized log-loss won't even be continuous in the model parameters $\mathbf{w}$. I think what happens is that the optimization of the regularized log-loss becomes NP-hard, essentially requiring an exhaustive search of a large parameter space. I think that's why "greedy" algorithms are employed in tree-based methods to solve a sequence of analogous optimizations, building out close-to-optimal trees one split at a time.

Returning to the question at hand (what's the same and what's different), the general framework for training and validation are the same for fraud detection (compared to other applications).  But some choices made in this framework tend to differ for fraud then other binary classification problems.  We address this next.

## 2.3 What's different

The differences will stem from these aspects of fraud data:

- Fraud is very rare (class imbalance).
- The data comes in a stream over time, and fraud patterns evolve over time.
- The cost of missing a case of fraud is much greater than the cost of misidentifying a legitimate transaction as fraudulent.
- When a transaction is flagged as potential fraud, analysts typically investigate all transactions from the same card made during some period of time.
- Some transactions might be so suspicious that it's best to block them.

The following table summarizes how these fraud characteristics are handled in fraud detection.

| Fraud characteristic                                 | How it is addressed                                                                                                                                                |
|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Class imbalance                                      | - Upweight fraudulent transactions in the labelled data<br>- Apply resampling to give fraudulent transactions more influence <br>- Use performance metrics like Precision-Recall that aren't dwarfed by the class imbalance         |
| Differential cost of misclassification               | - Apply cost-sensitive learning (incorporate costs in the class weights)                                                                                          |
| Stream of data over time                             | - Use prequential validation to reflect the temporal aspect                                                                                                       |
| Fraud patterns change over time                      | - Retrain models frequently<br>- Apply anomaly detection to detect new patterns<br>- Use adaptive learning to learn new patterns                                  |
| Fraud analysts investigate multiple transactions per card | Include Precision-$k$ at the card level as a performance metric, with $k$ set according to total analyst capacity                                        |
| Some transactions might be so suspicious that it's best to block them | Balance risk vs reward to identify thresholds for such transactions                                                                        |

## Up next

We will explore each of these in the coming sections. But next, I want to write down the formulas for our models and the optimizaiton problems solved when fitting them. 

## References

[^1]: Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: Data mining, inference, and prediction (2nd ed.). Springer. https://hastie.su.domains/ElemStatLearn/download.html

[^2]: Nielsen, D. (2016). Tree boosting with XGBoost: Why does XGBoost win "every" machine learning competition? [Master’s thesis, Norwegian University of Science and Technology]. NTNU Open. https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf?sequence=1&isAllowed=y. 

[^3]: Franceschi, L., Frasconi, P., Salzo, S., Grazzi, R., & Pontil, M. (2018). Bilevel programming for hyperparameter optimization and meta-learning. Proceedings of the 35th International Conference on Machine Learning (ICML), 80, 1568–1577. https://proceedings.mlr.press/v80/franceschi18a.html


<table width="100%">
  <tr>
    <td align="left">
      <a href="1-commonly-used-models.html">← Previous: 1. Commonly used models</a>
    </td>
    <td align="right">
      <a href="3-model-formulas.html">Next: 3. Model formulas →</a>
    </td>
  </tr>
</table>
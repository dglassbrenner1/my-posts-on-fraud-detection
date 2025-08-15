
## My goal

My goal for this section seems simple enough. For each of the models we've identified as useful for fraud detection, I want to do two seemingly simple things:  1) give the formula for the model's predicted fraud rates as a function of the feature vector and any necessary parameters, and 2) give the criteria that is optimized in fitting the model parameters.  

For instance, if we were talking (in a non-fraud scenario) about ordinary least squares linear regression on a single feature with an L2 regularization penalty, this is easy: 

$\textbf{\hspace{2cm}Formula for predictions}$: $y=w_0 + w_1 x$ where $w_0, w_1\in \mathbb{R}$ are the model parameters. 
        
$\textbf{\hspace{2cm}Optimization}$: The model parameters are determined by minimizing $\sum_{i=1}^n (y_i - w_0 - w_1 x_i)^2 + \lambda (w_0^2 + w_1^2)$ where $\{(x_1,y_1),..., (x_n, y_n)\}\subseteq \mathbb{R}^2$ is the data to which we are fitting the model. 

A secondary objective is that because class imbalance is sometimes dealt with in fraud detection with class weighting, I wanted to see where the class weights appeared in these formulations.
    
So how hard can this be...?

## Setup

We use the following notation for the our labeled input data $\mathcal{D}$:
$$\mathcal{D}:=\{(X_1, y_1),..., (X_n, y_n)\}\subseteq \mathcal{F} \times \{ 0,1 \} $$ 
where each $1\leq i\leq n$ represents a transaction, $\mathcal{F}\subseteq \mathbb{R}^m$ denotes the feature space, and the target class $y=1$ denotes fraudulent transactions.  We'll assume that any feature engineering has already taken place. (So the $m$ features include all engineered features.)  We also assume that any categorical features have already been numerically encoded in some fashion. 

I assume the reader is familiar with the concepts of training, validation, cross-validation, test data, and tuning hyperparameters. I'll be pretty loose about referring to the entire dataset versus the training data, assuming the reader can infer the choice from context (e.g. $\mathcal{D}$ should refer to the training data when training a model, versus the entire dataset when fitting a final model using tuned hyperparameters). 

As usual, $X$ will denote the $m \times n$ matrix whose columns are $X_1, ..., X_n$ and $y\in \mathbb{R}^n$ will be the column vector with entries $y_1,..., y_n$. The $(i,j)$th entry of $X$ is the value of the $j$th feature in the $i$th sample. In an abuse of notation, we will sometimes use $y\in \mathbb{R}^n$ to denote a variable, e.g. $P(y=1 \mid \mathbf{x}\in\mathcal{F})$, rather than a vector of known constants, with context (hopefully!) indicating our intention. 

### Models

Of course, we are interested in modeling $P(y=1 \mid \mathbf{x}\in\mathcal{F})$, the likelihood of fraud conditioned on the values of the features. 

Models have parameters.  If the vector of parameters $\mathbf{w}\in\mathbb{R}^W$ for some $W\geq 1$ defines the model $\mathcal{M}$, we'll use the notation $\mathcal{M}(\mathbf{x} \mid \mathbf{w})$ for the model's predicted probabilty of fraud at feature vector $\mathbf{x}$. So $\mathcal{M}(\mathbf{x} \mid \mathbf{w})$ is model $\mathcal{M}$'s prediction for $P(y=1 \mid \mathbf{x}\in\mathcal{F})$. 

Sometimes the model parameters $\mathbf{w}$ will simply consistent of coefficients for the $m$ features plus a bias term $w_0$, as with logisitic regression.  Sometimes, the model parameters will look different, as with decision trees. 

Note that for fixed $\mathbf{w}\in\mathbb{R}^W$, $\mathcal{M}(\mathbf{x} \mid \mathbf{w})$ can be viewed as a function of $\mathbf{x}\in\mathcal{F}$. And for fixed $\mathbf{x}\in\mathcal{F}$, $\mathcal{M}(\mathbf{x} \mid \mathbf{w})$ can be viewed as a function of $\mathbf{w}\in\mathbb{R}^W$. We will take each viewpoint at various times. 

### Class weights

Fraud data is often be adjusted by class weights to address the fact that we're most interested in the fraud signal and fraud is rare.  We'll use $s_1, ..., s_n \geq 0$ to denote the class weights, sometimes referring to them via $\mathbf{s}:=(s_1, ..., s_n) \in \mathbb{R}_n$.  (Fraud data rarely seems to involve probalistic sampling. But if it does, include the sample weights in the $s_i$ too.)

### Loss function

I'll just use the log-loss (aka cross-entropy, aka negative log-likelihood) function, which seems to be the most often used for fraud. Given a set of predictions $z_1, ..., z_n$ for the data $\mathcal{D}$, the log-loss is:

$$\text{LogLoss}(z_1, ..., z_n):=\frac{-1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i
\left(y_i \ln(z_i) + (1 - y_i) \ln(1 - z_i)\right)$$

To statisticians, this is familiar as the negative log-likelihood applied to a binomial distribution.  If the class weights were all 1, $-\text{LogLoss}z_1, ..., z_n)$ would be the chance of seeing the data $y_1,..., y_n$ if we flipped $n$ coins (independently) with $P(\text{heads})$ for the $i^{\text{th}}$ coin being $z_i$, and wrote down a list of the results, with 1 for every heads and 0 for every tails.

### Logistic loss function 

It will sometimes be convenient to express our predictions in terms of log-odds probabilities, rather than probabilities.  When we express the log-loss as a function of log-odds and class labels, it is calle the logistic loss function. 

$$\begin{align*}\text{LogisticLoss}(u_1, ..., u_n) 
&:=\text{LogLoss}\left(\frac{1}{1+\exp(-u_1)}, ..., \frac{1}{1+\exp(-u_n)}\right) \\ 
&= \frac{1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i
\left(- y_i u_i + \ln(1 + \exp(u_i))\right)\end{align*}$$

Note in particular that $\mathbf{u}^*\in\mathbb{R}^n$ minimizes $\text{LogisticLoss}(\mathbf{u})$ if and only if $\mathbf{z}^*:=\left(\frac{1}{1+\exp(-u_1^*)}, ..., \frac{1}{1+\exp(-u_n^*)}\right)$ minimizes $\text{LogLoss}(\mathbf{z})$.  

And of course, log-odds are used to express log loss as a function of logistic loss: $$\text{LogLoss}(z_1, ..., z_n)=\text{LogisticLoss}\left(\ln\frac{z_1}{1-z_1}, ..., \ln \frac{z_n}{1-z_n}\right)$$ And $\mathbf{z}^*\in\mathbb{R}^n$ minimizes $\text{LogLoss}(\mathbf{z})$ if and only if $\mathbf{u}^*:=\left(\ln\frac{z_1^*}{1-z_1^*}, ..., \ln\frac{z_1^*}{1-z_1^*}\right)$ minimizes $\text{LogisticLoss}(\mathbf{u})$.

Of course, our predictions will come from a model $\mathcal{M}$ specified by parameters $\mathbf{w}\in\mathbb{R}^W$ and so $\hat{p}_i = \mathcal{M} (X_i \mid \mathbf{w})$.  With the data and any class weights assigned to them being constant, the log-loss of any given model $\mathcal{M}$ on our data $\mathcal{D}$ can be viewed as a function in the unknown model parameters $\mathbf{w}$. This is how we will usually view it, writing:

$$F_\mathcal{M} (\mathbf{w}):= \text{LogLoss}(\mathcal{M} (X_1 \mid \mathbf{w}), ..., \mathcal{M} (X_n \mid \mathbf{w})) = \frac{-1}{\sum_{i=1}^n s_i} \sum_{i=1}^n s_i
\left(y_i \ln(\mathcal{M} (X_i \mid \mathbf{w})) + (1 - y_i) \ln(1 - \mathcal{M} (X_i \mid \mathbf{w}))\right)$$



#### The gradient and Hessian of the logistic loss

Note that if $f(z):=-yz+\ln (1+\exp(z))$ with $y$ viewed as a constant, then $f'(z):=-y+\frac{1}{1+\exp(-z)}$ and $f''(z):=\frac{1}{1+\exp(-z)} \left( 1 - \frac{1}{1+\exp(-z)}\right)$. So, if $z$ is a log-odds and $y\in \{y_1,...,y_n\}$, say $z=\ln \frac{p}{1-p}$, then $f'(z) = p-y$ is a residual and $f''(z) = p(1-p)$.  Further, noting that the logistic loss is a sum of such functions, 
$$\nabla \text{LogisticLoss}(z_1, ..., z_n) = (p_1-y_1,..., p_n-y_n)$$
is the residual vector where $p_i:=\frac{1}{1+\exp(z_i)} \ \forall 1\leq i\leq n$.  Since the $i$th entry of the gradient involves only $z_i$, the Hessian of the logistic loss is a diagonal matrix with the $i$th diagonal entry being $p_i (1-p_i)$.

So the second order Taylor series approximation of the logistic loss at a point $\mathbf{z}^*=(\ln \frac{p_1^*}{1-p_1^*}, ..., \ln \frac{p_1^*}{1-p_1^*})$ is very nicely expressed as:

$$\text{LogisticLoss}(\mathbf{z}^*+\mathbf{z}) \approx \text{LogisticLoss}(\mathbf{z}^*) 
+ \sum_{i=1}^n (y_i - p_i^*) z_i + + \sum_{i=1}^n p_i^* (1 - p_i^*) z_i^2$$

### Regularization

We will pretty much stick to L2 regularization, which seems to be commonly used.  The regularization function for a model $\mathcal{M} (x \mid \mathbf{w})$ is $$\Omega(\mathbf{w}) := \lambda\frac{\|w\|^2}{\sum_{i=1}^n s_i}$$
where $\lambda>0$ is a hyperparameter (the "regularization strength").

The regularized log-loss is 

$$\text{LogLoss}(\mathcal{M} (X_1 \mid \mathbf{w}), ..., \mathcal{M} (X_n \mid \mathbf{w})) + \lambda\frac{\|w\|^2}{\sum_{i=1}^n s_i} $$

The regularized log-loss evaluated at some values of hyperparameters is generally what we would expect to be minimized to determine model parameters. This function is twice-differentiable in $\textbf{w}$ and so you could minimize it by gradient descent or Newton's method, as mentioned in: Nielsen, D. (2016). Tree boosting with XGBoost: Why does XGBoost win "every" machine learning competition? [Master’s thesis, Norwegian University of Science and Technology]. NTNU Open. https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf?sequence=1&isAllowed=y. 

## Logistic regression

$\textbf{Model form}$: $$P(y=1 \ | \ (x_1,..., x_m)\in\mathcal{F}) = \frac{1}{1 + \exp(-\sum_{i=1}^n w_i x_i  - w_0)}$$
 where $w_0,..., w_m \in \mathbb{R}$. 

$\textbf{Optimization}$: The model parameters $w_0,..., w_m$ are determined by minimizing the regularized loss function:

 $$\text{LogLoss}(\mathcal{M}(X_1 \mid \mathbf{w}), ..., \mathcal{M}(X_n \mid \mathbf{w})) + \lambda\frac{\|w\|^2}{\sum_{i=1}^n s_i}$$

where $$\mathcal{M}(x_1,..., x_m \mid w_0,..., w_m):= \frac{1}{1 + \exp(-\sum_{i=1}^m w_i x_i  - w_0)}$$ (In scikit-learn, you specify the "inverse regularization strength $C$, defined as $C:=1/(2\lambda)$.)  

For any given $C>0$, the regularized log-loss is a function of $w_0,..., w_m$, and so an approximate minimum can be found by methods like gradient descent. 

Noting that the log-odds predictions are linear in the features, the optimization should be more nicely expressed in the logistic loss.  That is, we find $\textbf{w}$ that minimize

$$\text{LogisticLoss}\left(\sum_{i=1}^n w_i X_{i1}  - w_0,..., \sum_{i=1}^n w_i X_{im}  - w_0 \right) + \lambda\frac{\|w\|^2}{\sum_{i=1}^n s_i}$$

It's not particularly instructive to write out the regularized loss function as a function of the model parameters, but just for the heck of it, and ignoring the multiplicative constant $\frac{1}{\sum_{i=1}^n s_i}$, the model parameters $w_0,\ldots , w_m$ minimize $$\sum_{i=1}^n s_i
\left(y_i \ln(1 + \exp(-\sum_{i=1}^n w_i X_i  - w_0)) + (1 - y_i) \ln(1 + \exp(\sum_{i=1}^n w_i x_i  - w_0))\right) + \lambda\frac{\|w\|^2}{\sum_{i=1}^n s_i}$$

Notes:

•	Logistic models are easy to interpret.

•	Susceptible to redundant features, e.g. multicollinearity

•	Susceptible to features with wildly different means & variances, so standardize numeric features before applying


## Decision trees

We will adopt the sometimes-used convention that trees output log-odds, not probabilities. 

Trees are formed by repeatedly partitioning the feature space into half spaces $\{ \mathbf{x}\in\mathbb{R}^m: x_i \leq c_i\}$ and $\{ \mathbf{x}\in\mathbb{R}^m : x_i > c_i \}$ where $1\leq i\leq m$ and $c_i\in\mathbb{R}$. This gives rise to leaves of the form $\prod_{j\in J} \{a_j < x_j \leq b_j\}$ where $J\subseteq \{1,...,m\}$ and $a_j, b_j \in [ -\infty, +\infty ] \ \forall j\in J$. (The interval $(a_j, b_j]$ can be bounded at both ends if the $j$th feature is visited multiple times in the tree.) Geometrically, the leaves are rectanguloids. 

$\textbf{Model form}$: $$P(y=1 \mid (x_1,..., x_m)\in\mathcal{F}) = \frac{1}{1+ \exp(-\sum_{t=1}^T w_t \ I((x_1,..., x_m)\in L_t))}$$ where $I(.)$ is the boolean indicator function (taking the value 1 if its argument is true and 0 otherwise), the leaves $L_1,... L_T$ are rectanguloids partitioning the feature space, and the "leaf weights" $w_1,..., w_T$ are real numbers. Note that becuase the leaves partition the feature space, each prediction is the log-odds of one of the leaf weights.  Writing each leaf $L_t$ as $$L_t = \prod_{j\in J_t} (a_{tj}, b_{tj}]$$ where $J_t\subseteq \{1,...,m\}$ and $a_{tj}, b_{tj} \in [ -\infty, +\infty ] \ \forall j\in J_t$, we have $$P(y=1 \mid \mathbf{x}\in\mathcal{F}) = \frac{1}{1+ \exp(-\sum_{t=1}^T w_t \ I(\mathbf{x}\in \prod_{j\in J_t} (a_{tj}, b_{tj}]))} 
= \frac{1}{1+ \exp(-\sum_{t=1}^T w_t \  \prod_{j\in J_t} I(a_{tj}<x_j\leq b_{tj}))}$$

$\textbf{An aside - the leaf weights are determined when the leaves are}$: Note that the constant prediction $w$ that minimizes the log-loss on a any subset $A$ of the data $\mathcal{D}$ is $w=\ln \frac{\hat{p}_A}{1-\hat{p}_A}$ where $\hat{p}_A:=\sum_{i\in A} s_i y_i / \sum_{i\in A} s_i$ is the class-weighted fraud rate.  (The log-loss on $A$ is $\sum_{i\in A} s_i \left( y_i \ln(1+\exp(-w)) +(1-y_i)\ln(1+\exp(w))\right)$.  As a function of $w$, the sole critical point of this log-loss is $w=\ln \frac{\hat{p}_A}{1-\hat{p}_A}$ and its second derivative is a positive constant.)

Because of this observation, we can rewrite the decision tree's model predictions more explicitly by filling in the value of $w_t$ as the log-odds of the class-weighted fraud incidence in the leaf $L_t$. This gives $$P(y=1 \mid \mathbf{x}\in\mathcal{F}) = \frac{\sum_{\mathbf{X_i}\in L} s_i y_i}{\sum_{\mathbf{X_i}\in L} s_i}$$ where $L$ is the unique leaf containing $\mathbf{x}$.  That is a decision tree simply partitions the feature space into rectanguloids and predicts the chance of fraud in each rectanguloid to be the fraud incidence for the portion of the data $\mathcal{D}$ that falls into the rectanguloid.

Returning to building the tree, having determined the leaf weights, all that remains is determining the leaves themselves $-$ that is, the partitioning of the feature space into $L_1,..., L_T$ for some $T>0$. Based on the optimization for logistic regression, you might expect the leaves to be determined by minimizing a regularized log-loss.  The regularization won't involve an L2 term like $\sum_{i=1}^n w_i^2 /(2C)$, since we just saw that values of $w_1,..., w_T$ are known once $L_1,..., L_n$ are. But maybe something regularized by the number of leaves, like:  $$\text{LogLoss}(\mathcal{M}(X_1 \mid \mathbf{L}), ..., \mathcal{M}(X_n \mid \mathbf{L})) + \alpha T$$ for some regularizing parameter $\alpha>0$. (Here we have replaced the $\mathbf{w}$ in $\mathcal{M}(X_i \mid \mathbf{w})$ with $\mathbf{L} = (L_1,..., L_T)$, our sole remaining model "parameters". Knowing each $L_t$ has the form $L_t = \prod_{j\in J_t} (a_{tj}, b_{tj}]$, you could express the goal as a constrained optimization problem in which you are to find the $J_1,..., J_T \subseteq \{1,..., m\}$ and $\{a_{tj}, b_{tj}: 1\leq t\leq T, j\in J_t\}$ that minimize this regularized log-loss, subject to the constraint that the $L_t = \prod_{j\in J_t} (a_{tj}, b_{tj}]$ partition the feature space $\mathcal{F}$. Since the minimization is done using the data $\mathcal{D}$, you could limit the search for the $a_{tj}$ and $b_{tj}$ for each $j$ to the values of $X_j$ occurring in $D$. So in theory, you could exhaustively search the finitely many possibilities for the optimal solution. 

And papers like the following suggest that trees should be designed to minimize a regularized loss function:

- Konstantinov, A. V., & Utkin, L. V. (2025). A novel gradient-based method for decision trees optimizing arbitrary differentiable loss functions. arXiv preprint arXiv:2503.17855. https://arxiv.org/abs/2503.17855

 - Kohler, H., Akrour, R., & Preux, P. (2025). Breiman meets Bellman: Non-Greedy Decision Trees with MDPs. arXiv. https://arxiv.org/html/2309.12701v5#S3

But this isn't how it's done.  I imagine the search is too costly or not worth doing when you can implement a simpler approximate solution that seems to perform perfectly adequately.  Whatever the reason, here is what is done. First one recursively partitions the feature space into half-spaces in a way that minimizes the total (unregularized) log-loss of each candidate split. , with one step for building a tree to solve a sequence of optimization problems and then pruning it back to solve another.  

$\textbf{Optimization in tree building}$: The model parameters (i.e., the leaves and leaf weights) are determined by recursively partitioning the feature space into half spaces $\{ \mathbf{x}\in\mathbb{R}^m: x_j \leq c\}$ and $\{ \mathbf{x}\in\mathbb{R}^m : x_j > c \}$ in a way that minimizes the sum of the log-losses of the two resulting (temporary) leaves.  From the earlier aside, we know that the optimal predicted value on each half space (optimal in terms of being the constant prediciton that minimize log-loss) is the class-weighted proportion of fraud on the half space.  So we can write the log-loss of each half-space as a function of $j$, $c$, the data points $\mathcal{D}$, and the class weights $\mathbf{s}$.  Treating the data and class weights as constants, we can search through the possible values of $1\leq j\leq m$ and $c\in \{X_{1j},..., X_{nj}\}$ for the pair giving the lowest total log-loss. 

One continues in this manner, partitioning each child node to solve the same optimization problem (minimizing the total log-loss on their children).  We continue this process until reaching user-specified stopping criteria governing the minimum number of samples required per node and the maximum depth of the tree (two hyperparameters). 

The resulting tree doesn't necessarily minimize the log-loss among trees of the same maximum depth and min number of samples per node.  It is simply an (apparently good enough) approximation to the solution, formed by focusing on solving the analogous optimization at each step.  

$\textbf{Optimization in tree pruning}$:  Once the tree is built out to the stopping criteria, cost-complexity pruning guards against overfitting. It aims to minimize the above regularized log-loss (with the $\alpha T$ regularization term) by starting with the tree $T_{full}$ just built and removing the subtree $S$ that minimizes $$\frac{\text{LL}(S) - \text{LL}(T_{full})}{|Leaves(S) - Leaves(T_{full})}$$

where $LL(X)$ (respectitvely, $Leaves(X)$) denotes the log-loss (respectively, number of leaves) for tree $X$.  Continue removing subtrees until you reach the root node (the entire feature space).  You now have a sequence of trees.  Choose the one among them with the best regularized log-loss.

Again, the resulting tree doesn't solve any pre-specified optimization problem.  It is just a good-enough approximate solution to minimizing the log-loss regularized by the number of leaves.

$\mathbf{Notes}$:


•	Decision trees aren't susceptible to features with wildly different means & variances, so one needn't standardize numeric features before applying

•	The effect of redundant features on decision trees is a little nuanced. Trees aren't susceptible to redundant features, e.g. multicollinearity, in the sense that, e.g. if two features are highly correlated, a decision tree might use one feature or the other as it makes its splits.  But the tree will still be built and approximately solve the regularized log-loss. What will be lost though, if you're not aware of the correlation (or in general, which features are mutually redundant), is the magnitude of feature importance (which might be divided among the redundant features).

•	And they're easily interpreted

•	But if they're not sufficiently pruned, then they tend to overfit the data (low bias, high variance) 


## Random forests

Random forests address the high variance of decision trees. It does this by averaging the results of several trees, each built on a randomized pertubation of the data $\mathcal{D}$ and a random subset of the features.  

The randomized pertubation is accomplished by bootstrapping the data.  That is, by replacing $\mathcal{D}$ with a simple random sample with replacement of the same size $|\mathcal{D}|$.

The general technique of taking a class of models (like decision trees), and forming several models from the class by bootstrapping the data is called $\textit{bagging}$. So a random forest is a collection of bagged decision trees. Bagging might or might not involve using random subsets of features for the models (but for random forests, it does).

Random forests typically have lots of trees.  The default in scikit-learn is 100 trees.

Hastie, Trevor; Tibshirani, Robert; Friedman, Jerome (2008). The Elements of Statistical Learning (2nd ed.). Springer. ISBN 0-387-95284-5. reports that the number of features subsetted for each tree is typically $\sqrt{m}$ (rounded down) for classification trees and $m/3$ (rounded down) for regression, provided there are at least 5 samples per node.

Random forests do not aim to minimize some regularized loss function.

$\textbf{Model form}$: $$P(y=1 \mid \mathbf{x}\in\mathcal{F}) = \frac{1}{K} \sum_{k=1}^K P_k (y=1 \mid \mathbf{x})$$ where $P_k (y=1 \mid \mathbf{x})$ is the prediction from the $k$th tree in the forest.  And we saw earlier that the decision tree prediction is the class-weighted fraud incidence in the leaf to which $\mathbf{x}$ belongs. So the random forest predicts the chance of fraud given $\mathbf{x}\in\mathcal{F}$ to be the average of the class-weighted fraud incidence in the $K$ leaves to which $\mathbf{x}$ belongs.

$\textbf{Notes}$:

•	The random perturbations to the training data and features considered make random forests outperform decision trees. In fact, random forests are described in the Handbook as having state-of-the-art performance for fraud detection. 

•	Like decision trees, random forests aren't susceptible to features with wildly different means & variances, so one needn't standardize numeric features before applying.

•	Random forests are less susceptible to feature redundancy.  See for instance:

GeeksforGeeks. (2025, July 23). Solving the multicollinearity problem with decision tree. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/solving-the-multicollinearity-problem-with-decision-tree/

•	Random forests are harder to interpret than decision trees.  For any given feature vector, you can certainly identify the leaves it falls in, and their associated class-weighted fraud incidence as indicated by the data $\mathcal{D}$.  But with often 100 trees, this information is probably too complex to be helpful.

## Gradient boosted trees

Having looked at one type of ensemble (bagging), we now look at another (boosting).  While the decision trees in a random forest can be built in parallel, those in gradient boosting are built sequentially.

$\textbf{Model form}$: $$P(y=1 \mid \mathbf{x}\in\mathcal{F}) = \frac{1}{1+ \exp(-b-\eta \sum_{k=1}^K f_k(\mathbf{x}))}$$ where $b$ is the log-odds of the fraud rate in $\mathcal{D}$, $0<\eta<1$ is a hyperparameter (the "learning rate"), $K\geq 1$, and $f_1(\mathbf{x}),..., f_K(\mathbf{x})$ are the predictions from decision trees determined by the boosting algorithm.  (Although scikit-learn accepts learning rates larger than 1, it seems to make most sense to limit to smaller learning rates.) 

The idea of the boosting algorithm is to build a sequence of trees in which each is trained on the remaining residuals. So the first tree is the constant model that predicts the log-odds class-weighted fraud rate in $\mathcal{D}$.  The next tree is fitted to the residuals ${y_1 -r$
---
layout: default     # use your main layout
title: 8. Cost-sensitive and imbalanced learning         # page title
---

# 8. Cost-sensitive and imbalanced learning

Now another of the key ways that modeling fraud differs from modeling other things. Fraud is imbalanced in two ways: Fraud is rare. And the cost of false negatives (missing fraud) is typically much greater than the cost of false positives (flagging legit transactions as fraud).

## 8.1 Cost-sensitive learning

Suppose you estimate each false positive to cost $C_{FP}$ dollars and each false negative to cost 
$C_{FN}$.  An obvious way to incorporate these costs into the loss function is to set the class weights $s_i$ for $i\in\mathcal{D}$ to be  $C_{FN}$ for fraudulent transactions and 
$C_{FP}$ for legitimate transactions. That is:


$$
s_i := \begin{cases}
C_{FN} & \text{if } y_i = 1\\
C_{FP} & \text{if } y_i = 0
\end{cases}
$$

With these class weights and ignoring the constant factor $\frac{1}{\sum_{i=1}^n s_i}$, the cost-adjusted regularized log-loss function would be:

$$\text{RegLogLoss}(f_{\mathbf{w}}, \mathbf{\lambda}) = - \sum_{i=1}^n  
\left( C_{FN} \ y_i \ln f_{\mathbf{w}}(\mathbf{X}_i) + C_{FP} (1 - y_i) \ln(1 - f_{\mathbf{w}}(\mathbf{X}_i)) \right) + \Omega (\mathbf{w},\mathbf{\lambda})$$

For instance, when training the model from already-tuned hyperparameters, the differential costs make missing fraud a more costly error, all alse being equal, than wrongly flagging legit transactions.  And if the measure of validation loss also incorporates these costs, they can also impact the hyperparameter tuning.

The Handbook says that it is difficult to estimate the costs of false positives and false negatives. (I'm guessing a card issuer would have a good sense of the cost of missing fraud, at least in terms of refunding cardholders for transactions they didn't authorize. And I'm guessing card issuers have a good sense of investigating wrongly flagged transactions that turn out to be legitimate. But maybe it's hard to estimate the costs of losing customers who get annoyed by declined transactions and holds placed on their cards?) 

When these costs can't be reasonably reliably estimated, the Handbook notes that a popular heuristic is to assume that false negatives cost $1/IR$ times as much as false positives, where the *imbalance ratio* $IR$ is defined as the ratio of fraudulent transactions to legitimate transactions. 

This heuristic doesn't sound unreasonable, but I don't want to mix up imbalanced learning and cost-sensitive learning. It sounds like, in general, one of the ways of handling class imbalance is by upweighting the minority class, regardless of whether you are also attempting to incorporate the differential costs of misclassification. For instance, it seems like you can incorporate both concepts at once, learning in a manner that addresses both class imbalance and cost imbalance.  So let's explore this. 
 ( $ of  ing cost imbalance by 

Because transactions come in a stream of data and since fraud patterns vary over time, fraud models are often 
validated using a variant to cross-validation that reflects these. In *prequential validation*, the training and validation 
folds shift over time, as illustrated in this image from the 
prequential validation from the Handbook:


![prequential-image-fraud-detection-handbook](./images/prequential-image-fraud-detection-handbook.png)
**Prequential validation illustration, from “2. Validation strategies,” in** *Reproducible Machine Learning for Credit Card Fraud detection - Practical handbook* **by Yann-Aël Le Borgne, Wissam Siblini, Bertrand Lebichot, Gianluca Bontempi, Université Libre de Bruxelles and Worldline Labs.**  
Available at: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html  
Licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

I refit the models using 4-fold prequential validation and got ...

<details>
<summary>Click to expand/hide Python code</summary>

<pre> ```python

``` </pre>
</details>

<table width="100%">
  <tr>
    <td align="left">
      <a href="/6-the-cost-of-fraud-to-the-card-issuer.html">← Previous: The cost of fraud</a>
    </td>
    <td align="right">
      Next: Post to come! →</a>
    </td>
  </tr>
</table>


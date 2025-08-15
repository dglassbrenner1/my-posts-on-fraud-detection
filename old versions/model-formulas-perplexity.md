# Common Machine Learning Models for Fraud Detection

***
Here is a list of the most commonly used models in fraud detection, along with their predicted fraud rate formulas as functions of features, and the typical functions optimized to determine model parameters:

***

### 1. Logistic Regression  
**Predicted fraud probability:**  
$$
\hat{p}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
$$  
where $\mathbf{x}$ is the feature vector, $\mathbf{w}$ the weights (parameters), and $b$ the intercept.

**Optimized function:**  
The model parameters $\mathbf{w}, b$ are found by minimizing the **logistic loss (binary cross-entropy):**  
$$
\min_{\mathbf{w}, b} \; - \sum_{i=1}^n \left[ y_i \log \hat{p}(\mathbf{x}_i) + (1 - y_i) \log (1 - \hat{p}(\mathbf{x}_i)) \right]
$$  
where $y_i \in \{0,1\}$ are true fraud labels.

***

### 2. Decision Trees & Random Forests  
**Predicted fraud rate:**  
For a decision tree, the predicted probability is the fraction of fraud cases in a leaf node:  
$$
\hat{p}(\mathbf{x}) = \frac{\text{Number of fraud samples in leaf containing } \mathbf{x}}{\text{Total samples in that leaf}}
$$

Random Forest aggregates many trees, averaging their predictions:  
$$
\hat{p}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T \hat{p}_t(\mathbf{x})
$$

**Optimized function:**  
Trees are grown by greedily minimizing criteria like **Gini impurity** or **entropy** at splits, e.g. the Gini impurity for a node with classes $$k$$:  
$$
G = 1 - \sum_k p_k^2
$$  
where $$p_k$$ is the fraction of class $$k$$ samples in the node.

***

### 3. Gradient Boosting Machines (e.g., XGBoost, LightGBM)  
**Predicted fraud rate:**  
Boosting models produce an additive score combined via a logistic function to output probabilities:  
$$
\hat{p}(\mathbf{x}) = \sigma \left( \sum_{m=1}^M f_m(\mathbf{x}) \right)
$$  
where each $f_m$ is a decision tree learner.

**Optimized function:**  
Minimize the regularized negative log-likelihood or logistic loss:  
$$
\min_{f_m} \sum_{i=1}^n \log \left(1 + e^{-y_i \sum_{m=1}^M f_m(\mathbf{x}_i)}\right) + \Omega(f_m)
$$  
where $\Omega$ is the regularization for tree complexity.
The general optimization objective for tree ensemble methods (such as Gradient Boosted Decision Trees) can be written in LaTeX as:

$$
\left\{ \hat{w}_{jm}, \hat{R}_{jm} \right\}_{j=1,m=1}^{T_m, M} = \arg\min_{\{w_{jm}, R_{jm}\}_{j=1,m=1}^{T_m, M}} \sum_{i=1}^n L\left(y_i, \sum_{m=1}^M \sum_{j=1}^{T_m} w_{jm} \mathbb{I}(x_i \in R_{jm}) \right)
$$

where:
- $M$ is the number of trees,
- $T_m$ is the number of leaves in tree $m$,
- $w_{jm}$ is the weight of leaf $j$ in tree $m$,
- $R_{jm}$ is the region (leaf) $j$ in tree $m$,
- $L$ is the loss function (e.g., logistic loss),
- $\mathbb{I}(x_i \in R_{jm})$ is the indicator function for $x_i$ belonging to region $R_{jm}$.
***

### 4. Neural Networks (Deep Learning)  
**Predicted fraud probability:**  
Using a neural network parameterized by weights $\theta$, outputting a logit $z(\mathbf{x}; \theta)$:  
$$
\hat{p}(\mathbf{x}) = \sigma(z(\mathbf{x}; \theta))
$$

**Optimized function:**  
Cross-entropy loss (same as logistic regression):  
$$
\min_{\theta} - \sum_{i=1}^n \left[ y_i \log \hat{p}(\mathbf{x}_i) + (1 - y_i) \log (1 - \hat{p}(\mathbf{x}_i)) \right]
$$

***

### 5. Isolation Forest (Anomaly Detection)  
**Predicted fraud score:**  
An unsupervised model that isolates anomalies (fraud) by random partitioning, producing an anomaly score $$s(\mathbf{x})$$ inversely related to average path length in trees.

**Optimized function:**  
Not parameterized by a standard loss; instead, the model tries to isolate points via random splits, with fraud cases expected to have shorter paths.

***

### 6. Autoencoders (Anomaly Detection)  
**Predicted fraud score:**  
Reconstruction error from an autoencoder (neural network designed to reproduce input):  
$$
s(\mathbf{x}) = \| \mathbf{x} - \hat{\mathbf{x}} \|^2
$$  
where $\hat{\mathbf{x}}$ is the reconstructed input.

**Optimized function:**  
Minimize reconstruction loss, often MSE:  
$$
\min_{\theta} \sum_{i=1}^n \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2
$$

***

### Summary Table

| Model Type           | Predicted Fraud Rate Formula                                      | Function Optimized (Loss)                               |
|---------------------|------------------------------------------------------------------|--------------------------------------------------------|
| Logistic Regression  | $$\hat{p}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$ | Logistic loss (cross-entropy)                           |
| Decision Trees/Random Forest | Leaf class proportion average                                  | Gini impurity or entropy for node splits                |
| Gradient Boosting (XGBoost) | $$\hat{p}(\mathbf{x}) = \sigma(\sum f_m(\mathbf{x}))$$         | Regularized logistic loss                               |
| Neural Networks      | $$\hat{p}(\mathbf{x}) = \sigma(z(\mathbf{x}; \theta))$$           | Cross-entropy loss                                     |
| Isolation Forest     | Anomaly score $$s(\mathbf{x})$$ based on isolation depth          | No explicit loss; unsupervised anomaly isolation        |
| Autoencoders         | Reconstruction error $$\| \mathbf{x} - \hat{\mathbf{x}} \|^2$$     | Mean squared error (reconstruction loss)                |

***

Sources:

[1] https://trustdecision.com/resources/blog/5-new-machine-learning-algorithms-for-fraud-detection
[2] https://coralogix.com/ai-blog/how-to-optimize-ml-fraud-detection-a-guide-to-monitoring-performance/
[3] https://www.numberanalytics.com/blog/loss-function-techniques-ml
[4] https://sqream.com/blog/fraud-detection-machine-learning/
[5] https://kount.com/blog/precision-recall-when-conventional-fraud-metrics-fall-short
[6] https://arxiv.org/html/2508.02283v1
[7] https://www.reddit.com/r/learnmachinelearning/comments/1g6jx90/trying_to_build_an_effective_fraud_detection/
[8] https://learn.microsoft.com/en-us/fabric/data-science/fraud-detection
[9] https://pmc.ncbi.nlm.nih.gov/articles/PMC10332194/
[10] https://www.reddit.com/r/datascience/comments/1aes9jn/resources_for_fraud_detectionprevention/
[11] https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
[12] https://www.sciencedirect.com/science/article/abs/pii/S2210650225000926
[13] https://www.sciencedirect.com/science/article/pii/S2772662223000036
[14] https://ligsuniversity.com/financial-statement-fraud-detection-machine-learning-model-design/
[15] https://businessanalytics.substack.com/p/loss-functions-explained
[16] https://www.ravelin.com/insights/machine-learning-for-fraud-detection
[17] https://www.linkedin.com/pulse/working-metrics-loss-functions-aiml-reeshabh-choudhary-yqbhf
[18] https://www.sas.com/en_us/insights/articles/risk-fraud/fraud-detection-machine-learning.html
[19] https://pubmed.ncbi.nlm.nih.gov/37434626/
[20] https://arxiv.org/html/2307.02694v3
### 1. Logistic Regression  
**Predicted fraud probability:**  
$$
\hat{p}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
$$  
where $$\mathbf{x}$$ is the feature vector, $$\mathbf{w}$$ the weights (parameters), and $$b$$ the intercept.

**Optimized function:**  
The model parameters $$\mathbf{w}, b$$ are found by minimizing the **logistic loss (binary cross-entropy):**  
$$
\min_{\mathbf{w}, b} \; - \sum_{i=1}^n \left[ y_i \log \hat{p}(\mathbf{x}_i) + (1 - y_i) \log (1 - \hat{p}(\mathbf{x}_i)) \right]
$$  
where $$y_i \in \{0,1\}$$ are true fraud labels.

***

### 2. Decision Trees & Random Forests  
**Predicted fraud rate:**  
For a decision tree, the predicted probability is the fraction of fraud cases in a leaf node:  
$$
\hat{p}(\mathbf{x}) = \frac{\text{Number of fraud samples in leaf containing } \mathbf{x}}{\text{Total samples in that leaf}}
$$

Random Forest aggregates many trees, averaging their predictions:  
$$
\hat{p}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T \hat{p}_t(\mathbf{x})
$$

**Optimized function:**  
Trees are grown by greedily minimizing criteria like **Gini impurity** or **entropy** at splits, e.g. the Gini impurity for a node with classes $$k$$:  
$$
G = 1 - \sum_k p_k^2
$$  
where $$p_k$$ is the fraction of class $$k$$ samples in the node.

***

### 3. Gradient Boosting Machines (e.g., XGBoost, LightGBM)  
**Predicted fraud rate:**  
Boosting models produce an additive score combined via a logistic function to output probabilities:  
$$
\hat{p}(\mathbf{x}) = \sigma \left( \sum_{m=1}^M f_m(\mathbf{x}) \right)
$$  
where each $$f_m$$ is a decision tree learner.

**Optimized function:**  
Minimize the regularized negative log-likelihood or logistic loss:  
$$
\min_{f_m} \sum_{i=1}^n \log \left(1 + e^{-y_i \sum_{m=1}^M f_m(\mathbf{x}_i)}\right) + \Omega(f_m)
$$  
where $$\Omega$$ is the regularization for tree complexity.

***

### 4. Neural Networks (Deep Learning)  
**Predicted fraud probability:**  
Using a neural network parameterized by weights $$\theta$$, outputting a logit $$z(\mathbf{x}; \theta)$$:  
$$
\hat{p}(\mathbf{x}) = \sigma(z(\mathbf{x}; \theta))
$$

**Optimized function:**  
Cross-entropy loss (same as logistic regression):  
$$
\min_{\theta} - \sum_{i=1}^n \left[ y_i \log \hat{p}(\mathbf{x}_i) + (1 - y_i) \log (1 - \hat{p}(\mathbf{x}_i)) \right]
$$

***

### 5. Isolation Forest (Anomaly Detection)  
**Predicted fraud score:**  
An unsupervised model that isolates anomalies (fraud) by random partitioning, producing an anomaly score $$s(\mathbf{x})$$ inversely related to average path length in trees.

**Optimized function:**  
Not parameterized by a standard loss; instead, the model tries to isolate points via random splits, with fraud cases expected to have shorter paths.

***

### 6. Autoencoders (Anomaly Detection)  
**Predicted fraud score:**  
Reconstruction error from an autoencoder (neural network designed to reproduce input):  
$$
s(\mathbf{x}) = \| \mathbf{x} - \hat{\mathbf{x}} \|^2
$$  
where $$\hat{\mathbf{x}}$$ is the reconstructed input.

**Optimized function:**  
Minimize reconstruction loss, often MSE:  
$$
\min_{\theta} \sum_{i=1}^n \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2
$$

***

### Summary Table

| Model Type           | Predicted Fraud Rate Formula                                      | Function Optimized (Loss)                               |
|---------------------|------------------------------------------------------------------|--------------------------------------------------------|
| Logistic Regression  | $$\hat{p}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$ | Logistic loss (cross-entropy)                           |
| Decision Trees/Random Forest | Leaf class proportion average                                  | Gini impurity or entropy for node splits                |
| Gradient Boosting (XGBoost) | $$\hat{p}(\mathbf{x}) = \sigma(\sum f_m(\mathbf{x}))$$         | Regularized logistic loss                               |
| Neural Networks      | $$\hat{p}(\mathbf{x}) = \sigma(z(\mathbf{x}; \theta))$$           | Cross-entropy loss                                     |
| Isolation Forest     | Anomaly score $$s(\mathbf{x})$$ based on isolation depth          | No explicit loss; unsupervised anomaly isolation        |
| Autoencoders         | Reconstruction error $$\| \mathbf{x} - \hat{\mathbf{x}} \|^2$$     | Mean squared error (reconstruction loss)                |

***

If you'd like, I can provide more detailed derivations or examples for any specific model or loss function. Let me know!

[1] https://trustdecision.com/resources/blog/5-new-machine-learning-algorithms-for-fraud-detection
[2] https://coralogix.com/ai-blog/how-to-optimize-ml-fraud-detection-a-guide-to-monitoring-performance/
[3] https://www.numberanalytics.com/blog/loss-function-techniques-ml
[4] https://sqream.com/blog/fraud-detection-machine-learning/
[5] https://kount.com/blog/precision-recall-when-conventional-fraud-metrics-fall-short
[6] https://arxiv.org/html/2508.02283v1
[7] https://www.reddit.com/r/learnmachinelearning/comments/1g6jx90/trying_to_build_an_effective_fraud_detection/
[8] https://learn.microsoft.com/en-us/fabric/data-science/fraud-detection
[9] https://pmc.ncbi.nlm.nih.gov/articles/PMC10332194/
[10] https://www.reddit.com/r/datascience/comments/1aes9jn/resources_for_fraud_detectionprevention/
[11] https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
[12] https://www.sciencedirect.com/science/article/abs/pii/S2210650225000926
[13] https://www.sciencedirect.com/science/article/pii/S2772662223000036
[14] https://ligsuniversity.com/financial-statement-fraud-detection-machine-learning-model-design/
[15] https://businessanalytics.substack.com/p/loss-functions-explained
[16] https://www.ravelin.com/insights/machine-learning-for-fraud-detection
[17] https://www.linkedin.com/pulse/working-metrics-loss-functions-aiml-reeshabh-choudhary-yqbhf
[18] https://www.sas.com/en_us/insights/articles/risk-fraud/fraud-detection-machine-learning.html
[19] https://pubmed.ncbi.nlm.nih.gov/37434626/
[20] https://arxiv.org/html/2307.02694v3Here is a list of the most commonly used models in fraud detection, along with their predicted fraud rate formulas as functions of features, and the typical functions optimized to determine model parameters:

***

### 1. Logistic Regression  
**Predicted fraud probability:**  
$$
\hat{p}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
$$  
where $$\mathbf{x}$$ is the feature vector, $$\mathbf{w}$$ the weights (parameters), and $$b$$ the intercept.

**Optimized function:**  
The model parameters $$\mathbf{w}, b$$ are found by minimizing the **logistic loss (binary cross-entropy):**  
$$
\min_{\mathbf{w}, b} \; - \sum_{i=1}^n \left[ y_i \log \hat{p}(\mathbf{x}_i) + (1 - y_i) \log (1 - \hat{p}(\mathbf{x}_i)) \right]
$$  
where $$y_i \in \{0,1\}$$ are true fraud labels.

***

### 2. Decision Trees & Random Forests  
**Predicted fraud rate:**  
For a decision tree, the predicted probability is the fraction of fraud cases in a leaf node:  
$$
\hat{p}(\mathbf{x}) = \frac{\text{Number of fraud samples in leaf containing } \mathbf{x}}{\text{Total samples in that leaf}}
$$

Random Forest aggregates many trees, averaging their predictions:  
$$
\hat{p}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T \hat{p}_t(\mathbf{x})
$$

**Optimized function:**  
Trees are grown by greedily minimizing criteria like **Gini impurity** or **entropy** at splits, e.g. the Gini impurity for a node with classes $$k$$:  
$$
G = 1 - \sum_k p_k^2
$$  
where $$p_k$$ is the fraction of class $$k$$ samples in the node.

***

### 3. Gradient Boosting Machines (e.g., XGBoost, LightGBM)  
**Predicted fraud rate:**  
Boosting models produce an additive score combined via a logistic function to output probabilities:  
$$
\hat{p}(\mathbf{x}) = \sigma \left( \sum_{m=1}^M f_m(\mathbf{x}) \right)
$$  
where each $$f_m$$ is a decision tree learner.

**Optimized function:**  
Minimize the regularized negative log-likelihood or logistic loss:  
$$
\min_{f_m} \sum_{i=1}^n \log \left(1 + e^{-y_i \sum_{m=1}^M f_m(\mathbf{x}_i)}\right) + \Omega(f_m)
$$  
where $$\Omega$$ is the regularization for tree complexity.

***

### 4. Neural Networks (Deep Learning)  
**Predicted fraud probability:**  
Using a neural network parameterized by weights $$\theta$$, outputting a logit $$z(\mathbf{x}; \theta)$$:  
$$
\hat{p}(\mathbf{x}) = \sigma(z(\mathbf{x}; \theta))
$$

**Optimized function:**  
Cross-entropy loss (same as logistic regression):  
$$
\min_{\theta} - \sum_{i=1}^n \left[ y_i \log \hat{p}(\mathbf{x}_i) + (1 - y_i) \log (1 - \hat{p}(\mathbf{x}_i)) \right]
$$

***

### 5. Isolation Forest (Anomaly Detection)  
**Predicted fraud score:**  
An unsupervised model that isolates anomalies (fraud) by random partitioning, producing an anomaly score $$s(\mathbf{x})$$ inversely related to average path length in trees.

**Optimized function:**  
Not parameterized by a standard loss; instead, the model tries to isolate points via random splits, with fraud cases expected to have shorter paths.

***

### 6. Autoencoders (Anomaly Detection)  
**Predicted fraud score:**  
Reconstruction error from an autoencoder (neural network designed to reproduce input):  
$$
s(\mathbf{x}) = \| \mathbf{x} - \hat{\mathbf{x}} \|^2
$$  
where $$\hat{\mathbf{x}}$$ is the reconstructed input.

**Optimized function:**  
Minimize reconstruction loss, often MSE:  
$$
\min_{\theta} \sum_{i=1}^n \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2
$$

***

### Summary Table

| Model Type           | Predicted Fraud Rate Formula                                      | Function Optimized (Loss)                               |
|---------------------|------------------------------------------------------------------|--------------------------------------------------------|
| Logistic Regression  | $$\hat{p}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$ | Logistic loss (cross-entropy)                           |
| Decision Trees/Random Forest | Leaf class proportion average                                  | Gini impurity or entropy for node splits                |
| Gradient Boosting (XGBoost) | $$\hat{p}(\mathbf{x}) = \sigma(\sum f_m(\mathbf{x}))$$         | Regularized logistic loss                               |
| Neural Networks      | $$\hat{p}(\mathbf{x}) = \sigma(z(\mathbf{x}; \theta))$$           | Cross-entropy loss                                     |
| Isolation Forest     | Anomaly score $$s(\mathbf{x})$$ based on isolation depth          | No explicit loss; unsupervised anomaly isolation        |
| Autoencoders         | Reconstruction error $$\| \mathbf{x} - \hat{\mathbf{x}} \|^2$$     | Mean squared error (reconstruction loss)                |

***

If you'd like, I can provide more detailed derivations or examples for any specific model or loss function. Let me know!

[1] https://trustdecision.com/resources/blog/5-new-machine-learning-algorithms-for-fraud-detection
[2] https://coralogix.com/ai-blog/how-to-optimize-ml-fraud-detection-a-guide-to-monitoring-performance/
[3] https://www.numberanalytics.com/blog/loss-function-techniques-ml
[4] https://sqream.com/blog/fraud-detection-machine-learning/
[5] https://kount.com/blog/precision-recall-when-conventional-fraud-metrics-fall-short
[6] https://arxiv.org/html/2508.02283v1
[7] https://www.reddit.com/r/learnmachinelearning/comments/1g6jx90/trying_to_build_an_effective_fraud_detection/
[8] https://learn.microsoft.com/en-us/fabric/data-science/fraud-detection
[9] https://pmc.ncbi.nlm.nih.gov/articles/PMC10332194/
[10] https://www.reddit.com/r/datascience/comments/1aes9jn/resources_for_fraud_detectionprevention/
[11] https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
[12] https://www.sciencedirect.com/science/article/abs/pii/S2210650225000926
[13] https://www.sciencedirect.com/science/article/pii/S2772662223000036
[14] https://ligsuniversity.com/financial-statement-fraud-detection-machine-learning-model-design/
[15] https://businessanalytics.substack.com/p/loss-functions-explained
[16] https://www.ravelin.com/insights/machine-learning-for-fraud-detection
[17] https://www.linkedin.com/pulse/working-metrics-loss-functions-aiml-reeshabh-choudhary-yqbhf
[18] https://www.sas.com/en_us/insights/articles/risk-fraud/fraud-detection-machine-learning.html
[19] https://pubmed.ncbi.nlm.nih.gov/37434626/
[20] https://arxiv.org/html/2307.02694v3sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
$$

**Optimized function (logistic loss):**
$$
\min_{\mathbf{w}, b} \; - \sum_{i=1}^n \left[ y_i \log \hat{p}(\mathbf{x}_i) + (1 - y_i) \log (1 - \hat{p}(\mathbf{x}_i)) \right]
$$

---

## 2. Decision Trees & Random Forests

**Predicted fraud rate (tree):**
$$
\hat{p}(\mathbf{x}) = \frac{\text{Fraud samples in leaf}}{\text{Total samples in leaf}}
$$

**Random Forest:**
$$
\hat{p}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T \hat{p}_t(\mathbf{x})
$$

**Optimized function (Gini impurity):**
$$
G = 1 - \sum_k p_k^2
$$

---

## 3. Gradient Boosting Machines (e.g., XGBoost, LightGBM)

**Predicted fraud rate:**
$$
\hat{p}(\mathbf{x}) = \sigma \left( \sum_{m=1}^M f_m(\mathbf{x}) \right)
$$

**Optimized function (regularized logistic loss):**
$$
\min_{f_m} \sum_{i=1}^n \log \left(1 + e^{-y_i \sum_{m=1}^M f_m(\mathbf{x}_i)}\right) + \Omega(f_m)
$$

---

## 4. Neural Networks (Deep Learning)

**Predicted fraud probability:**
$$
\hat{p}(\mathbf{x}) = \sigma(z(\mathbf{x}; \theta))
$$

**Optimized function (cross-entropy):**
$$
\min_{\theta} - \sum_{i=1}^n \left[ y_i \log \hat{p}(\mathbf{x}_i) + (1 - y_i) \log (1 - \hat{p}(\mathbf{x}_i)) \right]
$$

---

## 5. Isolation Forest (Anomaly Detection)

**Predicted fraud score:**  
Anomaly score $$s(\mathbf{x})$$ based on isolation depth.

**Optimized function:**  
No explicit loss; isolates anomalies via random splits.

---

## 6. Autoencoders (Anomaly Detection)

**Predicted fraud score (reconstruction error):**
$$
s(\mathbf{x}) = \| \mathbf{x} - \hat{\mathbf{x}} \|^2
$$

**Optimized function (MSE):**
$$
\min_{\theta} \sum_{i=1}^n \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2
$$

---

## Summary Table

| Model Type           | Predicted Fraud Rate Formula                                      | Function Optimized (Loss)                               |
|---------------------|------------------------------------------------------------------|--------------------------------------------------------|
| Logistic Regression  | $\hat{p}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$ | Logistic loss (cross-entropy)                           |
| Decision Trees/Random Forest | Leaf class proportion average                                  | Gini impurity or entropy for node splits                |
| Gradient Boosting (XGBoost) | $\hat{p}(\mathbf{x}) = \sigma(\sum f_m(\mathbf{x}))$         | Regularized logistic loss                               |
| Neural Networks      | $\hat{p}(\mathbf{x}) = \sigma(z(\mathbf{x}; \theta))$           | Cross-entropy loss                                     |
| Isolation Forest     | Anomaly score $s(\mathbf{x})$ based on isolation depth          | No explicit loss; unsupervised anomaly isolation        |
| Autoencoders         | Reconstruction error $\| \mathbf{x} - \hat{\mathbf{x}} \|^2$     | Mean squared error (reconstruction loss)                |

---
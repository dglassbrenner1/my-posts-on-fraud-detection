---
layout: custom     # use your main layout
title: 8. Cost-sensitive and imbalanced learning         # page title
nav_order: 9
has_toc: true
nav_enabled: true
use_math: true
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

With these class weights, the cost-adjusted regularized log-loss function becomes:

$$\text{RegLogLoss}(f_{\mathbf{w}}, \mathbf{\lambda}) = \Omega (\mathbf{w},\mathbf{\lambda}) - \frac{\sum_{i=1}^n  
\left( C_{FN} \ y_i \ln f_{\mathbf{w}}(\mathbf{X}_i) + C_{FP} (1 - y_i) \ln(1 - f_{\mathbf{w}}(\mathbf{X}_i)) \right)}{\sum_{i=1}^n \left( C_{FN} \ y_i + C_{FP} (1 - y_i)\right)} $$


When training the model from already-tuned hyperparameters, the differential costs make assigning a fraud case a low probability a more costly error, all alse being equal, than assigning a legit transaction a high probability.  And if the measure of validation loss also incorporates these costs, they will similarly impact the hyperparameter tuning.

Suppose, for instance, that a false negative costs 20 times as much as a false positive.  For simplicity, let's say $C_{FN}=20$ and $C_{FP}=1$ and let's ignore the normalizing factor (which depends on the number of samples). Then predicting a fraudulent transaction to have only a 10% chance of being fraudulent adds 46 to the log-loss. But predicting a legitimate transaction to have a 90% chance of being fraudulent adds only 2 to the log-loss.  


<details>
<summary>Click to expand/hide Python code to generate the table and plot</summary>

<pre> ```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker

p = np.linspace(0.001, 0.999, 500)
cost_ratios = [1, 2, 5, 10, 20]

viridis_full = mpl.colormaps['viridis'](np.linspace(0, 1, 256))
start = int(0.1 * 256)
end = int(0.85 * 256)
viridis_trimmed = viridis_full[start:end]
custom_cmap = ListedColormap(viridis_trimmed)
colors = custom_cmap(np.linspace(0, 1, len(cost_ratios)))

plt.figure(figsize=(8, 5))
handles = []
labels = []

# Plot fraud curves and collect handles and labels
for cost_ratio, color in zip(cost_ratios, colors):
    log_loss_fraud = -np.log(p) * cost_ratio
    h, = plt.plot(p * 100, log_loss_fraud, label=f'Fraud (cost ratio={cost_ratio})', linestyle='--', color=color)
    handles.append(h)
    labels.append(f'Fraud (cost ratio={cost_ratio})')

# Plot legit curve
log_loss_legit = -np.log(1 - p)
h_legit, = plt.plot(p * 100, log_loss_legit, label='Legit (weight=1)', color='brown')

# Append the legit curve handle and label at the end
handles.append(h_legit)
labels.append('Legit (weight=1)')

# Create an ordering list:
# Indices of fraud curves sorted by decreasing cost ratio (since cost_ratios is increasing, reverse the order)
fraud_indices_desc = list(range(len(cost_ratios)-1, -1, -1))

# Append the legit curve index last
legend_order = fraud_indices_desc + [len(cost_ratios)]  # legit is last

# Reorder handles and labels according to desired legend order
handles_ordered = [handles[i] for i in legend_order]
labels_ordered = [labels[i] for i in legend_order]

plt.legend(handles_ordered, labels_ordered)
plt.title('Cost-Weighted Log-Loss Curves vs Predicted Probability')
plt.xlabel('Predicted Probability of Fraud (%)')
plt.ylabel('Log-Loss')
plt.grid(True)

# Add percent signs on the x-axis tick labels
plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

plt.savefig("cost-wgted-log-loss-curves-vs-pred-prob.png", bbox_inches='tight')
plt.show()

p_values = [0.1, 0.9]
cost_ratios = [1, 2, 5, 10, 20]

def log_loss_fraud(p, cost_ratio):
    return -np.log(p) * cost_ratio

def log_loss_legit(p):
    return -np.log(1 - p)

rows = []
headers = ["Class / Cost Ratio", "Log-Loss at p=10%", "Log-Loss at p=90%"]

for cost_ratio in cost_ratios:
    row = [
        f"Fraud (cost ratio={cost_ratio})",
        f"{log_loss_fraud(0.1, cost_ratio):.4f}",
        f"{log_loss_fraud(0.9, cost_ratio):.4f}"
    ]
    rows.append(row)

# Add legit row
rows.append([
    "Legit (weight=1)",
    f"{log_loss_legit(0.1):.4f}",
    f"{log_loss_legit(0.9):.4f}"
])

# Print markdown table
print("| " + " | ".join(headers) + " |")
print("|" + "|".join(["---"] * len(headers)) + "|")
for row in rows:
    print("| " + " | ".join(row) + " |")

``` </pre>
</details>


| Class / Cost Ratio | Log-Loss at P(fraud)=10% | Log-Loss at P(fraud)=90% |
|---|---|---|
| Fraud (cost ratio=1) | 2.3026 | 0.1054 |
| Fraud (cost ratio=2) | 4.6052 | 0.2107 |
| Fraud (cost ratio=5) | 11.5129 | 0.5268 |
| Fraud (cost ratio=10) | 23.0259 | 1.0536 |
| Fraud (cost ratio=20) | 46.0517 | 2.1072 |
| Legit (weight=1) | 0.1054 | 2.3026 |



<img src="./images/cost-wgted-log-loss-curves-vs-pred-prob.png" alt="cost weighted log loss curves vs predicted probability" />


The Handbook says that it is difficult to estimate the costs of false positives and false negatives. (I'm guessing a card issuer would have a good sense of the cost of missing fraud, at least in terms of refunding cardholders for transactions they didn't authorize. And I'm guessing card issuers have a good sense of investigating wrongly flagged transactions that turn out to be legitimate. But maybe it's hard to estimate the costs of losing customers who get annoyed by declined transactions and holds placed on their cards?) 

When these costs can't be reasonably reliably estimated, the Handbook notes that a popular heuristic is to assume that false negatives cost $1/IR$ times as much as false positives, where the *imbalance ratio* $IR$ is defined as the ratio of fraudulent transactions to legitimate transactions. 

This heuristic doesn't sound unreasonable, but I don't want to mix up imbalanced learning and cost-sensitive learning. That is, one way to handle class imbalance is to upweight the minority class, regardless of whether you also incorporate the differential costs of misclassification. But you can also incorporate both concepts at once, learning in a manner that addresses both class imbalance and cost imbalance.  So let's explore this. 




<details>
<summary>Click to expand/hide Python code</summary>

<pre> ```python

``` </pre>
</details>

<table width="100%">
  <tr>
    <td align="left">
      <a href="/7-the-cost-of-fraud-to-the-card-issuer.html">← Previous: 7. The cost of fraud</a>
    </td>
    <td align="right">
      Next: Post to come! →</a>
    </td>
  </tr>
</table>


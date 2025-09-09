---
layout: default     # use your main layout
title: 8. Prequential validation         # page title
---

# 8. Prequential validation

Now back to modeling and one of the other ways that modeling fraud is different from other machine learning applications.

Because transactions come in a stream of data and since fraud patterns vary over time, fraud models are often 
validated using a variant of cross-validation that reflects these patterns. In *prequential validation*, the training and validation 
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
      <a href="/7-the-cost-of-fraud-to-the-card-issuer.html">← Previous: 7. The cost of fraud</a>
    </td>
    <td align="right">
      Next: Post to come! →</a>
    </td>
  </tr>
</table>


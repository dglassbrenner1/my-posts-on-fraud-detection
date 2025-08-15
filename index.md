---
layout: default     # use your main layout
title: My understanding of supervised learning for fraud detection         # page title
---

# My understanding of supervised learning for fraud detection 

## The goal of this series of posts

I’m writing this series to lock in my own understanding of supervised learning techniques in fraud detection. I’m not writing this as a tutorial for others, but if you find this helpful to advance your own understanding, great. I’ll use the mathematical and statistical language that makes sense to me and skip explanations of those foundations.  I’ll assume familiarity with basic elements of machine learning, like model fitting, validation, and tuning hyperparameters and familiarity with Python, Jupyter, and scikit-learn.

Having read a lot of material explaining machine learning in non-specific terms, I am most interested in:

•	a clear understanding of the models, metrics, and assorted techniques commonly used in fraud detection, and

•	what sets supervised learning for fraud detection apart from supervised learning in general (such as the extreme class imbalance, and the different costs for false positives vs false negatives).

When possible, I’d also like to have a visual understanding (yeah, right).  

I’ll be working with the synthetic dataset from the Fraud Detection Handbook.[^1] It’s designed to mirror real-world transaction streams and already includes several engineered features. Crucially, it simulates two common fraud scenarios—compromised point-of-sale devices and compro-mised cards—so I can see how models react to different attack patterns. I'll refer to this handbook in my series of posts as the "Handbook".

Here is a first post, on models commonly used for fraud detection and who uses them: [Commonly used supervised learning models](1-commonly-used-models.md).

Caution: These posts reflect my own understanding of the techniques I write about. No one has reviewed or verified the accuracy of my statements besides me. As you can tell from the “lock in my understanding” phrase, I am a newbie to fraud detection. Despite a lack of credentials, I will occasionally put forth my own reactions to choices made in the Handbook, along with my reasoning, for what they are worth. That all said, I cite sources where relevant and welcome constructive comments.

[^1]: Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook. Université Libre de Bruxelles. Retrieved from https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook. The data are at: Fraud-Detection-Handbook/simulated-data-transformed as individual .pkl files.  I combined them into a Parquet file for easy loading. 


<table width="100%">
  <tr>
    <td align="right">
      <a href="/my-posts-on-fraud-detection/1-commonly-used-models.html/">Next: 1. Commonly used models →</a>
    </td>
  </tr>
</table>
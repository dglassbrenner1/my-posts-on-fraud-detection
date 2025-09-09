---
layout: default     # use your main layout
title: 1. Commonly used supervised learning models         # page title
has_toc: true
nav_enabled: true
---

# 1. Commonly used supervised learning models

I want to focus on models commonly used in fraud detection. Maybe not
surprisingly, a web search doesn't uncover, e.g., exactly which types of
models Visa uses. And if you did find a post on what Visa uses, it could
well be out-of-date.

That said, here is what a web search in July 2025 uncovered:

## The models we'll look at and examples of who has used them

### Logistic Regression

- A 2023 post by Capital One mentions logistic and other forms of
  regression, suggesting Capital One has used them in some fashion.[^1]

- A 2021 post by Capital One mentions logistic regression among models
  considered for detecting money laundering. [^2]

- An undated post in Insider Finance Wire describes logistic regression
  as a "fundamental tool" in fraud detection.[^3]

<!-- -->

- A 2022 post in Fintech News says that PayPal used to use logistic
  regression: "PayPal used logistic regression for fraud detection.
  However, now it leverages advanced techniques like gradient boosted
  trees (GBTs) to improve its accuracy of ML models. Recently, it has
  started to turn to more advanced AI tech like deep learning, active
  learning and transfer learning" [^4]

<!-- -->

- The Handbook includes logistic regression among its models considered.

### Decision Trees 

- Decision trees are also used in the Handbook.

- While decision trees seem too simplistic to use as standalone fraud models, they are the fundamental
  building blocks for the next two more commonly used models (random forests and gradient-boosted
  trees).

### Random Forests 

- Random forests were the best performing model in the 2021 Capital One post on money laundering.

- The Handbook describe random forests as having state-of-the-art performance
  for fraud detection.

### Gradient-Boosted Trees 

- The 2022 Fintech News post reported gradient-boosted trees to be the then-current technique used
  by PayPal.

- XGBoost was among the models considered in the 2021 Capital One post
  on money laundering.

- Like random forests, the Handbook also characterized gradient-boosted trees as having
  state-of-the-art performance for fraud detection.

- A 2020 post by NVIDIA reported that American Express included gradient-boosted models in its portfolio. [^5]

### Support Vector Machines  

- Support Vector Machine classifiers aren't used in the Handbook or any posts I could find, they figured prominently
  in a 2019 survey by Priscilla et al that is used in the Handbook.[^6]

### k-Nearest Neighbors 

- Like Support Vector Machines, k-Nearest Neighbor models weren't used in the Handbook, but they were mentioned in the 2019
  by Priscilla et al survey.

### Neural Networks (NNs) 

- The 2022 Fintech News post reported that PayPal used them.

- A 2021 post by Stripe said they use neural networks.[^7]

- The 2021 Capital One post said they use neural networks to detect money
  laundering.

- The 2020 NVIDIA post mentions American Express also using neural networks.

These seven models seem to be the most often used models for fraud detection. Next, let's look at what aspects of applying these models to fraud detection differ from other binary classification problems.

[^1]: Capital One Tech. (2023, July 27). Boost model performance:
    Logistic regression in R. Medium.
    <https://medium.com/capital-one-tech/boost-model-performance-logistic-regression-in-r-615d18327034>

[^2]: Munoz, P., & Minnis, R. (2021, September 22). How machine learning
    can help fight money laundering. Capital One Tech. Retrieved July
    24, 2025, from
    <https://www.capitalone.com/tech/machine-learning/how-machine-learning-can-help-fight-money-laundering/>

[^3]: Logistic Regression: A Simple Powerhouse in Fraud Detection.
    (n.d.). Insider Finance Wire. Retrieved July 24, 2025, from
    <https://wire.insiderfinance.io/logistic-regression-a-simple-powerhouse-in-fraud-detection-15ab984b2102>

[^4]: FintechNews Staff. (2022, February 9). PayPal taps AI/ML in battle
    against fraud. Fintech News.
    <https://www.fintechnews.org/paypal-taps-ai-ml-in-battle-against-fraud/>

[^5]: Ashley, J. (2020, October 5). American Express Adopts NVIDIA AI to
    Help Prevent Fraud and Foil Cybercrime. NVIDIA Blog. Retrieved July
    30, 2025, from
    <https://blogs.nvidia.com/blog/american-express-nvidia-ai/>

[^6]: C Victoria Priscilla and D Padma Prabha. Credit card fraud
    detection: a systematic review. In *International Conference on
    Information, Communication and Computing Technology*, 290--303.
    Springer, 2019.

[^7]: Stripe. (2021, December 15). *A primer on machine learning for
    fraud detection*. Stripe. Retrieved July 30, 2025, from
    <https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection>


<table width="100%">
  <tr>
    <td align="left">
      <a href="/">← Previous: The goal for these posts</a>
    </td>
    <td align="right">
      <a href="2-whats-the-same-and-not.html">Next: 2. What's the same and not →</a>
    </td>
  </tr>
</table>


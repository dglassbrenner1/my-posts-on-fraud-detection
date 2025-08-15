# 1. Commonly used supervised learning models

I want to focus on models commonly used in fraud detection. Maybe not
surprisingly, a web search doesn't uncover e.g. exactly which types of
models Visa uses. And if you did find a post on what Visa uses, it could
well be out-of-date.

That said, here is what a web search in July 2025 uncovered:

## The models we'll look at and examples of who has used them

### Logistic Regression

- A 2023 post by Capital One mentions logistic & other forms of
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

- Used in the Handbook, due to its simplicity.

### Decision Trees (DTs)

- Used in the Handbook, due to its simplicity.

- Not mentioned specifically in any other posts, but DTs are fundamental
  to two other models that are used (random forests & gradient boosted
  trees).

### Random Forests (RFs)

- The chosen model in the 2021 Capital One post on money laundering.

- Used in the Handbook, described as having state-of-the-art performance
  for fraud detection.

### Gradient Boosted Trees (GBTs) 

- Mentioned in the 2022 Fintech News post as then-current technique used
  by PayPal.

- XGBoost was among the models considered in the 2021 Capital One post
  on money laundering.

- Also used in the Handbook, and also described as having
  state-of-the-art performance for fraud detection.

- A 2020 post by NVIDIA said that American Express included gradient
  boosted models in its portfolio. [^5]

### Support Vector Machines (SVMs) 

- Not used in the Handbook or any posts, but seems to figure prominently
  in a 2019 survey by Priscilla et al that is used in the Handbook.[^6]

### k-Nearest Neighbors (kNN) 

- Not used in the Handbook or any posts, but also mentioned in the 2019
  by Priscilla et al survey.

### Neural Networks (NNs) 

- The 2022 Fintech News post mentions PayPal using them.

- A 2021 post by Stripe said they use neural networks.[^7]

- The 2021 Capital One post said they use NNs to detect money
  laundering.

- The 2020 NVIDIA post mentions American Express also using NNs.

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
      <a href="index.md">← Previous: The goal for these posts</a>
    </td>
    <td align="right">
      <a href="2-model-formulas-250814.md">Next: 2. Model formulas →</a>
    </td>
  </tr>
</table>


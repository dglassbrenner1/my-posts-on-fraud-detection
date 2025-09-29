# Supervised Learning for Fraud Detection
This repository contains the work-in-progress series of posts and associated code for exploring supervised learning techniques applied to fraud detection. The project is intended primarily for deepening understanding of machine learning methods in the context of fraud detection, specifically credit card and transaction fraud.

# Project Overview
The focus of this series is to comprehensively explore:

- Commonly used supervised learning models for fraud detection.

- Challenges specific to fraud detection including extreme class imbalance, evolving fraud patterns, and asymmetric costs of false positives and false negatives.

- Business-driven objectives such as resource-limited fraud investigation capacity, minimizing fraud-related costs, and rapid detection adaptation.

- Mathematical and statistical understanding of model fitting, tuning, validation, and performance metrics.

- Two example deployments demonstrating:

  - Integration of an XGBoost fraud model within a Databricks interactive dashboard for monitoring transaction fraud and model performance.

  - An interactive fraud detection API hosted on Hugging Face Spaces with Streamlit.

This research is grounded in a synthetic dataset designed to reflect real-world fraud scenarios, sourced from the *Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook* by Le Borgne et al. This extremely helpful book is freely available online at: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html.

# Repository Contents
- **Posts**/: Markdown posts explaining models, metrics, and fraud detection concepts. All Jupyter notebooks, as well as screenshots of the Databricks dashboards and Streamlit app, are embedded in the posts.

# Getting Started
To reproduce or extend the work:

1. Clone the repository:

```bash
git clone https://github.com/dglassbrenner1/fraud-detection-supervised-learning.git
```

2. Set up a Python environment with required dependencies (e.g., scikit-learn, XGBoost, pandas, Streamlit, Databricks SDK).

3. Load the synthetic dataset and follow the posts and notebooks for guided exploration of supervised learning techniques.

4. Run the Jupyter notebooks embedded in the posts.

# Business Objectives Addressed
This work aims to provide insights and methods to answer critical questions like:

- How many fraud cases can be detected given limited analyst review capacity?

- What is the trade-off between investigation costs and fraud loss savings?

- How quickly can detection models adapt to new fraud patterns?

- What confidence bounds can be placed on fraud detection performance estimates?

# Caution and Disclaimer
This repository reflects the author's understanding and is a personal research endeavor. It is not an official tutorial and is not audited for production or compliance use. Users should supplement with domain expertise and real-world validation before deploying.

# Chief reference
Le Borgne, Y.-A., et al. (2022). #Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook#. Université Libre de Bruxelles. GitHub Repo

# Contributions
Contributions, feedback, and discussions are welcome to improve the understanding and applications shared in this series.

# Contact
Donna Glassbrenner, Ph.D.

*GitHub*: dglassbrenner1

*Website*: https://dglassbrenner1.github.io/my-posts-on-fraud-detection/

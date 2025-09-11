---
layout: custom     # use your main layout
title: Supervised learning for fraud detection         # page title
nav_order: 1
has_toc: true
nav_enabled: true
---

# My understanding of supervised learning for fraud detection 

## The goal of this series of posts

I’m writing this series to lock in my own understanding of supervised learning techniques in fraud detection. I’m not writing this as a tutorial for others, but if you find this helpful to advance your own understanding, great. I’ll use the mathematical and statistical language that makes sense to me and skip explanations of those foundations.  I’ll assume familiarity with basic elements of machine learning, like model fitting, validation, and tuning hyperparameters and familiarity with Python, Jupyter, and scikit-learn.

Having read a lot of material explaining machine learning in less-than-mathematically-specific terms, I am most interested in:

-	a clear understanding of the models, metrics, and assorted techniques commonly used in fraud detection, 

-	what sets supervised learning for fraud detection apart from supervised learning in general (such as the extreme class imbalance, the fact that fraud patterns change over time, and the different costs for false positives vs false negatives), and

- what types of business objectives can we solve?

I want to get enough into the mathematical weeds to be able to modify methods to incorporate particular business circumstances.  And I’d like to have some sense of what plots of the models look like (yeah, right), as this helps me confirm my mathematical understanding.  

## Primary data source

I’ll be working with the synthetic dataset from the online book _Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook_. [^1] The synthetic dataset the authors create is designed to mirror real-world transaction streams and already includes several engineered features. Crucially, it simulates two common fraud scenarios — compromised point-of-sale devices and compromised cards — so I can see how models react to different attack patterns. I'll refer to this online book, which also informs a lot of the technical material in these posts, as the "Handbook".

## Business objectives

To get the juices flowing, let's think about business objectives. In general terms, you want to catch as much fraud as possible, while minimizing false alarms. And you have limited resources with which to catch fraud. You have fraud analysts who can investigate a certain number of suspicious transactions (or cards with suspicious transactions) per day. And there are different costs for missing fraud vs incurring false alarms.  Thinking more specifically, you might want to ask:

- I have fraud analysts who can can collectively review x cards with suspicious transactions per hour.  How much fraud can I catch (what percent of volume or dollars) with these resources?  How many false alarms will I incur to do this?  What if I increased my investigative capacity (hired more fraud analysts)?

- Given our current transaction volume, what's the smallest total fraud-related cost (including the cost of fraud itself and the amount I spend investigating suspicious transactions) I would need to spend? How many investigators would I need to achieve it? 

- How much money will your proposed fraud detection scheme save me?

- How rapidly can your proposed fraud detection scheme respond to changing patterns of fraud?

- And, for all of the above, how confident are you in your estimates? What kinds of qualifiers and bounds would you need to put on your answers to be, say, 95% confident that your estimates will hold going forward?  

Some of these questions will surely be easier to answer than others.

Here is a first post, on models commonly used for fraud detection and who uses them: [Commonly used supervised learning models](1-commonly-used-models.md).

## Caution 
These posts reflect my own understanding of the techniques I write about. No one has reviewed or verified the accuracy of my statements besides me. As you can tell from the “lock in my understanding” phrase, I am a newbie to fraud detection. Despite a lack of credentials, I will occasionally put forth my own reactions to choices made in the Handbook, along with my reasoning, for what they are worth. That all said, I cite sources where relevant and welcome constructive comments.

[^1]: Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook. Université Libre de Bruxelles. Retrieved from https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook. The data are at: Fraud-Detection-Handbook/simulated-data-transformed as individual .pkl files.  I combined them into a Parquet file for easy loading. 


<table width="100%">
  <tr>
    <td align="right">
      <a href="1-commonly-used-models.html">Next: 1. Commonly used models →</a>
    </td>
  </tr>
</table>
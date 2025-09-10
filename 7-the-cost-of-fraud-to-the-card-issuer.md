---
layout: default     # use your main layout
title: 7. Incorporating the cost of fraud         # page title
nav_order: 8
has_toc: true
nav_enabled: true
use_math: true
---

# 7. Incorporating the cost of fraud

In the introduction, we posed questions involving cost like:

- I have fraud analysts who can can collectively review x cards with suspicious transactions per hour.  How much fraud can I catch (what percent of volume or dollars) with these resources?  How many false alarms will I incur to do this?  What if I increased my investigative capacity (hired more fraud analysts)?

Let's put together a simple formula for the cost of fraud to the card issuer.  The Handbook mentions that some fraud detection systems are developed with the costs of false positives and negatives in mind, but doesn't seem to develop this notion further.  I was curious to look at how this could play out.

## 7.1 A formula for the cost of fraud to the card issuer

Let's consider the type of scenario we were starting to flesh out at the end of the last post, but let's add in "automatic declines".  

Suppose we have a ML model with estimated CardPrecision@k and CardRecall@k. We plan to run the model every hour (or minute or second) on the cards that have not yet been identified as compromised and look at the most suspicious cards and transactions. The model will flag the most suspicious transactions (those over a certain very high threshold probaility) for automatic declines (declining all future transactions). The most suspicious cards that remain are parceled out to the fraud analysts for review, according to the number they are expected to be able to review in an hour (or minute or second). 

Assume that each time an analyst reviews a card with fraud, it is marked as compromised and instantly removed from circulation (so it won't be in the next run of the model). 
***


### Step 1: Define Input Variables

- $N$: Total number of transactions in the period
- $f$: Fraction of fraudulent transactions (transaction-level fraud rate)
- $t_{\text{avg}}$: Average number of transactions per card during the period
- $N_f$: Number of fraudulent cards in the period, calculated as $N_f = \frac{N \times f}{t_{\text{avg}}}$
- $P$: Precision of fraud model at the card level (fraction of flagged cards that are truly fraudulent)
- $R$: Recall of fraud model at the card level (fraction of fraudulent cards detected)
- $n_a$: Number of cards an analyst can review per period
- $S$: Analyst salary cost per period
- $A$: Number of analysts employed (24/7)
- $\alpha$: Avg fraction of flagged cards automatically declined (no review)
- $\gamma = 1 - \alpha$: Avg fraction of flagged cards sent for analyst review

Additional cost variables:

- $V$: Average refund cost per missed fraud card (value lost to fraud)
- $C_{fp}$: Cost of false positive card flagged among cards reviewed by analysts

***

### Step 2: Calculate Base Quantities

#### Number of Fraudulent Cards
$$
N_f = \frac{N \times f}{t_{\text{avg}}}
$$

#### True Positives (Fraudulent Cards Detected)
$$
TP = R \times N_f
$$

#### Number of Flagged Cards 
$$
N_{\text{flagged}} = \frac{TP}{P} = \frac{R N_f}{P}
$$

***

### Step 3: Partition Flagged Cards

- Automatically declined cards:
$$
N_{\text{declined}} = \alpha \times N_{\text{flagged}} = \alpha \frac{R N_f}{P}
$$

- Cards routed to analysts for review:
$$
N_{\text{reviewed, needed}} = \gamma \times N_{\text{flagged}} = (1-\alpha) \frac{R N_f}{P}
$$

***

### Step 4: Analyst Review Capacity

- Analyst review capacity (max cards reviewed per period):
$$
\text{Capacity} = A \times n_a
$$

- Actual number of cards reviewed:
$$
N_{\text{reviewed}} = \min(N_{\text{reviewed, needed}}, \text{Capacity}) = \min\left((1-\alpha) \frac{R N_f}{P}, A n_a\right)
$$

***

### Step 5: Fraud and False Positives in Reviewed Set

Assuming flagged cards are uniformly sampled with respect to fraud and legit:

- True positives among reviewed:
$$
TP_{\text{reviewed}} = \min\left(\gamma TP, P \times N_{\text{reviewed}}\right) = \min\left((1-\alpha) R N_f, P \times N_{\text{reviewed}}\right)
$$

- False positives among reviewed:
$$
FP_{\text{reviewed}} = N_{\text{reviewed}} - TP_{\text{reviewed}}
$$

***

### Step 6: Fraud Detected and Missed Overall

- Fraud detected by automatic declines:
$$
\text{Fraud}_{\text{auto}} = \alpha \times TP = \alpha R N_f
$$

- Fraud detected by analyst review:
$$
\text{Fraud}_{\text{review}} = TP_{\text{reviewed}}
$$

- Total fraud detected:
$$
\text{Fraud}_{\text{detected}} = \text{Fraud}_{\text{auto}} + \text{Fraud}_{\text{review}} = \alpha R N_f + TP_{\text{reviewed}}
$$

- Fraud missed (refunded):
$$
FN = N_f - \text{Fraud}_{\text{detected}} = N_f - \alpha R N_f - TP_{\text{reviewed}}
$$

***

### Step 7: Cost Components

- Analyst staffing cost:
$$
\text{Staffing Cost} = A \times S
$$

- False positive cost (only for reviewed cards):
$$
\text{False Positive Cost} = FP_{\text{reviewed}} \times C_{fp}
$$

- Fraud refund cost (for missed fraud cards):
$$
\text{Fraud Refund Cost} = FN \times V
$$

***

### Step 8: Final Total Cost Formula

$$
\boxed{
\begin{aligned}
\text{Total Cost} &= A S + FP_{\text{reviewed}} C_{fp} + FN V \\
&= A S + \left(N_{\text{reviewed}} - TP_{\text{reviewed}}\right) C_{fp} + \left(N_f - \alpha R N_f - TP_{\text{reviewed}}\right) V
\end{aligned}
}
$$

where

$$
\begin{aligned}
N_f &= \frac{N \times f}{t_{\text{avg}}} \\
N_{\text{reviewed}} &= \min\left((1-\alpha) \frac{R N_f}{P}, A n_a\right) \\
TP_{\text{reviewed}} &= \min\left((1-\alpha) R N_f, P \times N_{\text{reviewed}}\right)
\end{aligned}
$$

***

Or as a function of the input variables alone, the cost to the card issuer per period is:

$$
\text{Total Cost} = A \cdot S + \left( \min \left((1-\alpha) \frac{R \frac{N f}{t_{\text{avg}}}}{P}, A n_a \right) - \min \left( (1-\alpha) R \frac{N f}{t_{\text{avg}}}, P \cdot \min \left( (1-\alpha) \frac{R \frac{N f}{t_{\text{avg}}}}{P}, A n_a \right) \right) \right) C_{fp} + \left( \frac{N f}{t_{\text{avg}}} - \alpha R \frac{N f}{t_{\text{avg}}} - \min \left( (1-\alpha) R \frac{N f}{t_{\text{avg}}}, P \cdot \min \left( (1-\alpha) \frac{R \frac{N f}{t_{\text{avg}}}}{P}, A n_a \right) \right) \right) V
$$

For example, if we used the following input:

To produce a more reasonable total cost estimate for a single card issuer fraud detection per hour, here is a revised set of default input values based on transaction volume and typical cardholder behavior:

- $N = 100,000$$ transactions per hour 
- $f = 0.0005$ (0.05%) 
- $t_{avg} = 0.04$ transactions per card per hour (about 1 transaction per day) — consistent with ~257 transactions per year per cardholder
- $P = 0.85$ (85%) — precision at card level reflecting a good fraud model
- $R = 0.65$ (65%) — recall at card level
- $n_a = 500$ cards reviewed per analyst per hour
- $S = 5000/(30*24)$ analyst salary cost per hour (about $5,000/month)
- $A = 5$ analysts 
- $\alpha = 0.25$ fraction of transactions automatically declined
- $C_fp = 75$ dollars - cost per false positive card review
- $V = 1500$ dollars - average cost per missed fraud card

we'd get a cost of about $664,350 per hour.

<details>
<summary>Click to expand/hide Python code</summary>

<pre> ```python
def calculate_total_cost(N, f, t_avg, P, R, n_a, S, A, alpha, C_fp, V):
    N_f = (N * f) / t_avg
    flagged_cards = (R * N_f) / P
    reviewed_needed = (1 - alpha) * flagged_cards
    capacity = A * n_a
    N_reviewed = min(reviewed_needed, capacity)
    TP_reviewed = min((1 - alpha) * R * N_f, P * N_reviewed)
    FP_reviewed = N_reviewed - TP_reviewed
    Fraud_auto = alpha * R * N_f
    Fraud_detected = Fraud_auto + TP_reviewed
    Fraud_missed = N_f - Fraud_detected
    cost = A * S + FP_reviewed * C_fp + Fraud_missed * V
    return cost

# Default values for one-hour period
N = 100_000          # total transactions in one hour
f = 0.0005              # 0.1% fraud transaction rate
t_avg = 0.04           # avg transactions per card per hour (~1.5 transactions/day)
P = 0.85                # precision at card level
R = 0.65                # recall at card level
n_a = 500             # cards reviewed per analyst per period (hour)
S = 5000 / (30*24)     # analyst salary per hour (approx $5000/month converted)
A = 5                 # number of analysts
alpha = 0.25            # fraction auto declined
C_fp = 75             # cost per false positive card reviewed
V = 1500               # fraud cost per missed card

cost = calculate_total_cost(N, f, t_avg, P, R, n_a, S, A, alpha, C_fp, V)
print(f"Total Cost to Issuer for One Hour: ${cost:,.0f}")

``` </pre>
</details>


## 7.2 The minimum cost of fraud with a perfect ML model

Suppose we had a model that captured all fraud with no false positives, i.e. $P=1$ and $R=1$. The the cost simplifies to:

$$
\boxed{
\text{Total Cost} = A \times S + \left( \frac{N f}{t_{\text{avg}}} - \alpha \times \frac{N f}{t_{\text{avg}}} - \min\left( (1-\alpha) \times \frac{N f}{t_{\text{avg}}}, A \times n_a \right) \right) \times V
}
$$

This sums analyst staffing cost plus the fraud cost of cards missed after automated declines plus analyst review capacity, considering no false positives or missed fraud for the model's perfect accuracy.

This reflects analyst staffing plus refunds for any fraud not caught by automatic declines or analyst review capacity.

For example, using the values from before, with a perfect model the cost would drop to a mere $35 per hour.

<details>
<summary>Click to expand/hide Python code</summary>

<pre> ```python
P=1
R=1
cost = calculate_total_cost(N, f, t_avg, P, R, n_a, S, A, alpha, C_fp, V)
print(f"Total Cost to Issuer for One Hour w Perfect Model: ${cost:,.0f}")

``` </pre>
</details>

## 7.3 Optimizing staffing for a given model

Suppose we have a model with the card precision and card recall we were using earlier, namely $P = 0.85, R = 0.65$. Then the minimal cost would come from reducing staff to two fraud analysts, bringing the hourly cost of fraud down to $664,329.

<details>
<summary>Click to expand/hide Python code</summary>

<pre> ```python
# Search for integer A between 1 and 50 to minimize cost
P = 0.85               
R = 0.65

best_A = 1
best_cost = calculate_total_cost(N, f, t_avg, P, R, n_a, S, best_A, alpha, C_fp, V)

for A in range(2, 51):
    cost = calculate_total_cost(N, f, t_avg, P, R, n_a, S, A, alpha, C_fp, V)
    if cost < best_cost:
        best_cost = cost
        best_A = A

print(f"Optimal integer number of analysts: {best_A}")
print(f"Minimal total cost: ${best_cost:,.2f}")

``` </pre>
</details>

There are some weaknesses in our computations.  For instance, when automated declines remove the most suspicious transactions from the stream of potential fraud, the precision and recall of our models will surely decrease. Still, in this section, we have figured out a basic framework for how to answer some of the questions set forth at the introduction.  We calculated the minimum amount of money a card issuer would need to spend on fraud, given data on the number of transactions, prevalence of fraud, the share of transactions that can be handled with automation, and staff capacity to review cards with suspicious transactions.  We also saw how to determine the optimal number of investigative staff under these same parameters and running a model run every hour (other other time period) with solid estimates of card precision and recall.  

But right now our models are pretty simple. We haven't looked at better incorporating class imbalance into our model development or trying to model the types of fraud patterns that make the Handbook simulated data special.  So we will return to our questions of business needs after refining our models. 


<table width="100%">
  <tr>
    <td align="left">
      <a href="6-performance-metrics.html">← Previous: 6. Performance metrics</a>
    </td>
    <td align="right">
      <a href="8-prequential-validation.html"> Next: 8. Prequential validation →</a>
    </td>
  </tr>
</table>

## Incorporating automated declines and holds

Certainly! Let’s derive the modified total cost formula for the card issuer step-by-step, incorporating automated declines, automated holds, and limited analyst capacity to review remaining flagged transactions.

***

## Goal:  
Calculate the **total cost of fraud to the issuer**, considering:

- Fraud detection model flags some transactions,
- Some flagged transactions are automatically declined ($$\alpha$$),
- Some flagged transactions are automatically held ($$\beta$$),
- Remaining flagged transactions are reviewed by analysts but capacity is limited,
- Costs arise from staffing analysts, false positives flagged among reviewed transactions, fraud missed (refunds), and potential costs from holds/declines.

***

## Step 1: Define Input Variables

- $N$: Total transactions per period  
- $f$: Fraudulent transaction rate (fraction)  
- $V$: Average transaction value (refund cost per missed fraud)  
- $C_{fp}$: Cost of false positive per incorrectly flagged legitimate transaction  
- $P$: Precision of fraud model among flagged transactions  
- $R$: Recall of fraud model (frauds detected among total frauds)  
- $n_a$: Number of cases reviewed per analyst per period  
- $S$: Analyst salary per period  
- $A$: Number of analysts employed  
- $\alpha$: Fraction of flagged transactions automatically declined (no review)  
- $\beta$: Fraction of flagged transactions automatically held (no immediate review)  
- $\gamma = 1 - \alpha - \beta$: Fraction of flagged transactions sent for analyst review  

***

## Step 2: Calculate Base Quantities

- Number of fraudulent transactions:  
$$
N_f = N \times f
$$

- True positives (frauds detected):  
$$
TP = R \times N_f
$$

- Number of flagged transactions:  
$$
N_{\text{flagged}} = \frac{TP}{P}
$$

***

## Step 3: Partition Flagged Transactions

- Automatically declined transactions:  
$$
N_{\text{declined}} = \alpha \times N_{\text{flagged}}
$$

- Automatically held transactions:  
$$
N_{\text{held}} = \beta \times N_{\text{flagged}}
$$

- Transactions routed to analysts for review:  
$$
N_{\text{reviewed\_needed}} = \gamma \times N_{\text{flagged}}
$$

***

## Step 4: Analyst Review Capacity

- Analyst capacity (max cases reviewed per period):  
$$
\text{Capacity} = A \times n_a
$$

- Actual transactions reviewed (limited by capacity):  
$$
N_{\text{reviewed}} = \min(N_{\text{reviewed\_needed}}, \text{Capacity})
$$

***

## Step 5: Fraud and False Positives in Reviewed Set

- Assuming the flagged transactions are uniformly sampled with respect to fraud and legit:

- True positives among reviewed:  
$$
TP_{\text{reviewed}} = \min(\gamma \times TP, P \times N_{\text{reviewed}})
$$

- False positives among reviewed:  
$$
FP_{\text{reviewed}} = N_{\text{reviewed}} - TP_{\text{reviewed}}
$$

***

## Step 6: Fraud Detected and Missed Overall

- Fraud detected by automated declines:  
$$
\text{Fraud}_{\text{auto detected}} = \alpha \times TP
$$

- Fraud detected by analyst review:  
$$
\text{Fraud}_{\text{review detected}} = TP_{\text{reviewed}}
$$

- Total fraud detected:  
$$
\text{Fraud}_{\text{detected}} = \text{Fraud}_{\text{auto detected}} + \text{Fraud}_{\text{review detected}}
$$

- Fraud missed (refunded):  
$$
FN = N_f - \text{Fraud}_{\text{detected}}
$$

***

## Step 7: Cost Components

- Analyst staffing cost:  
$$
\text{Staffing Cost} = A \times S
$$

- False positive cost (only for reviewed transactions):  
$$
\text{False Positive Cost} = FP_{\text{reviewed}} \times C_{fp}
$$

- Fraud refund cost (for missed fraud):  
$$
\text{Fraud Refund Cost} = FN \times V
$$

- (Other costs related to declines or holds may be added if estimated.)

***

## Step 8: Final Total Cost Formula

$$
\boxed{
\begin{aligned}
\text{Total Cost}_{\text{issuer}} = \quad & A \times S \\
& + FP_{\text{reviewed}} \times C_{fp} \\
& + FN \times V 
\end{aligned}
}
$$

Where:

$$
\begin{aligned}
& N_{\text{flagged}} = \frac{TP}{P}, \quad TP = R N_f \\
& N_{\text{reviewed}} = \min(\gamma N_{\text{flagged}}, A n_a) \\
& TP_{\text{reviewed}} = \min(\gamma TP, P \times N_{\text{reviewed}}) \\
& FP_{\text{reviewed}} = N_{\text{reviewed}} - TP_{\text{reviewed}} \\
& \text{Fraud}_{\text{detected}} = \alpha TP + TP_{\text{reviewed}} \\
& FN = N_f - \text{Fraud}_{\text{detected}}
\end{aligned}
$$

Or, expressed as a function of the input variables:

$$
\boxed{
\begin{aligned}
\text{Total Cost}_{\text{issuer}} = & \quad \underbrace{A \times S}_{\text{Staffing Cost}} \\
& + \underbrace{\left( \min\left( (1 - \alpha - \beta) \times \frac{R N f}{P},\, A n_a \right) - \min\left( (1 - \alpha - \beta) R N f,\, P \times \min\left( (1 - \alpha - \beta) \frac{R N f}{P},\, A n_a \right) \right) \right) \times C_{fp}}_{\text{False Positive Cost}} \\
& + \underbrace{\left( N f - \alpha R N f - \min\left( (1 - \alpha - \beta) R N f,\, P \times \min\left( (1 - \alpha - \beta) \frac{R N f}{P},\, A n_a \right) \right) \right) \times V}_{\text{Fraud Refund Cost}}
\end{aligned}
}
$$

***

This formula explicitly factors in automated declines and holds, analyst capacity constraints, and their impact on costs related to analyst staffing, false positives handled, and fraud missed.


## Fraud cost with a perfect fraud detection model

If we set **Precision $P = 1$** and **Recall $R = 1$**, this means:

- Every flagged transaction is truly fraudulent (no false positives),
- All fraudulent transactions are detected (no fraud missed).

Let's simplify the total cost formula under these assumptions.

***

### Recall the formula (without intermediate variables):

$$
\begin{aligned}
\text{Total Cost}_{\text{issuer}} = & \quad A \times S \\
& + \left( \min\left( (1 - \alpha - \beta) \times \frac{R N f}{P},\, A n_a \right) - \min\left( (1 - \alpha - \beta) R N f,\, P \times \min\left( (1 - \alpha - \beta) \frac{R N f}{P},\, A n_a \right) \right) \right) \times C_{fp} \\
& + \left( N f - \alpha R N f - \min\left( (1 - \alpha - \beta) R N f,\, P \times \min\left( (1 - \alpha - \beta) \frac{R N f}{P},\, A n_a \right) \right) \right) \times V
\end{aligned}
$$

***

### Substitute $P=1$ and $R=1$:

- $ \frac{R N f}{P} = \frac{1 \times N f}{1} = N f $,
- $ P \times \min(\cdots) = 1 \times \min(\cdots) = \min(\cdots) $,
- $ R N f = N f $.

Rewrite terms:

- $ \min((1 - \alpha - \beta) \times N f, A n_a) $,
- The difference inside the false positive cost term becomes:
  $$
  \min\left( (1 - \alpha - \beta) N f, A n_a \right) - \min\left( (1 - \alpha - \beta) N f, \min\left( (1 - \alpha - \beta) N f, A n_a \right) \right)
  $$
  
Since the second min is always less than or equal to the first min, their difference is zero:

$$
\min(X, A n_a) - \min(X, \min(X, A n_a)) = 0
$$

where $X = (1 - \alpha - \beta) N f$.

Thus, the **false positive cost term disappears** because with perfect precision, there are no false positives.

***

### Remaining formula becomes:

$$
\text{Total Cost}_{\text{issuer}} = A \times S + \left( N f - \alpha N f - \min\left( (1 - \alpha - \beta) N f, A n_a \right) \right) \times V
$$

Simplify inside parentheses:

$$
N f - \alpha N f = (1 - \alpha) N f
$$

So:

$$
\text{Total Cost}_{\text{issuer}} = A \times S + \left( (1 - \alpha) N f - \min\left( (1 - \alpha - \beta) N f, A n_a \right) \right) \times V
$$

***

### Interpretation:

- $A \times S$: Analyst salaries (fixed cost),
- $(1 - \alpha) N f$: Fraud not caught by automated declines,
- $\min((1 - \alpha - \beta) N f, A n_a)$: Fraud caught by analyst review (limited by capacity),
- The difference is fraud missed (refund cost),
- No false positive costs since $P=1$.

***

### Summary:

$$
\boxed{
\text{Total Cost}_{\text{issuer}} = A S + \left[ (1 - \alpha) N f - \min\left( (1 - \alpha - \beta) N f, A n_a \right) \right] V
}
$$

This shows cost depends on salary plus refunds for fraud missed after automated declines ($\alpha$) and holds ($\beta$) plus analyst review capacity constraints.
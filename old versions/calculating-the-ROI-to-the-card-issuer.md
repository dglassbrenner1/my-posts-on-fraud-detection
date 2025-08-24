# 1. Who pays what?

If the business in the LinkedIn fraud detection case study were the **card issuer** instead of the merchant, the calculation of ROI (Return on Investment) for fraud detection would differ in several ways because the card issuer’s costs, risks, and revenue impacts are different from those of the merchant.

### Key Differences in ROI Calculation for Card Issuer vs. Merchant

| Aspect                      | Merchant ROI Calculation                                         | Card Issuer ROI Calculation                                     |
|-----------------------------|-----------------------------------------------------------------|----------------------------------------------------------------|
| **Fraud Losses**             | Merchant loses the transaction amount plus chargeback fees.     | Issuer loses money through fraud payouts and reimbursements to merchants. |
| **Chargeback Fees & Costs**  | Merchant pays the chargeback fees (e.g., $25 per disputed transaction seen in the LinkedIn post). | Issuer does not pay chargeback fees but processes them; issuer may incur investigation, operational, and compliance costs. |
| **Revenue Impact**           | Merchant faces lost sale and operational costs in investigating fraud. | Issuer may face increased costs for claim processing, fraud investigative units, and possible regulatory fines if non-compliant. |
| **Recovery Opportunities**  | Merchant may recover some amounts by disputing chargebacks.      | Issuer may attempt recovery from merchants or cardholders but typically bears ultimate cost. |
| **Customer Relationship**   | Merchant risk losing customer loyalty due to false positives and declined transactions. | Issuer manages cardholder relationships and focuses on customer protection, trust, and retention costs. |
| **Operational Costs**        | Costs involve fraud analyst labor, chargeback management, and loss of sales opportunities. | Costs include fraud monitoring systems, customer service for disputes, regulatory compliance, and account protection. |
| **Fraud Detection Impact KPIs** | Focus on reducing fraud losses, false positives that reduce sales, and chargeback rates. | Focus on minimizing fraud payouts, managing dispute resolution costs, and reducing fraud volume on the card portfolio. |
| **ROI Focus**               | ROI = (Fraud losses prevented + chargeback fees saved + operational savings - fraud detection costs) / fraud detection costs | ROI = (Fraud payouts prevented + operational savings + improved customer retention - fraud detection costs) / fraud detection costs |

### Practical Example Differences

- **Merchant ROI Example:** Detecting fraud reduces lost sales and chargebacks, saves chargeback fees, and reduces investigation costs, directly impacting merchant revenue.
- **Issuer ROI Example:** Fraud detection aims to reduce payouts reimbursed to merchants, reduce chargeback dispute management costs, and avoid regulatory penalties, with additional benefits from maintaining cardholder satisfaction and loyalty.

### Card Issuer-Specific Considerations

- Issuers typically calculate ROI by modeling:
  - Fraud loss savings from prevented unauthorized transactions.
  - Savings from reduced operational costs of fraud investigations and dispute management.
  - Impact of false positives on customer experience and potential churn.
  - Regulatory compliance costs or fines avoided.

- ROI metrics for issuers often leverage data like total fraud losses as a percentage of total transaction volume, fraud detection effectiveness, and cost per fraud claim processed.

***

**In summary:**  
If the business is the card issuer, ROI moves from direct sales and chargeback fee savings (merchant perspective) toward fraud payout reduction, dispute management savings, and customer retention value (issuer perspective). The cost components and revenue impacts differ, so ROI calculations incorporate issuer-specific fraud loss and operational cost structures rather than merchant-specific sales and chargeback fee considerations.

Let me know if you want, I can provide an ROI formula or case example tailored for card issuers as well.

[1](https://www.sardine.ai/blog/revenue-attach-rate)
[2](https://www.inscribe.ai/fraud-detection/fraud-detection-roi)
[3](https://www.datavisor.com/blog/how-much-application-fraud-can-cost-banks-downloadable-roi-calculator)
[4](https://www.datavisor.com/blog/how-to-calculate-the-roi-of-transaction-fraud-prevention)
[5](https://withpersona.com/blog/speak-roi-fraud-prevention)
[6](https://ekata.com/identity-roi-fraud-calculator/)
[7](https://chargebacks911.com/fraud-scoring/)
[8](https://www.rippleshot.com/post/how-to-calculate-the-roi-of-your-decision-rules)
[9](https://www.about-fraud.com/cracking-the-fraud-roi-maturity-curve/)
[10](https://risk.jaywing.com/news-views/roi-ai-in-risk/)

# 2. Calculating the ROI to the card issuer of a ML model 

I'm basing figures on this LinkedIn post: https://www.linkedin.com/pulse/from-ai-metrics-business-impact-fraud-detection-case-study-singh-tzsfe/ [^1]



***

### Given:
- Total transactions per month $$N = 1,000,000$$
- Fraudulent transaction rate $$f = 0.002$$ (0.2%)
- Number of fraudulent transactions $$N_f = N \times f = 2,000$$
- Average transaction value $$V = 100$$
- Cost to review a flagged transaction $$C_r = 5$$
- Cost of a chargeback $$C_c = 25$$
- Cost of incorrectly flagged legitimate transaction $$C_{fp} = 10$$
- Model precision $$P = 0.8$$
- Model recall $$R = 0.3$$
- Accuracy $$A = 0.97$$ (informative but less used here)

***

### Step 1: Calculate number of True Positives (TP), False Positives (FP), etc.

- $$TP = R \times N_f = 0.3 \times 2,000 = 600$$
- Number of flagged transactions $$N_{\text{flagged}} = \frac{TP}{P} = \frac{600}{0.8} = 750$$  
  (because Precision $$P = \frac{TP}{TP+FP}$$)
- False Positives $$FP = N_{\text{flagged}} - TP = 750 - 600 = 150$$

***

### Step 2: Costs and savings

- **Cost of reviewing flagged transactions:**  
  $$C_{review} = N_{\text{flagged}} \times C_r = 750 \times 5 = 3750$$

- **Cost due to false positives (customer dissatisfaction, lost sales):**  
  $$C_{fp\_cost} = FP \times C_{fp} = 150 \times 10 = 1500$$

- **Total fraud loss without model:**  
  $$L_{\text{no model}} = N_f \times V = 2000 \times 100 = 200,000$$

- **Fraud loss with model (frauds missed × value + chargebacks for missed):**  
  Frauds missed = $$N_f - TP = 2000 - 600 = 1400$$  
  Since these are undetected frauds, issuer pays the full amount plus chargebacks. The chargeback cost is embedded in total loss or assumed separate. For clarity:  
  $$L_{\text{with model}} = 1400 \times (V + C_c) = 1400 \times (100 + 25) = 175,000$$

- **Fraud loss saved by model:**  
  $$L_{\text{saved}} = L_{\text{no model}} - L_{\text{with model}} = 200,000 - 175,000 = 25,000$$

***

### Step 3: Net savings

Net savings to issuer from ML model considering detection costs:

$$
\text{Net Savings} = L_{\text{saved}} - (C_{review} + C_{fp\_cost})
= 25,000 - (3,750 + 1,500) = 19,750
$$

***

### Summary:

| Quantity                             | Value            |
|------------------------------------|------------------|
| True Positives (detected frauds)   | 600              |
| False Positives                    | 150              |
| Flagged transactions               | 750              |
| Review cost (flagged × $5)          | $3,750           |
| Cost from false positives          | $1,500           |
| Fraud losses without ML            | $200,000         |
| Fraud losses with ML               | $175,000         |
| Fraud loss saved (benefit)         | $25,000          |
| Net savings after detection costs | $19,750          |

***

This quantifies the issuer's ROI considering fraud detection effectiveness, operational costs, and the business context. You could calculate ROI as:

$$
ROI = \frac{\text{Net Savings}}{\text{Detection Costs}} = \frac{19,750}{3,750 + 1,500} \approx 3.3
$$

meaning every dollar spent on detection costs returns about $3.30 in net benefits.

## Formula for net savings to the card issuer

Here is the formula for the **net savings to the card issuer** as a function of the input variables:

- $N$: Total transactions per period  
- $f$: Fraudulent transaction rate (fraction)  
- $V$: Average transaction value  
- $C_r$: Cost of reviewing a flagged transaction  
- $C_c$: Cost of a chargeback  
- $C_{fp}$: Cost of an incorrectly flagged legitimate transaction (false positive cost)  
- $P$: Precision of the fraud detection model  
- $R$: Recall of the fraud detection model  

***

$$
\text{Net Savings} = \left[ N f V - (N f - R N f)(V + C_c) \right] - \left[ \frac{R N f}{P} C_r + R N f \left(\frac{1}{P} - 1\right) C_{fp} \right]
$$

***

This formula expresses net savings in terms of model performance metrics and business parameters.

[^1]: Opus Technologies. (2024, October 28). From AI metrics to business impact: A fraud detection case study [LinkedIn post]. LinkedIn. https://www.linkedin.com/posts
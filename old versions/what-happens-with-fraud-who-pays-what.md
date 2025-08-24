Suppose a fraudster gets my credit card # and makes a purchase for $100.  Walk me through what happens (including who pays how much to whom), depending on whether the fraud is suspected, confirmed, or never suspected, and by whom (me, the credit card issuer, or the merchant).  Assume the merchant's bank charges a "chargeback" fee to cover the costs of refunding the fraudulent transaction to the card issue.


| Scenario                                    | You (Cardholder)                                          | Card Issuer                                  | Merchant                             |
|---------------------------------------------|-----------------------------------------------------------|----------------------------------------------|------------------------------------|
| Fraud never suspected (by anyone) | Pays $100                                             | No loss                                     | Receives $100                     |
| Card issuer blocks suspicious transaction (declines before completion) | No charge                                            | Pays operational fraud detection costs      | No payment received               |
| I report the fraud. Issuer investigates and confirms the fraud | No charge                                            | Pays investigation cost, recovers $100 from merchant         | Pays $125 ($100 to issuer, $25 to the merchant's bank)    |
| I report what I think is fraud, but I'm wrong (e.g., Friendly Fraud where I don’t recognize legitimate charge). Issuer investigates and finds the charge legitimate.  | Pay $100 (charged after dispute reversal)           | Pays investigation cost                       | Receives $100      |
| I never report fraud but the issuer detects fraud after charge | No charge                                            | Pays investigation cost, recovers $100 from merchant | Pays $125 ($100 to issuer, $25 to the merchant's bank)     |

***

## A note on chargebacks

If my credit card is a Visa card with First Person Bank and the merchant puts their revenue in First Merchant Bank, who issues the chargeback fee and who receives it?

For a Visa credit card with your card issued by **First Person Bank** (the Issuer) and the merchant’s funds deposited via **First Merchant Bank** (the Acquirer), here is how the chargeback fee is handled:

- The **chargeback fee is imposed by the Acquirer** (First Merchant Bank) on the merchant. This fee is intended to cover costs related to processing chargebacks, investigations, and administrative overhead.

- The **merchant pays the chargeback fee to their Acquirer bank** (First Merchant Bank), not directly to the Issuer or Visa.

- The **Issuer (First Person Bank) recovers the disputed transaction amount** (e.g., $100) from the Acquirer bank through the Visa network during the chargeback process.

- Visa sets network rules and may charge fees to Acquirers or Issuers for dispute management, but the direct fee that merchants pay is administered by their Acquirer bank.

### Summary flow:

| Party               | Role                                     | Payment/Chargeback Fee Interaction                     |
|---------------------|------------------------------------------|--------------------------------------------------------|
| Cardholder          | Holds Visa card issued by First Person Bank | Reports fraud, protected from fraudulent charges       |
| Issuer (First Person Bank) | Issues Visa card, manages account and fraud detection | Recovers transaction amount ($100) through Visa from Acquirer |
| Acquirer (First Merchant Bank) | Merchant’s bank, processes payments, handles chargebacks | Charges the merchant fee (e.g., $25) for chargeback processing |
| Merchant            | Receives payments from Acquirer          | Pays chargeback fee to Acquirer and loses transaction amount on chargebacks |

***

In short, **the chargeback fee comes from the Acquirer (merchant’s bank) and is paid by the merchant to this bank**. The Issuer recovers the disputed funds via Visa but does not charge the merchant fee directly.

This is consistent with Visa’s dispute and fee structure as managed through Issuer and Acquirer banks within the Visa payment network.

[1](https://chargebacks911.com/visa-chargebacks/)
[2](https://justt.ai/blog/merchant-chargeback-fee/)
[3](https://www.signifyd.com/blog/stripe-visa-chargeback-changes/)
[4](https://chargebacks911.com/visa-rule-changes-april-2025/)
[5](https://justt.ai/blog/visa-chargeback-fee-changes-april-2025/)
[6](https://chargebackhelp.com/what-is-a-chargeback-fee/)
[7](https://www.chargeflow.io/blog/understanding-chargeback-fees-and-costs-a-merchants-guide)
[8](https://usa.visa.com/support/small-business/regulations-fees.html)
[9](https://www.chargebackgurus.com/blog/visas-new-dispute-fee-changes)
[10](https://usa.visa.com/dam/VCOM/download/about-visa/visa-rules-public.pdf)
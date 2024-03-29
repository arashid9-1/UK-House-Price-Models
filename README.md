# UK-House-Price-Models
Repeat sales models coded in Python: Gao and Wang’s (2007) Unbalanced Panel and an Autoregressive Mixed-Effects model adapted from Nagaraja et.al (2011)

***

### Abstract

The following work was requested and supervised by Fathom Consulting to improve the quality of their in-house UK house pricing model: Gao and Wang’s (2007) Unbalanced Panel. The model however is oversimplistic, mis specifying the error structure and suffering
from sample selection bias. We form our own variant of Nagaraja et al. (2011) model, coined
the Autoregressive Mixed Effects Model. The model corrects the inefficiencies of the
Unbalanced Panel method and is less prone to selection bias. The two models, UP and
ARME, were trained and tested on the [HM Land Registry Price Paid dataset](http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.txt). The ARME
model outperformed the UP model in root mean squared error. Our results also suggest that
the ARME model is more representative of the overall UK housing market. The [full report]() goes into further depth, detailing the full methodology, data descriptives and model diagnostics. 
***
### Repeat Sales Models vs Hedonic Models

Repeat Sales and Hedonic models are the primary methods in constructing house price indices. Hedonic models predict house prices from their observed characteristics (number of bedrooms, distance to nearest bus stop, air quality ... ) and a set of fixed time effects. The coefficients on the characteristic variables act as shadow prices, they represent the change in house price for a marginal change in characteristic. The time fixed effects are used to model the trend in sales prices over time. The major drawback to these models is the difficulty in obtaining property characterises data. Conversely, repeat sales models only require property sale price data, constructing an index by linear regression using the log price difference of two successive sales of a property. The earlier sale acts as a proxy for hedonic information since the previous price gives some indication of the home’s quality. This is an advantage of repeat sales methods because instead of carefully choosing explanatory variables, the hedonic information can be captured by the previous sale.
***

### Unbalanced Panel Model (UP)

Essentially, the model proposes that property value at a given time is a function of two factors, its "intrinsic value" and a time trend. Suppose a property was built in 1996 and subsequently exchanged hands in 2005 and 2015. Using OLS, a simple time trend can be found from the change in sale prices over time. The estimated coefficient can then deflate the house price to 1996 levels to estimate the intrinsic value of the house. 

$$ \ln{P_{i,t}} = A\tau_i + M\beta_t + \epsilon_{i,t} $$

$$ t = 1... T, i = 1... N $$

To construct price indices for 9720 UK postcode sectors the above is estimated via pooled OLS using all property sales from homes that have sold at least twice in the sector over the full period. There are T time periods (months from 1996 to 2023 in this case) and N repeat-sales homes in the sector. A and M are house specific fixed effects and sector time trend fixed effects respectively, with $\tau$ and $\epsilon$ the estimated coefficients. 

The index is formed by multiplying the intrinsic value of the property with exponential market index value for every period. 

$$ \beta = \left( \bar{A}e^{\beta_1}, ... ,\bar{A}e^{\beta_T}  \right) $$

***
### Autoregressive Mixed Effects Model (ARME)

This model incorporates three key changes:

1. Model House-specific effects as random and not fixed 
   
   When house-specific effects are independent of the other regressors it is more statistically efficient to use random effects. Additionally, fixed effects likely misspecify the variable since house quality is likely to change over time due to depreciation and refurbishment. 

2. Incorporate error heteroscedasticity and serial correlation 
   
   The white noise process of the UP error term suggests that exogenous shocks to house prices, such as house deterioration, have a permanent effect on future valuations of the property. Data and previous research ([see report for further details]()) imply a house price shock to one property pushes its valuation away from average house price for the region, however is mean reverting in the future. As such, a previous sale of a property contains less information regarding the current value of the house than the mean sector house price the further the gap between sales. Hence, error autoregression is modelled by $\phi^\gamma$, where $\gamma$ is the gap time between sales in years. 

3. Use single sales in estimation process
   
   Repeat sales estimates likely suffer from sample selection bias since those homes are different from properties that had only transferred hands once over the period. For example, starter homes in large cities are likely to sell repeatedly unlike a countryside mansion. Including single sales into the estimation should help mitigate estimator bias. 

A explanation of model specification and implementation can be found in the report and the supplied [code]().
***
### Performance Results 

The dataset had a 85:15 training-testing split, using only repeat-sales homes. As such, a repeat-sales only ARME variant, ARME-RS, was also tested to remove the inherent advantage given to the UP model. 

|                | UP         | ARME       | ARME - RS  |
|----------------|------------|------------|------------|
| ME             | 3991       | -3147      | -1228      |
|                | (0.97%)    | (-0.20%)   | (0.08%)    |
| MAE            | 17802      | 18006      | 17602      |
|                | (6.63%)    | (6.75%)    | (6.55%)    |
| RMSE           | 68160      | 54136      | 53994      |
|                | (11.9%)    | (11.1%)    | (11.1%)    |
| Note: metric given in terms of pound sterling | | |

Mean Error:

ME measures estimator bias. The ARME model has a slightly lower ME in magnitude, showing a slight reduction in estimator bias

Mean Absolute Error:

Measures average magnitude of prediction error. The UP model beats the ARME model here. However, when the inherent advantage of using only repeat sales properties is removed, the ARME-RS model is superior 

Root Mean Squared Error:

Measures estimator variance. The improvement of the ARME models shows that changing the error specification improved estimator efficiency. 

***

### Concluding Remarks

The ARME models outperformed the UP model in ME, exhibiting less bias, prediction accuracy and estimator variance. Hence, the ARME is ultimately superior for price estimation and overall UK-wide representation. 

***

### Contributers

Eloise Morrison-Clare (Co-Author - University of Bath)

Andrew Brigden (Project Supervisor - Fathom Consulting)

Dr Chaitra Nagaraja (Supplied auxiliary code, Co-Author of [_An Autoregressive Approach To House
Price Modelling (2011)_](https://arxiv.org/abs/1104.2719))


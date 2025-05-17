---
layout: post
title: "How to encode categorical variables and interpret them?"
date: 2025-04-01
excerpt: "We are interested in encoding categorical variables, because machine learning models work best with numerical data rather than text. Additionally, by encoding the categories into equal weights, we prevent introducing bias in the model."
permalink: /blog/categorical-variables-encoding/
tags: [blog, statistics, fundamentals]
pinned: true
---

# How do you encode categorical variables?

There are two types of categorical data:

* Ordinal data
* Nominal data

 **Ordinal data** has *inherent* order (i.e., data points can be ranked and there is some meaningful differences in the ranking). Test scores such as A+, A, A- can be ordered. In contrast, **nominal data** does not have inherent order, such as names of places.

We are interested in encoding categorical variables, because machine learning models work best with numerical data rather than text. Additionally, by encoding the categories into equal weights, we prevent introducing bias in the model.

## Dummy encoding
To demonstrate each encoding strategies, suppose you have the following dataset

ID| Age | Weight | Smoker | Place of Birth| Heart risk
---|---|---|---|---|---
1|30|75 | No | Japan| Low
2|20 | 70 | Yes | Vietnam | High
3| 70 | 60 | No | UK | Low

Suppose you are predicting the heart risk of an individual using only age, weight and smoker status, you may potentially want to fit a logistic regression.

Here, you can use *dummy encoding* to encode smoker status. If `smoker = No` is your reference, you can assign value of 0 to No, and 1 to Yes. So the resulting data set may look like this:

ID| Age | Weight | Smoker=Yes | Place of Birth| Heart risk|
---|---|---|---|---|---|
1|30|75 | 0 | Japan| Low|
2|20 | 70 | 1 | Vietnam | High|
3| 70 | 60 | 0 | UK | Low|

and your resulting regression may look like this:

$$Risk = \beta_0 + \beta_1\cdot Age + \beta_2\cdot Weight + \beta_3\cdot Smoker_{yes}
$$


 The summary statistic output of your regression in R may be as follows:

| Covariate | Coefficient Estimate | P-value |
|---|---|---|
|Intercept| -1.4 | 0.95 | 
|Age| 1.3 | 0.04 | 
|Weight| 2.3 | 0.01 | 
|Smoker=Yes| 1.3 | 0.03 |

In this fictitious example, if $\beta$ is the *average change in log odds of response variable* [[1]](#1), then $e^\beta$ is the *average change in odds of response variable*. So, in our case, 
> if all other covariates are kept constant, on average smokers have $e^{1.3}=3.67$ higher odds of having heart problems than non-smokers.

Suppose you have more than two categories in your column, such as the `place of birth`. If you followed the example above, you could select one of the place as your reference (e.g., UK) and convert your data as follows to predict the heart risk based on place of birth:

ID| PoB=Japan | PoB=Vietnam | Heart risk|
---|---|---|---|
1|1| 0 | Low|
2|0 | 1 | High|
3| 0 | 0 | Low|

The associated summary statistics may be as follows:


| Covariate | Coefficient Estimate | P-value |
|---|---|---|
|Intercept| -1.4 | 0.95 | 
|PoB=Japan| 2.3 | 0.04 | 
|PoB=Vietnam| 1.3 | 0.01 | 

Similar to above, the interpretation of the coefficients will be relative to the reference,
>Person born in Japan on average will have $e^{2.3}=9.97$ higher odds of having heart problems compared with a person born in the UK.

## One Hot Encoding

In the table above, we have ommitted a column `PoB = UK`. This is because of the multicolinearity problem [[2]](#2). If we were using linear regression, there would be more than 1 unique solutions. However, this is not the problem when using neural networks, decision trees or any model that does not have the assumption of non-multicolinearity.

## Ordinal Encoding

If your categorical column has some inherent order, you may consider using ordinal encoding. This is fairly straightforward in that your data is converted to numerical values that preserve the ranking of the data points.

The interpretation of the coefficients is similar to other continuous variables, such that a change in one unit causes a change in the dependent variable equal to the coefficient.

## Label Encoding

The main disadvantage of One-Hot encoding is that it may introduce many extra columns. Label encoding is the type an interger encoding that convert each categorical value to a unique integer. The main flow of this encoding scheme is that it may inadvertently introduce ordinality in the dataset where there is no such relationship. According to the `sklearn` documentation,
>the `LabelEncoder` must only be used to encode target values, i.e. `y`, and not the input `x`.[[3]](#3)

## Frequency Encoding

Instead of arbitrarily assigning numbers to categorical values, one strategy is to convert the categorical values based on how many times they are observed in the dataset. For example,

City | Frequency Encoding (Occurences)
---|---
New York | 50 000
Los Angeles | 30 000
Chicago | 10 000

If we used frequency coding in a linear regression model to predict revenue

$$ Revenue = \beta_0 + \beta_1 \cdot Frequency(City)
$$

We can interpret the coefficient $\beta_1$ as follows,
- If $\beta_1 = 0$, there is no effect on the revenue due to the city frequency.
- If $\beta_1 > 0$, cities with more occurences will contribute more to the revenue.
- If $\beta_1 < 0$, cities with more occurences will contribute less to the revenue.

## Target Encoding

Alternatively, we can assign the categorical values using the target values, such as the mean of revenue in each city. 

City | Frequency Encoding (Occurences) | Mean revenue
---|--- | ---
New York | 50 000 | 1000
Los Angeles | 30 000 | 2000
Chicago | 10 000 | 500

We can use target encoding when there is likely relationship between category and the target variable. However, we should not use it to perform classification, as it can lead to data leakage.


## Summary

Below is the summary of several encoding methods [[4]](#4)

Encoding technique | Advantage | Disadvantage
---|--- | ---
Label Encoding | - Easy to implement  | - May introduce arbitrary ordinality 
One hot encoding | - Suitable for nominal data <br> - Does not introduce ordinality | - May not be suitable for large number of features
Ordinal encoding | - Preserve the order of the categories| - The spacing between orders are equal, which may not always be the case
Target encoding | - Can improve model performance by incorporating target information | - May introduce overfitting with small datasets.

## References

<a id="1"></a> [1] [https://www.statology.org/interpret-logistic-regression-coefficients/](https://www.statology.org/interpret-logistic-regression-coefficients/)

<a id="2"></a> [2] [https://datascience.stackexchange.com/questions/98172/what-is-the-difference-between-one-hot-and-dummy-encoding](https://datascience.stackexchange.com/questions/98172/what-is-the-difference-between-one-hot-and-dummy-encoding)

<a id="3"></a> [3] [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

<a id="4"></a> [4] [https://www.geeksforgeeks.org/encoding-categorical-data-in-sklearn/](https://www.geeksforgeeks.org/encoding-categorical-data-in-sklearn/)
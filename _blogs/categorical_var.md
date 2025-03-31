---
layout: post
title: "How to encode categorical variables and interpret them?"
date: 2025-03-19
excerpt: "As the appetite for black box models increases, so is our lack of clear understanding of the "
permalink: /blogs/categorical-variables-encoding/
tags: [statistics, fundamentals]
---

# Introduction

*How do you encode categorical variables?*

This was a question I was posed in a recent data science interview. I was stumped, after talking an hour about LLMs, my mind drew blanks when being asked such a simple question.

"*Well you obviously can do One Hot Encoding, where each of the categorical variables is encoded as 1 or 0...and there is a Label Encoding, where the categories are encoded as 1,2,3...*", I said.
"*The disadvantage of One-Hot Encoding is that when you have a lot of variables, you will have to create a very large number of features. I think you would use Label Encoding when you have less number of categories*".

"*Well, if we were to encode the cities (e.g., NY, London, Hanoi) with Label Encoding, how would you intepret if one city is encoded as 1 and the other as 3?*", as soon as the interviewer said this, I knew I screwed up. Never having worked with dataset beyond two categories (e.g., Male vs. Female), for the life of me I could not think of how to explain regression models where there are multi-class categories. In fact, I realised I have always relied on the statistical packages provided and have not trully understood the reasoning behind it. 

# How do you encode categorical variables?

There are two types of categorical data:
* Ordinal data
* Nominal data

**Ordinal data** are those where there are *inherent* order (i.e., they can be ranked and there is some meaningful differences in the ranking). For example, the grade scores such as A, B, C, D can be ordered.
**Nominal data** are those where there are no inherent order, such as names of places.

We are interested in encoding categorical variables, because machine learning models work best with numerical data rather than text. Additionally, by encoding the categories into equal weights, we prevent introducing bias in the model.

## One Hot Encoding

This type of encoding is easy to understand.

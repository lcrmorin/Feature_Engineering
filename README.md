# Opinionated Guide on Feature Engineering for tabular data

A collection of notes on features engineering for tabular data. It uses [Feature Engineering and Selection: A Practical Approach for Predictive Models](https://www.amazon.com/Feature-Engineering-Selection-Practical-Predictive/dp/1138079227) as a Guidelines. The general idea is to list FE ideas, provide an opinion with aims to improve the performance of a gbdt model. It doesn't make an assumption on which flavor of gbdt is used (but you probably need to start with lightgbm). 

# Data exploration

First step is data exploration (exploratory data analysis or EDA). It will help understand what is wrong with your data and what needs to be done. Often subjective and domain specific, but some good guidelines:
- distributions and extremes values (pandas .describe())
- look at plots (matplotlib hist(bins=100))
- what are the main 1,2,3 features that drives the target ?
- what are the main categories (time, id, groups) ?
- special focus on targets (distribution, plots, interactions with main features and categories)
- overused but optional: features correlations, features correlation with target. Usefull to understand the problem. But action taken a priori rarely lead to improvement on performance. 

# Problem design

Validation design is a problem in itself, especially in the context of feature engineering. As you want to perform multiple actions and ideally tests their impact independently you would need multiple staged population. It can get really difficult when you have a time component of the problem and/or when the dataset is small. So I often fallback to simpler data splits: oly perform train/tests splits and train everything on the same datasets. This approach is prone to overfitting but you can check this at once on the tests. The main drawback is that if you have overfitting you can't easily know which part is at fault. 

# Feature engineering

There is actually two kinds of feature engineering:
- Simple feature engineering (row-wise or columns-wise). While it might helps the model learn it will not lead to huge increase in model performance. It might serves some other purposes (explainability). Generally speaking gbdt are very performant. Notably tree models do not benefit from basic columns wise FE (scaling, monotonic transformation). Row-wise FE (descriptive statistics, basic aggregation) may help the model a little bit by learning more complex patterns.
- More complex feature engineering (rules of thumbs: FE involving multiple rows). By providing the model information with respect to other instance these technics are better candidates to improve the model.

# Simple Feature engineering - (columns-wise)

- Encoding categorical predictors: mostly don't. Moderns gbdts are now able to handle categoricals features natively.
- Tranforming continuous predictors: while some requirements are valids for NN, most of the usual transformations do not matter for tree models. Notable exception is when you start to mix models or enable linear_trees approaches. 
- Missing values / Extreme values / Sentinels values: modern gbdt can handle those. you can encode some of these as 0-1 (notably for explainability) but it shouldn't bring too much performance gain.

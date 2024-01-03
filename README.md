# Opinionated Guide on Data / Feature Engineering for Performance-led Modelling on Tabular Data

A collection of opinions on features engineering for tabular data. It uses [Feature Engineering and Selection: A Practical Approach for Predictive Models](https://www.amazon.com/Feature-Engineering-Selection-Practical-Predictive/dp/1138079227) as a Guideline. The general idea is to list FE ideas, provide an opinion with aims to improve the performance of a gbdt model. It doesn't make an assumption on which flavor of gbdt is used (but you probably need to start with lightgbm). It also takes into account some efficiency constraint (not too many features) and explainability constraint (again, not too many features). If you are purely performance led (à la Kaggle), the best approach is to do all the FE below and dump it into a big ensemble of models.

# Data exploration

The first step is data exploration (exploratory data analysis or EDA). It will help understand what is wrong with your data and what needs to be done to improve performance of the model. Often subjective and domain specific; Here are some basics:
- distributions / extremes values (pandas.describe()) AND look at plots (Matplotlib hist(bins=100))
- what are the main 1,2,3 features that drives the target ?
- what are the main categories (time, ID, groups) ?
- in that matter, target can be treated as another columns (all of the above should be performed distribution, plots, interactions with main features and categories)
- Optional: features correlations, including with targets. Nice plots, useful to understand the problem, but corrective actions taken a priori rarely lead to improvement in performance. 

# Data Quality

- The question of initial data quality is often overlooked as the data scientist come to a relatively clean dataset. But in practices, the data quality, handled at the Data Engineering / Machine Learning Engineering level, is often detrimental both for training and inference. If available, it is best to rely on available documentation. 

# Additional data

- Additional Features: as you'll see below, basic interactions are not difficult to learn for gbdts. However, it is often a good idea to loop through all features and look at individual performance of features of the form f1+f2, f1-f2, f1*f2, f1/f2, and add most performant ones. It might help you understand the problem / help the efficiency of the model.
- Additional Data: yes, that is a no-brainer, but you might want to think about getting more data before going into complex FE. Both in terms of additional instances (rows) and additional features (columns). You might have to deal with trade-off between quantity and quality.

 # Target engineering

 - All of the above data quality concerns apply to targets.
 - Additionally, make sure that the loss and metric are in line with your goal and the target

# Problem design

Validation design is a problem in itself, especially in the context of feature engineering. As you want to perform multiple actions and ideally tests their impact independently, you would need multiple staged sets. It can get really challenging when the problem has a time component and/or when the dataset is small. One option is to fall back to a simpler split: only perform train/tests splits and train everything on the same dataset. This approach is prone to overfitting, but you can check this at once on the test set. The main drawback is that if you have an overfitting problem, you can't easily know which part is at fault. 

# Feature selection

- As a general rule of thumb, it is a bad idea to perform a priori feature selection
- Most of the feature selection should be done through the tuning of the L1 regularisation parameter (let the model decide which feature to use)
- It is often a good idea to include pure noise features to see how your pipeline / model handle them
- Conversely, it is often interesting to test your pipeline / design with purely informative features (leaking the targets), to check if the pipeline is not too harsh.
- You can perform some a posteriori feature selection. It is not ideal, as it doesn't check each feature individually - and rely on correlation, not causality-, but iteratively removing the least 
10-20% important features might help.

# Feature engineering

There are generally two kinds of feature engineering:
- Simple feature engineering (row-wise or columns-wise). While it might help the model learn, it will not lead to huge increase in model performance. It might serve some other purposes (explainability). Generally speaking, gbdt are very performant. Notably, tree models do not benefit from basic columns-wise FE (scaling, monotonic transformation). Row-wise FE (descriptive statistics, basic aggregation) may help the model a little bit by learning more complex patterns.
- More complex feature engineering (rules of thumbs: FE involving multiple rows). By providing the model information with respect to other instances, these techniques are better candidates to improve the model.

## Simple Feature engineering - (columns-wise)

- Encoding categorical predictors: mostly don't. Moderns gbdts are now able to handle categorical features natively.
- Transforming continuous predictors: while some distributions requirements are valid for NN, most of the usual transformations do not matter for tree models. A notable exception is when you start to enable linear_trees approaches. 
- Missing values / Extreme values / Sentinels values: modern gbdt can handle those. You can encode some of these as 0-1 (notably for explainability) but it shouldn't bring too much performance gain.
- Some supervised transformation techniques have to be mentioned (target encoding...) but rarely works in practice. 

## Simple Feature engineering - (rows-wise)

- Gbdt models are very performant on their owns, so they should be able to learn complex patterns.  However, they are usually limited on the amount of features they can consider at once. Typically, a tree with max depth = 3 will only have 7–8 splits. So it will have a hard time learning patterns that involve more than that features. If you have hints that, for example, average of all features might play a role, you are better using it directly. Performance gain will usually be limited, but efficiency / explainability of the model might greatly increase.
- If you are really chasing small bits of performance, the way to go is to define a list of aggregation function and subcategories of columns. So that you can get an exhaustive approach of building each features for each groups of columns.

## Complex Feature engineering - (multiple columns)

- PCA helps in the same fashion: it can extract complex information from a lot of noisy columns. The big risk is that it works in an unsupervised manner: using PCA to drop columns might drop non-dominant but informative features. Most of the time, you should just append principal components to the original datasets. (Also don't forget scaling)

## Complex Feature engineering - (multiple rows)

- Those techniques are the most important candidates to improve your model. It is quite unlikely that your instances are independent. So you want to perform operations that will share some information between rows. The general way to do this it to perform some grouping or embedding, then to perform FE on those, conveying more global information into the features:
- Build clusters (k-means, hdbscan), then add features of the clusters, features relatives to the clusters (intra-clusters ranks, distance to centroid). The clusters can also be natural groups that appear in the data. 
- Build neighborhoods (k-nns), then add features of the neighborhoods, or relative to the neighborhoods.
- These clusters / neighborhoods can be built on the original data or on specific embedding (PCA, umap) or on selections of categories of columns.

## Domain specific feature engineering

- Some specific problems (time-series, text, CV) can be dealt trough gbdts, but will require domains specific FE. Time-series problems can be dealt with lagged features. Text and CV will require you to be very creative. 



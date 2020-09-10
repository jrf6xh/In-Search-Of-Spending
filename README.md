# In Search of Spending
**Predicting Revenue in Online Shopping**

## Overview
**Goals:**
* Predict how much a given vustomer will spend in a visit to the online store.
* Determine which factors have the greatest influence on consumer spending.

The idea behind this project is to predict how much a given customer will spend when visiting an online store. Once this can be predicted, we can provide personalized marketing to incentivize larger purchases

For example, if we think a customer will spend 150 USD in the store, we can send them a coupon for rewards points or free shipping if they spend 200 USD. Additionally, if the model predicts a customer will not make a purchase, they can be sent a coupon for 10% off if they make a purchase today. This personalized marketing could increase revenue for the company in question.

Additionally, our model will allow us to pinpoint which factors influence spending. Resources can then be deployed to improve features that deter purchases.

**Links:**
* [High Level Project Overview](https://github.com/jrf6xh/Capstone-Revenue_Prediction/blob/master/Presentation.pdf)
* [Code & Detailed Walkthrough](https://github.com/jrf6xh/Capstone-Revenue_Prediction/blob/master/Notebooks/Technical_Notebook.ipynb)


## Readme Navigation
[Overview](https://github.com/jrf6xh/Capstone-Revenue_Prediction#overview) -
[Data](https://github.com/jrf6xh/Capstone-Revenue_Prediction#data) -
[Data Cleaning](https://github.com/jrf6xh/Capstone-Revenue_Prediction#data-cleaning) -
[Modeling](https://github.com/jrf6xh/Capstone-Revenue_Prediction#modeling) -
[Model Results](https://github.com/jrf6xh/Capstone-Revenue_Prediction#model-results) -
[Limitations](https://github.com/jrf6xh/Capstone-Revenue_Prediction#limitations) -
[Future Improvements](https://github.com/jrf6xh/Capstone-Revenue_Prediction#future-improvements) -
[Reproduction Instructions](https://github.com/jrf6xh/Capstone-Revenue_Prediction#reproduction-instructions) -
[Sources](https://github.com/jrf6xh/Capstone-Revenue_Prediction#sources) -
[Project Information](https://github.com/jrf6xh/Capstone-Revenue_Prediction#project-information)

## Data
The data used in this project relates to Google's online store.  Each row in the dataset represents a visit to the store.  The data is available through [Kaggle](https://www.kaggle.com/c/ga-customer-revenue-prediction/data).

**Data Summary:**
* Dependent variable - revenue in USD.
* 717k rows/visits to the Google Store.
* Data collected between 2016-2018.
* 2.46% of visits result in a purchase.
* Average purchase size is 124 USD.

**Independent Variable Types:**
* Geographical
* Device
* Traffic Source
* Page Views
* Time
* Price

**Purchase Breakdown:**
Most purchases are under 100 USD, though there are outlier purchases that range up to 23k USD.  Average revenue from a purchase is 124 USD.

![](https://github.com/jrf6xh/Capstone-Revenue_Prediction/blob/master/Images/revenue_hist.png)

**Channel Grouping:**
Customers that get to the Google store through banner and other display advertisements spend a larger amount than customers that get to the store in any other way.

Customers that go to the store directly or are referred also spend a large amount.

Customers that get to the store through social media or affiliate marketing tend to spend less on average.

![](https://github.com/jrf6xh/Capstone-Revenue_Prediction/blob/master/Images/avg_rev_channel.png)

Although 'Display' customers spend large amounts if they make a purchase, they are not as likely as other customers to make a purchase in the first place.

'Direct' and 'Referral' customers, on the other hand, are more likely than other groups of visitors to actually make a purchase.

'Social Media' and 'Affiliate' visitors are the least likely to make a purchase.

![](https://github.com/jrf6xh/Capstone-Revenue_Prediction/blob/master/Images/purchase_percent_channel.png)

**Number of Visits to the Store:**
Customers that have visited the store before are more likely to make a purchase than those who are visiting the store for the first time.

![](https://github.com/jrf6xh/Capstone-Revenue_Prediction/blob/master/Images/avg_visits_vs_purchases.png)

## Data Cleaning
**Steps:**
* **Unpacking the nested structure of the original data:**  The data set from Kaggle is populated with some columns that have multiple features in a nested format.  In order to conduct proper modeling, this nested data was separated into distinct columns.
* **Dealing with categorical values:**  
The data set includes many categorical variables, many of which have hundreds or thousands of unique values.  In order to model this data, values with <500 occurances were grouped into a single 'Other' category.
* **Dealing with missing values:**
Some features were not made available by Google.  These columns were removed from the dataset.  Values of '(not set)', '(not applicable)', or similar were all standardized to a single 'None' value.  Missing values for continuous variables were imputed based on average values.

## Modeling
The modeling stage of this project was done in an iterative process of tuning models, trying new models, and engineering new features.

Throughout the modeling process R-Squared and RMSE were taken into account to judge model performance.  Cross validation and comparison vs training and test folds were also used to limit overfitting to the training data.

* **Baseline Model - KNN:**
This untuned model was used as a baseline off of which to judge the performance of new model iterations.

* **Tuned KNN Models:**
Tuned KNN models performed much better than the untuned model in terms of our metrics.  However, feature importances cannot be interpreted from KNN model outputs, so from here we transition to tree-based models.

* **Decision Tree Model:**
This model performed worse than the tuned KNN model, but well enough to indicate promise in further tree-based models.

* **Random Forest Models:**
These models performed about as well as the KNN model, but progress slowed after iterative rounds of tuning the models.

* **Feature Engineering:**
Because progress in terms of R-Squared and RMSE had slowed, we pulled in more features from the original data in order to improve performance.  K Means clustering was also employed here, with cluster labels and silhouette scores being added as columns to the modeling data set.

* **Random Forest Models - With New Data:**
These models performed the best out of all of the previous iterations.  Due to time constraints the modeling process was suspended here and the random forest model was selected as the final model for this stage of the project.

## Model Results
* **Cross Validation:**
The model had an R-Squared value of 8.5%, meaning that 8.5% of the variation in revenue could be explained by the model.  This is lower than optimal and will be improved with further iterations of this project.

* **Test Data:**
When using the model on the test data the R-Squared value falls to 0.9%.  This indicates a severe problem with overfitting to training data or an unrepresentative test-train split.  The primary goal of further iterations of this project will be to improve this metric.

## Limitations
**The Data:**
* Some data points were missing and had to be removed or imputed.
* Some data available to Google was not available to us.
* Data is not generalizable to all online stores.
* It is unknown if this is an exhaustive data set of all visits to the store within the 2016-2018 time frame.

**The Model:**
* Predictive power is very limited using the current best model.
* Exhaustive tuning is impossible due to limits on computational power.

## Future Improvements
**Additional Data:**
* Using more data to train the model could yield better results.
* Incorporating economic data could help correct for overall increases/decreases in consumer spending over time.
* Accessing additional data from Google could allow for more precise modeling.

**Improved Modeling:**
* Additional model types should be tested in an effort to increase evaluation metrics.  XGBoost regression could yield better results and will be tested in further iterations of this project.
* Generalizing the model to use only more common features would allow the model to be deployed for online stores other than Google's.

**Model Deployment:**
* Deploy a version of the model as a web application using Flask or a similar tool.

## Reproduction Instructions
* Download the original data from the Kaggle link in the source section.
* Run the first section of the Technical Notebook to prepare the data for modeling.
* The data is saved and loaded periodically throughout the code to limit the memory strain on your local computer and to allow for modular running of certain sections of the notebook.
* All code is contained in the Technical Notebook.  The code can also be run modularly in the following order: Preprocessing -> EDA -> Modeling -> Results_Analysis

## Sources
* The original data is made available by Google through a [Kaggle Competition.](https://www.kaggle.com/c/ga-customer-revenue-prediction/data)
* Code to unpack the nested data structure of the original data was written by Kagglers [Julian Peller](https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields) and [Changhao Lee.](https://www.kaggle.com/leechh/a-way-to-input-all-columns-include-hits?select=train_v2.csv)
* Images used in the high level presentation are sourced from [Pexels.](https://www.pexels.com/)
* Icons used in the high level presentation are sourced from [FlatIcon](https://www.flaticon.com/home)
* Information about Google and Ecommerce in general used in the high level presentation was gathered from [Statista](https://www.statista.com/statistics/664770/online-shopping-frequency-worldwide/), [Digital Commerce 360](https://www.digitalcommerce360.com/article/us-ecommerce-sales/#:~:text=2019%20Amazon%20Report&text=Total%20retail%20sales%20increased%203.8,estimates%20using%20Commerce%20Department%20figures.&text=That's%20the%20largest%20share%20of,63.8%25%20of%20all%20sales%20growth.), and [The Verge](https://www.theverge.com/2017/10/6/16426932/googles-2017-gadget-collection-photos).

## Project Information
**Contributors:** 
Jim Fay

Contact Info:
* [LinkedIn](https://www.linkedin.com/in/james-fay/)
* [Github](https://github.com/jrf6xh)

**Languages:** Python

**Libraries:** sklearn, pandas, matplotlib, numpy, seaborn

**Duration:** August 24 - September 2, 2020

**Last Update:** September 2, 2020

# Campaign Response Prediction

Market campaigns are one of the powerful tools to grow businesses. Introducing Machine Learning in target-customer selection can greatly improve campaign performance.

For example, instead of sending marketing emails to all 39999 customers at once, the specialist could first target 19999 random-sampled customers, and then adjust the targeting strategy in the second round based on the model prediction to get a better result.

![profit](https://user-images.githubusercontent.com/64244735/132591502-cdae7860-387c-40e4-8e42-788f7fcdb74e.png)

![result](https://user-images.githubusercontent.com/64244735/132591369-a8523b1c-3162-421d-9aec-7c91f640be6d.png)

PS: assumes that the variable cost of each email is $3 and that the average revenue from each response is $12.

---

### 1.0 Data Description

`Dataset Name`: Member Statistics

`Date of Data Acquisition`: XXXX/XX/XX

`Number of Fields`: 13

`Number of Records`: 39999

### 2.0 Data Cleaning

#### 2.1 Filling missing values

10 columns have missing values.

For the `edu` (education level), I fill the missing value with the mode in its education age.

For the `edu_ages`, I fill the missing value with the mode in its education level

The missing values of the other columns will be filled with either median or mode.

#### 2.2 Chnage data types

`industry` and `region` should be transformed into categorical variables.

### 3.0 EDA on 1st Round

![age](https://user-images.githubusercontent.com/64244735/132592705-3287c105-3fec-4da0-897b-ba6f7789e7ff.png)

The distributions of age between the two response types are quite different. Age could be a strong predictor variable.

![total_pageviews](https://user-images.githubusercontent.com/64244735/132592975-97196103-4e17-49a9-84a1-e065ecee8605.png)

It seems that `total_pageviews` is not strongly correlated with `response`.

![edu_age](https://user-images.githubusercontent.com/64244735/132593417-13bfb048-8878-454f-8459-3cd4c4549493.png)

The proportion of 8 and 12 is abnormal. There may be a strong correlation between `edu_ages` and `response`.

![blue_money](https://user-images.githubusercontent.com/64244735/132593748-ab0771a2-1c32-4716-9014-51f22bdcd84a.png)

The distribution of the response 1 group is different from that of the response 0 group from the second bins to the last bins. Thus it is reasonable to speculate that `blue_money` may be a good predictor variable.

![corr](https://user-images.githubusercontent.com/64244735/132594066-19095bf0-14f8-4ad1-906f-953e6bf59d41.png)

This correlation map justifies the previous inferences on variables.

For more details on EDA, please see the Jupiter notebook.

### 4.0 Feature Engineering

Always work with domain experts to create smart variables if you have the opportunities.

In this case, I have build two groups of new variables: age ratios and work-hour ratios

As `region` and `industry` are the categorical variables, I have to encode them into dummy variables and get 67 features in total.

The drawback of dummy encoding is dimension explosion, which may lead to overfitting. So feature selection is a reasonable next step.

### 5.0 Feature Selection

I will select 30 features by Kolmogorov-Smirnov score (KS score) and then narrow them down through the stepwise selection.

Kolmogorov-Smirnov test measures how far the distribution of a feature can be separated by the dependent variable. In the test, I will calculate the ks score for each column, sort the score descendingly, and take the top 30 to the next step.

After that, a forward selection is employed. It begins by building 30 Random Forests, each with only one feature. The feature with the best F1 score is kept and combined with one additional feature in the next round. The algorithm keeps the best feature in each round.

![ss](https://user-images.githubusercontent.com/64244735/132595184-f23992d3-de3a-439e-8434-b04d0fd98b3d.png)

I keep the features number generating highest F1 score.


### 6.0 Model building

F1 score is the model selection criterion, considering the imbalanced distribution of the dependent variable.

Before modeling, I split the first round dataset into a 70% training set and 30% validation set. 

#### 6.1 Baseline-Logistic Regression

I apply GridSearchCV taking F1-score as the metric to tune the hyperparameters of the model.

#### 6.2 Other non-linear models: Random Forest, Gradient Boosting Tree, and Neural Network

For each type of algorithm, two models are tuned: one with the default settings and the other optimized with GridSearchCV.

<table>
  <tr>
    <th>Model</th>
    <th>Train</th>
    <th>Val</th>
  </tr>
 <tr>
	  <td align="left">Logistic Regression</td>
	  <td align="center">0.582</td>
    <td align="center">0.604</td>
 </tr>
  <tr>
	  <td align="left">Random Forest (default)</td>
	  <td align="center">0.764</td>
    <td align="center">0.661</td>
 </tr>
  <tr>
	  <td align="left">Random Forest (tuned)</td>
	  <td align="center">0.688</td>
    <td align="center">0.661</td>
 </tr>
 <tr>
	  <td align="left">Gradient Boosting Decision Tree (default)</td>
	  <td align="center">0.669</td>
    <td align="center">0.662</td>
 </tr>

  <tr>
	  <td align="left">Gradient Boosting Decision Tree (tuned)</td>
	  <td align="center">0.711</td>
    <td align="center">0.684</td>
 </tr>
 <tr>
	  <td align="left">Neural Network (default)</td>
	  <td align="center">0.539</td>
    <td align="center">0.565</td>
 </tr>
 <tr>
	  <td align="left">Neural Network (tuned)</td>
	  <td align="center">0.674</td>
    <td align="center">0.652</td>
 </tr>
</table>

I choose the optimized Gradient Boosting Decision Tree because it has the best performance for validation data and does not overfit.

### 7.0 Prediction

The model decides for each customer if we should send the campaign ads. 

**F1 score** : 0.68

**Result Summary**:

<table>
  <tr>
    <th>Response Number</th>
    <th>Response Type</th>
    <th>Plan Type</th>
  </tr>
 <tr>
	  <td align="center">4813</td>
	  <td align="center">Yes</td>
    <td align="center">Original</td>
 </tr>
  <tr>
	  <td align="center">15187</td>
	  <td align="center">No</td>
    <td align="center">Original</td>
 </tr>
    <tr>
	  <td align="center">2958</td>
	  <td align="center">Yes</td>
    <td align="center">New</td>
 </tr>
      <tr>
	  <td align="center">946</td>
	  <td align="center">No</td>
    <td align="center">New</td>
 </tr>
</table>

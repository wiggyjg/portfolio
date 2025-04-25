[Home](./README.md)

# Homebrew Recipe for Success
### Can we predict the end abv alcohol level of a beer from the ingredients used?

## Why would we want to do this?
This project is intended to optimise brewing techniques in ale’s by finding the relationship between ingredients and the resulting alcohol (abv) content to hit UK taxation thresholds.

## Methodology
Create a regression model to determine how quantities of ingredients, specifically sugar influence the final abv alcohol percentage.

## Sourcing and Cleaning
The dataset used is: https://www.kaggle.com/datasets/liviam/homebrew-recipes/data.  GDPR and data protection guidelines do not apply to this dataset as it is publicly available.
A combination of SQL server and Tableau was used for data cleaning and exploration due to the scalability and potential dashboarding of pipelines afforded by this (should it be expanded to a larger dataset if productionised in the future).
This Kaggle dataset included the ingredients, alcohol level and description of around 1171 drinks of different types.  Issues with missing data and inaccurate categorisations were addressed through normalisation and grouping techniques to produce a clean dataset.  No anonymisation was required as no sensitive data was present.

Each drink in the data set has an entry with numerical values for the following:

Abv- Alcohol by volume (percentage of pure alcohol)
Ibu- Internation Bitterness units (scale of perceived bitterness)
Srm – Standard Reference Method (where a beer falls on a colour spectrum)
Og- Original Gravity (A measure of solids e.g. sugar before fermentation)
Fg- Final Gravity (A measure of solids e.g. sugar after fermentation)
Rs – Residual Sugar (sugars that are still present after fermentation)

Data processing involved imputing missing values with averages and, normalising outliers to the maximum value for that drink type. To deal with missing categorisations, a new high-level category was created by consolidating subcategories and using wildcard values to extract from the drink name column.  This technique generated new categories with more entries to facilitate the creation of more robust models.

### Feature Engineering of Categorical Data
The drinks in the sample are split into three main types, Beer, Cider and Mead.

Count of entries per category:
Category	count
Beer		1140
Cider		37
Mead		93

Due to the higher sample size, beer would be most suitable for a regression model as smaller sample sizes risk poor performing overfitted models.

When comparing [category] to [master_type] it is noted that some of Beers could be categorized as Ales, some as Lagers, some as Hybrid with some unknowns.  These drinks require different brewing techniques and have a different target audience, so are worth separating for the proposed analysis; a wildcard lookup is used to extract key words where it appears in [master_type] or [name]. 

#### The output is added as a new column:
Category	new_category
Beer		Ale
Beer		Hybrid
Beer		Lager
Beer		Unspecified
Cider		Cider
Mead		Mead

### New Category Count
![AbvPlaceholder](assets/DrinksCount.png) 

### Cleaning Numerical Data
Investigation into the numerical values uncovered null values in some columns; not all of these will be dealt with the same. 
Abv: As abv. Is the dependent variable and needs to be present and accurate, rows with missing values are dropped rather than using an average.
Other values: The average value for the relevant new_category are be used in its place.  This is not possible for ibu in the case of Cider as no values exist. 

### Dealing with outliers
#### Boxplot of Avb by Category
![AbvPlaceholder](assets/BoxPlot.png) 

In two ales, the Abv is clearly much higher and is likely an outlier or data error.  In some rare cases an alcohol can reach these extreme levels, however as the maximum recorded abv for beer is 16% abv (World Population Review, 2025), these values are reduced to 16%, so preserving the drinks as high Abv, but not having them skew the dataset and affect averages.

## Data Analytics
The remaining analysis was performed in Python.

```python
# Import Libs
import numpy as np
import pandas as pd
import pyodbc
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
```

### Import Data


```python
df = pd.read_excel("BeerData.xlsx",sheet_name = 'data', skiprows=0)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>category</th>
      <th>abv</th>
      <th>ibu</th>
      <th>srm</th>
      <th>og</th>
      <th>fg</th>
      <th>rs</th>
      <th>master_type</th>
      <th>new_master</th>
      <th>new_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Challenge Accepted American Amber Ale</td>
      <td>American Amber Ale</td>
      <td>Beer</td>
      <td>5.7</td>
      <td>33.0</td>
      <td>15.0</td>
      <td>1.056</td>
      <td>1.014</td>
      <td>0.042</td>
      <td>Ale</td>
      <td>Ale</td>
      <td>Ale</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Horizon Pale Ale</td>
      <td>American Pale Ale</td>
      <td>Beer</td>
      <td>6.5</td>
      <td>57.0</td>
      <td>7.0</td>
      <td>1.057</td>
      <td>1.009</td>
      <td>0.048</td>
      <td>Ale</td>
      <td>Ale</td>
      <td>Ale</td>
    </tr>
    <tr>
      <th>2</th>
      <td>One Way or Another Guava Blonde Ale</td>
      <td>Fruit Beer</td>
      <td>Beer</td>
      <td>3.4</td>
      <td>18.0</td>
      <td>4.0</td>
      <td>1.032</td>
      <td>1.006</td>
      <td>0.026</td>
      <td>Ale</td>
      <td>Ale</td>
      <td>Ale</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Old Man Johnson’s Farm Catharina Sour</td>
      <td>Sour Ale</td>
      <td>Beer</td>
      <td>5.1</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.045</td>
      <td>1.006</td>
      <td>0.039</td>
      <td>Ale</td>
      <td>Ale</td>
      <td>Ale</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kilt Dropper Wee Heavy</td>
      <td>Amber Ale</td>
      <td>Beer</td>
      <td>8.9</td>
      <td>26.0</td>
      <td>20.0</td>
      <td>1.083</td>
      <td>1.020</td>
      <td>0.063</td>
      <td>Ale</td>
      <td>Ale</td>
      <td>Ale</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1171</th>
      <td>Three Witches Wood-Aged Beer</td>
      <td>Wood-aged Beer</td>
      <td>Beer</td>
      <td>9.6</td>
      <td>29.0</td>
      <td>17.0</td>
      <td>1.102</td>
      <td>1.031</td>
      <td>0.071</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
    </tr>
    <tr>
      <th>1172</th>
      <td>Chipotle Lichtenhainer</td>
      <td>Spice &amp; Herb Beer</td>
      <td>Beer</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.042</td>
      <td>1.008</td>
      <td>0.034</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
    </tr>
    <tr>
      <th>1173</th>
      <td>Eis Eis Baby Wood-Aged Beer</td>
      <td>Wood-aged Beer</td>
      <td>Beer</td>
      <td>11.0</td>
      <td>29.0</td>
      <td>14.0</td>
      <td>1.107</td>
      <td>1.033</td>
      <td>0.074</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
    </tr>
    <tr>
      <th>1174</th>
      <td>Goats in a Tree in a Mezcal Barrel</td>
      <td>Wood-aged Beer</td>
      <td>Beer</td>
      <td>8.4</td>
      <td>18.0</td>
      <td>24.0</td>
      <td>1.087</td>
      <td>1.023</td>
      <td>0.064</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
    </tr>
    <tr>
      <th>1175</th>
      <td>Blade Runner Szechuan Pepper Beer</td>
      <td>Spice &amp; Herb Beer</td>
      <td>Beer</td>
      <td>5.3</td>
      <td>95.0</td>
      <td>9.5</td>
      <td>1.053</td>
      <td>1.012</td>
      <td>0.041</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
      <td>Unspecified</td>
    </tr>
  </tbody>
</table>
<p>1176 rows × 12 columns</p>
</div>




```python
df.sample()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>category</th>
      <th>abv</th>
      <th>ibu</th>
      <th>srm</th>
      <th>og</th>
      <th>fg</th>
      <th>rs</th>
      <th>master_type</th>
      <th>new_master</th>
      <th>new_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>279</th>
      <td>Trappist Tripel</td>
      <td>Belgian Ale</td>
      <td>Beer</td>
      <td>9.5</td>
      <td>35.0</td>
      <td>5.0</td>
      <td>1.079</td>
      <td>1.008</td>
      <td>0.071</td>
      <td>Ale</td>
      <td>Ale</td>
      <td>Ale</td>
    </tr>
  </tbody>
</table>
</div>



### Check expected data has been loaded


```python
# Find the number of columns and rows
df.shape
```




    (1176, 12)




```python
# Check for missing values
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1176 entries, 0 to 1175
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   name          1176 non-null   object 
     1   type          1176 non-null   object 
     2   category      1176 non-null   object 
     3   abv           1176 non-null   float64
     4   ibu           1142 non-null   float64
     5   srm           1176 non-null   float64
     6   og            1176 non-null   float64
     7   fg            1176 non-null   float64
     8   rs            1176 non-null   float64
     9   master_type   1176 non-null   object 
     10  new_master    1176 non-null   object 
     11  new_category  1176 non-null   object 
    dtypes: float64(6), object(6)
    memory usage: 110.4+ KB
    


```python
df['new_category'].value_counts()
```




    new_category
    Ale            858
    Lager          115
    Mead            83
    Hybrid          64
    Cider           34
    Unspecified     22
    Name: count, dtype: int64



#### Split To Ale
The large number of entries in the Ale new_category would likely ensure a robust model, and so the data was reduced to only Ale for the initial regression model. The below code block can be adjusted to run the model for other categories.


```python
## Extract only Ale
df = df[df['new_category'] == 'Ale']
```


```python
# Create a pairplot to visualise the relationships between our variables & the distributions of our data
sns.pairplot(df, diag_kind='kde', kind='reg', markers='+')
```




    <seaborn.axisgrid.PairGrid at 0x27f0cd35010>




    
![png](RecipeForSuccessRegressionPublic_files/RecipeForSuccessRegressionPublic_10_1.png)
    


The intention of this regression model is to determine the factors that can predict abv levels (the dependant variable), so I need to consider which other datapoint to use as the independent variable. It is common knowledge that sugar and alcohol are related as sugar is eaten by active yeast to produce alcohol, so that is the most likely candidate which confirmed in the above pairplot. It seems that Rs (Residual Sugar) is also a viable option, however this is a byproduct of the brewing process (the left over sugar in the drink) and so can't be used as a predictor. I suspect that Og and Rs are closely related, I can see that better in a heatmap.

## Heatmap


```python
dfnum = df.select_dtypes(exclude=['object']) ## Drop all non numerical dtypes
```


```python
##Correlaton Matrix
dfnum.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abv</th>
      <th>ibu</th>
      <th>srm</th>
      <th>og</th>
      <th>fg</th>
      <th>rs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>abv</th>
      <td>1.000000</td>
      <td>0.267507</td>
      <td>0.312264</td>
      <td>0.902800</td>
      <td>0.316737</td>
      <td>0.929304</td>
    </tr>
    <tr>
      <th>ibu</th>
      <td>0.267507</td>
      <td>1.000000</td>
      <td>0.139081</td>
      <td>0.301130</td>
      <td>0.190072</td>
      <td>0.280017</td>
    </tr>
    <tr>
      <th>srm</th>
      <td>0.312264</td>
      <td>0.139081</td>
      <td>1.000000</td>
      <td>0.414767</td>
      <td>0.243407</td>
      <td>0.325999</td>
    </tr>
    <tr>
      <th>og</th>
      <td>0.902800</td>
      <td>0.301130</td>
      <td>0.414767</td>
      <td>1.000000</td>
      <td>0.504604</td>
      <td>0.903349</td>
    </tr>
    <tr>
      <th>fg</th>
      <td>0.316737</td>
      <td>0.190072</td>
      <td>0.243407</td>
      <td>0.504604</td>
      <td>1.000000</td>
      <td>0.249213</td>
    </tr>
    <tr>
      <th>rs</th>
      <td>0.929304</td>
      <td>0.280017</td>
      <td>0.325999</td>
      <td>0.903349</td>
      <td>0.249213</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(5, 5))
sns.heatmap(dfnum.corr(), annot=True)
```




    <Axes: >




    
![png](RecipeForSuccessRegressionPublic_files/RecipeForSuccessRegressionPublic_15_1.png)
    


This heatmap confirms a high correlation coefficient between abv and og (0.89) suggesting that a linear regression model could capture this relationship well.  There is multicollinearity between og and rs which makes sense in this context as Residual Sugar (Rs) will be directly related to the amount of sugar added in the Og measurement.  This would only be an issue if I were to use these values in a multiple linear regression.

## Simple Linear Regression - Run 1

### Split the dependent and independent variables


```python
# Split the independent and dependent variables into X and y
X = df[['og']]
y = df['abv']
```


```python
# Check the shapes
print('X shape:', X.shape)
print('y shape:', y.shape)
```

    X shape: (858, 1)
    y shape: (858,)
    

### Split the data into 80/20 train/test


```python
# Perform the train/test split with random_state=23
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
```


```python
# Check the shapes
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)
```

    X_train: (686, 1)
    y_train: (686,)
    X_test: (172, 1)
    y_test: (172,)
    

### Examine and interpret the coefficients


```python
# Define the model variable
lin_reg = LinearRegression()
```


```python
# Fit it
lin_reg.fit(X_train, y_train)
```




<style>#sk-container-id-19 {color: black;}#sk-container-id-19 pre{padding: 0;}#sk-container-id-19 div.sk-toggleable {background-color: white;}#sk-container-id-19 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-19 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-19 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-19 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-19 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-19 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-19 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-19 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-19 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-19 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-19 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-19 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-19 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-19 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-19 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-19 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-19 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-19 div.sk-item {position: relative;z-index: 1;}#sk-container-id-19 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-19 div.sk-item::before, #sk-container-id-19 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-19 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-19 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-19 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-19 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-19 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-19 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-19 div.sk-label-container {text-align: center;}#sk-container-id-19 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-19 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-19" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" checked><label for="sk-estimator-id-19" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
# View the intercept
lin_reg.intercept_
```




    -92.0953732732754



That seems unlikely!  The model is suggesting that if no sugar is added, the end alcohol level will reach -91% which isn't possible.  The likely explanation for this is that the model, though it appears to suit linear regression would be best suited to polynomial regression at low alcohol levels.  The model also wouldn't account for natural sugars that would occur in other ingredients, meaning that the abv would never be zero.  As none of the drinks in the sample were alcohol free (though these would still contain trace alcohol) I can't explore that easily, but I am comfortable that the model will be fine for alcoholic drinks.


```python
# View the slope
lin_reg.coef_
```




    array([92.71873126])



Another unexpected result! The model is suggesting here that for every unit of sugar that is added, the alcohol will increase by 92%.  This is most likely because the two features are on different scales, I will scale the data and see if this improves.

### Scale the data

Only numerical data can be scaled so I will use the dfnum dataframe used for the heatmap.


```python
# Initialize the MinMaxScaler
##scaler = MinMaxScaler()

# StandardScaler
scaler = StandardScaler()

# RobustScaler
##scaler = RobustScaler()

# Normalize the data
normalised_data = scaler.fit_transform(dfnum)

# Create a new DataFrame with the normalized data
df_normalised = pd.DataFrame(normalised_data, columns=dfnum.columns)

print("Normalised DataFrame:")
print(df_normalised)
```

    Normalised DataFrame:
              abv       ibu       srm        og        fg        rs
    0   -0.375742 -0.214489 -0.025160 -0.359522 -0.149515 -0.477875
    1    0.029372  0.934782 -0.655702 -0.307051 -0.547520 -0.051901
    2   -1.540445 -0.932783 -0.892155 -1.618827 -0.786323 -1.613805
    3   -0.679577 -1.459532 -0.892155 -0.936703 -0.786323 -0.690862
    4    1.244714 -0.549693  0.368929  1.057195  0.328091  1.013033
    ..        ...       ...       ...       ...       ...       ...
    853  2.389161  0.007566  0.015331  2.316499  0.487293  2.574936
    854 -1.003669  0.007566  0.015331 -0.779290 -0.149515 -1.045840
    855 -0.735281  0.007566  0.015331 -0.726819 -0.388318 -0.761858
    856  2.323330  0.007566  0.015331  2.788739  1.283303  2.503941
    857  1.391568  0.007566  0.015331  1.529434  0.487293  1.510002
    
    [858 rows x 6 columns]
    

## Simple Linear Regression - Run 2

### Split the dependent and independent variables


```python
# Split the independent and dependent variables into X and y
X_normalised = df_normalised[['og']]
y_normalised = df_normalised['abv']
```


```python
# Check the shapes
print('X shape:', X_normalised.shape)
print('y shape:', y_normalised.shape)
```

    X shape: (858, 1)
    y shape: (858,)
    

### Split the data into 80/20 train/test


```python
# Perform the train/test split with random_state=23
X_train, X_test, y_train, y_test = train_test_split(X_normalised, y_normalised, test_size=0.2, random_state=23)
```


```python
# Check the shapes
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)
```

    X_train: (686, 1)
    y_train: (686,)
    X_test: (172, 1)
    y_test: (172,)
    

### Examine and interpret the scaled coefficient


```python
# Define the model variable
lin_reg = LinearRegression()
```


```python
# Fit it
lin_reg.fit(X_train, y_train)
```




<style>#sk-container-id-20 {color: black;}#sk-container-id-20 pre{padding: 0;}#sk-container-id-20 div.sk-toggleable {background-color: white;}#sk-container-id-20 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-20 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-20 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-20 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-20 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-20 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-20 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-20 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-20 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-20 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-20 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-20 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-20 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-20 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-20 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-20 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-20 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-20 div.sk-item {position: relative;z-index: 1;}#sk-container-id-20 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-20 div.sk-item::before, #sk-container-id-20 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-20 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-20 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-20 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-20 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-20 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-20 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-20 div.sk-label-container {text-align: center;}#sk-container-id-20 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-20 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-20" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" checked><label for="sk-estimator-id-20" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
# View the intercept
lin_reg.intercept_
```




    0.00450802136460763




```python
# View the slope
lin_reg.coef_
```




    array([0.89481914])



These numbers look more realistic on the scaled data but I will need to scale them back to get anything interpretable.

### Scale back to see the coefficients


```python
# Get the coefficients and intercept from the model
coef_normalized = lin_reg.coef_[0]
intercept_normalized = lin_reg.intercept_

# Inverse transform the coefficients and intercept to original scale
coef_original = coef_normalized * (df['abv'].max() - df['abv'].min()) / (df['og'].max() - df['og'].min())
intercept_original = intercept_normalized * (df['abv'].max() - df['abv'].min()) + df['abv'].min() - coef_original * df['og'].min()

print(f"Original Coefficient: {coef_original}")
print(f"Original Intercept: {intercept_original}")
```

    Original Coefficient: 67.48635968742961
    Original Intercept: -65.802824074944
    

This still looks incorrect, but  this is because as standard, the coefficient is generated for every 1 unit increase in the independent variable, but as my units are so small, the increments on this axis need to be 100 times smaller. 


```python
# Get the coefficients and intercept from the model
coef_normalized = lin_reg.coef_[0]
intercept_normalized = lin_reg.intercept_

# Inverse transform the coefficients and intercept to original scale
coef_original = coef_normalized * (df['abv'].max() - df['abv'].min()) / (df['og'].max() - df['og'].min())*0.01
intercept_original = intercept_normalized * (df['abv'].max() - df['abv'].min()) + df['abv'].min() - coef_original * df['og'].min() *0.01

print(f"Original Coefficient: {coef_original}")
print(f"Original Intercept: {intercept_original}")
```

    Original Coefficient: 0.6748635968742961
    Original Intercept: 2.5540219201858663
    

I added * 0.01 to the rescaling lines to multiply my standard coefficient by 0.01 and show the increase for every 0.01 unit increase of Og. Though the intercept is more realistic now (assuming a natural level of sugar in other ingredients would create an ale of a low alcohol content) achieving 2.6% Abv with no added sugar is likely inaccurate as there are non-alcoholic ales on the market with just trace alcohol levels.  As I stated earlier, accuracy for low alcohol levels may be improved with a polynomial regression, though that would require more data with ales in the 0% to xxx% Abv range.  



However, the final result for the coefficient an increase of 0.67% Abv for every 0.01 increase in Og.  This information can be used to optimise brewing techniques to hit tax thresholds and refine brewing optimisatio.



## Modelling

### Generate predictions for the test set


```python
# Create X and y variables
X = df[['og']]
y = df['abv']
```


```python
## Test and Train - 80% Train 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=423)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
```

    X_train shape: (686, 1)
    X_test shape: (172, 1)
    y_train shape: (686,)
    y_test shape: (172,)
    


```python
# Train a simple linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
```




<style>#sk-container-id-21 {color: black;}#sk-container-id-21 pre{padding: 0;}#sk-container-id-21 div.sk-toggleable {background-color: white;}#sk-container-id-21 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-21 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-21 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-21 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-21 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-21 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-21 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-21 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-21 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-21 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-21 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-21 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-21 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-21 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-21 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-21 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-21 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-21 div.sk-item {position: relative;z-index: 1;}#sk-container-id-21 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-21 div.sk-item::before, #sk-container-id-21 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-21 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-21 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-21 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-21 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-21 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-21 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-21 div.sk-label-container {text-align: center;}#sk-container-id-21 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-21 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-21" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" checked><label for="sk-estimator-id-21" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
preds = lin_reg.predict(X_test)

sns.scatterplot(x=X_train['og'], y=y_train, label='Train Data')
sns.scatterplot(x=X_test['og'], y=y_test, label='Test Data')
sns.lineplot(x=X_test['og'], y=preds, color='red', label='Predicted Line')

plt.legend()
plt.show()
```


    
![png](RecipeForSuccessRegressionPublic_files/RecipeForSuccessRegressionPublic_57_0.png)
    


### Calculate error metrics


```python
# Calculate predictions for the train set
preds_train = lin_reg.predict(X_train)
```


```python
# Calculate predictions for the test set
preds_test = lin_reg.predict(X_test)
```


```python
# Calculate and print metrics
mae = mean_absolute_error(y_test, preds_test)
rmse = mean_squared_error(y_test, preds_test)

print('MAE:', mae)
print('RMSE:', rmse)
```

    MAE: 0.4861422828376704
    RMSE: 0.4619438551355548
    

The Mean Absolute Error Value only appears to deviate by around 0.49 units which means the model is quite accurate and with the Root Mean Squared Error being 0.46 it seems that the model's predictions are close to actual values with few major deviations or errors. Overall, the model is performing well and making accurate predictions.

### Checking R Squared 


```python
## Model summary
X = sm.add_constant(X)
lin_reg = sm.OLS(y, X)
model = lin_reg.fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>abv</td>       <th>  R-squared:         </th> <td>   0.815</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.815</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3772.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 23 Apr 2025</td> <th>  Prob (F-statistic):</th> <td>6.04e-316</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:17:46</td>     <th>  Log-Likelihood:    </th> <td> -1077.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   858</td>      <th>  AIC:               </th> <td>   2159.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   856</td>      <th>  BIC:               </th> <td>   2168.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>  -92.9832</td> <td>    1.619</td> <td>  -57.430</td> <td> 0.000</td> <td>  -96.161</td> <td>  -89.805</td>
</tr>
<tr>
  <th>og</th>    <td>   93.5457</td> <td>    1.523</td> <td>   61.418</td> <td> 0.000</td> <td>   90.556</td> <td>   96.535</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>797.360</td> <th>  Durbin-Watson:     </th> <td>   1.860</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>86636.119</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 3.767</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>51.648</td>  <th>  Cond. No.          </th> <td>    112.</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Though I don't necessarily need to consider Adjusted R Squared (as I used only one dependent and independent variable), the R squared and Adjusted R Squared scores of 0.815, mean that 82% of the variability in Abv can be explained by the Original Gravity even when adjusting for the predictors in the model, which is pretty good and overall I'm satisfied that it is performing well.



I am aware that formulas have existed for many years that aim to predict the alcohol content of ale from the original gravity and a good next step to verify the performance would be to benchmark this model against real life examples and test each method.



If I were to improve this model I would consider incorporating other features such as brewing time, temperature or yeast content into a multiple regression model, however all the remaining features in the dataset are byproducts of the brewing process and can't be adjusted before fermentation, so I would need a larger dataset to do tis.




```python

```


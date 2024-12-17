import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Read Dataset
### Load the dataset (e.g., using pandas.read_csv()).
df = pd.read_csv('Datasets/Employee.csv')

### Inspect the data structure (.head(), .info(), .describe()).
print(df.head())
print(df.info())
print(df.describe())
print()

# 2. Explore the Data
## 1. Basic Exploration:
### Print the first few rows to understand the structure of the dataset.
print(df.head())
print()

## 2. Check Datatypes:
### Identify numerical and categorical columns.
numericalColumns = df.select_dtypes(include=['number']).columns
categoricalColumns = df.select_dtypes(include=['object']).columns

print(f'Numerical columns: {numericalColumns.tolist()}')
print(f'Categorical columns: {categoricalColumns.tolist()}')
print()

### Convert categorical columns to the category datatype for better memory efficiency.
df[categoricalColumns] = df[categoricalColumns].astype('category')

## 3. Categorical Column Analysis:
### Display the number of unique categories in each categorical column.
for column in categoricalColumns:
    print(f'Column `{column}` has {df[column].nunique()} categories.')
print()

## 4. Missing Values:
### Check for missing values in each column.
missingValues = df.isnull().sum()

### Calculate the percentage of missing values.
missingPercentage = (missingValues / len(df)) * 100

print(pd.DataFrame(data=[missingValues, missingPercentage], index=['Missing Values', 'Missing Percentage']))
print()

# 3. Handle Missing Values
## 1. High Null Ratios:
### Drop columns where the null-value percentage is too high (e.g., >50%).
columnsToDrop = missingPercentage[missingPercentage > 50].index
df.drop(columns=columnsToDrop)

## 2. Categorical Columns:
### Fill missing values with the mode of the column.
categoricalColumns = df.select_dtypes(include=['category']).columns
for column in categoricalColumns:
    df.fillna({column: df[column].mode()[0]}, inplace=True)

## 3. Numerical Columns:
### Visualize the distribution of each column (e.g., using histograms or skewness statistics).
numericalColumns = df.select_dtypes(include=['number']).columns
for column in numericalColumns:
    print(f'Column {column} has {df[column].nunique()} unique values ({df[column].unique().tolist()}).')

df[numericalColumns].hist(bins=30, figsize=(15, 10))
plt.show()

### If skewed, fill missing values with the median to reduce the effect of outliers.
numericalColumns = df.select_dtypes(include=['number']).columns
for column in numericalColumns:
    if df[column].isnull().sum() > 0:
        skewness = df[column].skew() # There are multiple ways to calculate skewness, but the most common formula is based on the Pearson moment coefficient of skewness.
        if skewness > 1: # Skewed distribution.
            df.fillna({column: df[column].median()}, inplace=True)
        else:
            df.fillna({column: df[column].mean()}, inplace=True)

## 4. Validate Null Handling:
### Recheck the dataset to ensure no missing values remain.
assert df.isnull().sum().sum() == 0

# 4. Outlier Detection and Treatment
"""
    An outlier is an extremely high or extremely low value in a dataset.
    We can identify an outlier using the Interquartile Range (IQR) Method.

    Data is divided into four equal parts when sorted.
    Q1 (First Quartile): The value below which 25% of the data lies (25th percentile).
    Q3 (Third Quartile): The value below which 75% of the data lies (75th percentile).
    Interquartile Range: IQR = Q3 - Q1

    Outliers are data points that fall significantly outside the central range. To detect them:
        1. Calculate the lower and upper bounds:
            Upper bound = Q3 + 1.5 * IQR.
            Lower bound = Q1 - 1.5 * IQR.
        2. Any data point below the lower bound or above the upper bound is considered an outlier.
"""
## 1. Visualize Outliers:
### Use box plots to detect outliers in numerical columns.
#### See this https://www.youtube.com/watch?v=nV8jR8M8C74
numericalColumns = df.select_dtypes(include=['number']).columns
plt.figure(figsize=(15, 10))
for index, column in enumerate(numericalColumns):
    plt.subplot((len(numericalColumns) + 2) // 3, 3, index + 1)
    sns.boxplot(x=df[column])
    plt.title(f"Box plot of {column}")

plt.show()

## 2. Capping Outliers:
### Replace values above the upper whisker with the maximum non-outlier value (upper bound).
### Replace values below the lower whisker with the minimum non-outlier value (lower bound).
def func(x):
    if x < lowerBound:
        return lowerBound
    if x > upperBound:
        return upperBound
    return x

numericalColumns = df.select_dtypes(include=['number']).columns
for column in numericalColumns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR
    df[column] = df[column].apply(func=func)

## 3. Categorical Outliers:
### For rare categories (low frequency), replace them with the mode of the column.
categoricalColumns = df.select_dtypes(include=['category']).columns
for column in categoricalColumns:
    ## Returns a Series whose index is the unique values of that column.
    ## The corresponding values are the frequency of each value normalized.
    mode = df[column].mode()[0]
    fp = df[column].value_counts(normalize=True)
    rareCategories = fp[fp < 0.05].index ## Extracts the values with low frequency percentage.
    df[column] = df[column].apply(
        func=lambda x: mode if x in rareCategories else x
    )

# 5. Check for Duplicates
### Remove duplicate rows using drop_duplicates().

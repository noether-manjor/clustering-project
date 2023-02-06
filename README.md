# Tasting For Wine Quality<br>
## Project Description<br>
For this project we will be gathering some wine data, analyzing it for particular features that will later be used as drivers for wine quality. 
We will project is to analyze the data, find driving features of wine quality, predict quality of wine, and recommend actions to produce good quality of wine.

## Project Goals<br>
- Discover drivers of quality
- Use drivers to develop a machine learning model that accurately predicts wine quality
- This information could be used on future datasets to help find high quality wine

## Initial Questions<br>
- Does wine color affect wine quality?
- Does a higher quality mean higher alcohol content?
- Is there a relationship between Citric Acid and Quality?
- Is there a relationship between Free Sulfur Dioxide and Quality?

## The Plan<br>
#### Acquire data
- Data aquired Data.World
- Red wine containted 1599 rows × 12 columns and white wine contained 4898 rows × 13 columns before cleaning and concatenating
- Each row represents a wine
- Each column represents a feature associated with the wine

#### Prepare
- Renamed colums to read pythonic
- Checked for nulls and outliers
- Checked that column data types were appropriate and had to change as necessary
- Added column after concatenating to clarify whether it is a red or white wine
- Split data into train, validate and test, stratifying on 'quality'

#### Explore data in search of drivers of quality and answer the following questions:
- Does wine color affect wine quality?
- Does a higher quality mean higher alcohol content?
- Is there a relationship between Citric Acid and Quality?
- Is there a relationship between Free Sulfur Dioxide and Quality?

## Data Dictionary<br>
| Name                 | Definition |
| -------------------- | ---------- |
| fixed acidity        | Specifies the fixed acidity of the wine |
| volatile acidity     | Specifies the volatile acidity of the wine |
| citric acid          | Specifies the citric acid of the wine |
| residual sugar       | Specifies the residual sugar of the wine |
| chlorides            | Specifies the chlorides of the wine |
| free sulfur dioxide  | Specifies the free sulfur dioxide of the wine |
| total sulfur dioxide | Specifies the total sulfur dioxide of the wine |
| density              | Specifies the density of the wine |
| pH                   | Specifies the pH of the wine |
| sulphates            | Specifies the sulphates of the wine |
| alcohol              | Specifies the alcohol of the wine |
| quality              | Scale 1-10, specifies quality of wine |
| wine_color           | Red or white, specifies color of wine | <br>

## Steps to Reproduce
- Clone this repo.
- Acquire the data from Data.World
- Put the data in the file containing the cloned repo.
- Run notebook.

## Takeaways and Conclusions

- Wine color is not a big driver of quality
- Alcohol is the biggest driver of quality
- Free sulfur dioxide and citric acid were also not big drivers
- Clustering did show us a similar plot to to one of our questions, but nothing particularly interesting

## Recommendations
- Focus more on investigation of sulfur content in soil due to correlation to quality
- Get a more robust dataset with red wine
- Add a feature such as vineyard or location to investigate higher rated wines
- If provided more time to work on this project we would take a longer look at subetting on hyperparameters, clustering, and modeling on just white wine as it was a more robust dataset 



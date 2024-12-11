# Recipe Analysis Project (Ashley Chu and Feiwei Peng DSC 80 Final Project)

Authors: Feiwei Peng and Ashley Chu

## Introduction

In this project, we examine, analyze and try to predict the popularity of recipes from food.com, which we defined by a recipe’s average rating. We were interested in this question due to our personal interests in cooking and finding highly rated and delicious recipes, and were wondering what factors could contribute to making a recipe of such qualities. 

Our dataset was scraped from food.com by Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni and Julian McAuley for their paper called “Generating Personalized Recipes from Historical User Preferences”, and consists of two CSV files containing recipe attributes and recipe interactions (ratings and reviews). We only used the recipes and reviews posted since 2008 since the original data was quite large, and thus ended up with 83,782 rows in our recipe dataset and 731,927 rows in our interactions dataset. We merged the two dataframes, and our resulting columns of interest in the are:

| Column             | Description
| :----------------- | :----------------------------------------|
| `'id'`             | Recipe ID                     |
| `'minutes'`        | Minutes to prepare recipe      |  
| `'tags'`           | Food.com tags for recipe|
| `'nutrition'`      | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| `'n_steps'`        | Number of steps in recipe        |
| `'n_ingredients'`  | Number of ingredients in recipe  |
| `'rating'`    | Rating given        |

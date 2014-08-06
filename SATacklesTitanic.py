import csv as csv
import numpy as np
import pandas as pd
import pylab as P
from sklearn.ensemble import RandomForestClassifier

#------------------TRAIN DATA START--------------------#
train_df = pd.read_csv("train.csv", header=0)

train_df['Gender'] = train_df.Sex.map({'female':0, 'male':1}).astype(int)
train_df['EmbarkedInt'] = train_df.Embarked.fillna('NoEntry').map({'NoEntry':-1, 'S':0, 'C':1, 'Q':2}).astype(int)
train_df['AgeFill'] = train_df.Age

median_ages = np.zeros((2,3))

#Fill in missing ages
for i in range(0,2):
    for j in range (0,3):
        median_ages[i,j] = train_df[(train_df.Gender == i) & (train_df.Pclass == j+1)].Age.dropna().median()

for i in range(0,2):
    for j in range (0,3):
        train_df.loc[((train_df.Age.isnull()) & (train_df.Gender == i) & (train_df.Pclass == j+1), 'AgeFill')] = median_ages[i,j]

train_df['AgeIsNull'] = pd.isnull(train_df.Age).astype(int)

#Calculate familysize from parents/children + siblings/spouses
train_df['FamilySize'] = train_df.Parch + train_df.SibSp
#Calculate a value based on age and class
train_df['Age*Class'] = train_df.AgeFill * train_df.Pclass

#Drop unnecessary columns
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId'], axis=1)

#Drop fare column as this column is biased
train_df = train_df.drop(['Fare'], axis=1)

train_data = train_df.values

#------------------TRAIN DATA END----------------------#

#------------------TEST DATA START---------------------#
test_df = pd.read_csv("train_test.csv", header=0)

test_df['Gender'] = test_df.Sex.map({'female':0, 'male':1}).astype(int)
test_df['EmbarkedInt'] = test_df.Embarked.fillna('NoEntry').map({'NoEntry':-1, 'S':0, 'C':1, 'Q':2}).astype(int)
test_df['AgeFill'] = test_df.Age

median_ages = np.zeros((2,3))

#Fill in missing ages
for i in range(0,2):
    for j in range (0,3):
        median_ages[i,j] = test_df[(test_df.Gender == i) & (test_df.Pclass == j+1)].Age.dropna().median()

for i in range(0,2):
    for j in range (0,3):
        test_df.loc[((test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1), 'AgeFill')] = median_ages[i,j]

test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)

#Calculate familysize from parents/children + siblings/spouses
test_df['FamilySize'] = test_df.Parch + test_df.SibSp
#Calculate a value based on age and class
test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass


# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values

#Drop unnecessary columns
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId'], axis=1)

#Drop fare column as this column is biased
test_df = test_df.drop(['Fare'], axis=1)

test_data = test_df.values
#------------------TEST DATA END-----------------------#

print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("mymodel.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
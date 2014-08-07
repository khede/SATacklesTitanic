import csv as csv
import numpy as np
import pandas as pd
import pylab as P
from sklearn.ensemble import RandomForestClassifier

#------------------DATA CLEAN VALUES START-------------#
#Get harvest data for predicting missing values from both the train set and the test set
train_and_test_df = pd.read_csv("train_and_test.csv", header=0)

train_and_test_df['Gender'] = train_and_test_df.Sex.map({'female':0, 'male':1}).astype(int)

#Initialise the salutation
train_and_test_df['Salutation'] = -1
#All salutations in the test and training data
salutations = ['Mr\.', 'Mrs\.', 'Miss\.', 'Master\.', 'Dr\.', 'Rev\.', 'Sir\.', 'Lady\.', 'Col\.', 'Capt\.', 'Major\.', 'Don\.', 'Dona\.', 'Countess\.', 'Jonkheer\.', 'Mme\.', 'Mlle\.', 'Ms\.']
#The corresponding salutation ids
salutationIds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#Set the salutation for each person
for i in range(0,len(salutations)):
    train_and_test_df['Salutation'] = np.where(train_and_test_df['Name'].str.contains(salutations[i]),salutationIds[i],train_and_test_df['Salutation'])

salutation_ages_pclass = np.zeros((3,len(salutations)))

#Fill in missing ages based on salutation and class
for i in range(0,3):
    for j in range(0,len(salutations)):
        salutation_ages_pclass[i,j] = train_and_test_df[(train_and_test_df['Salutation'] == j) & (train_and_test_df['Pclass'] == i+1)].Age.dropna().mean()

median_ages = np.zeros((2,3))

#Fill in missing ages based on class and gender
for i in range(0,2):
    for j in range (0,3):
        median_ages[i,j] = train_and_test_df[(train_and_test_df.Gender == i) & (train_and_test_df.Pclass == j+1)].Age.dropna().median()

#Display correlation between age and salutation
#for i in range(0,len(salutations)):
    #P.subplot(3,6,i+1)
    #train_and_test_df.Age[train_df.Salutation == i+1].dropna().hist(bins=(100), range=(0,80), alpha=.5)
    #P.title(salutations[i])
    #P.xlim([0,train_df.Age.max()])

#P.show()

#------------------DATA CLEAN VALUES END---------------#

#------------------TRAIN DATA START--------------------#
train_df = pd.read_csv("train.csv", header=0)

print train_df.info();

train_df['Gender'] = train_df.Sex.map({'female':0, 'male':1}).astype(int)
train_df['EmbarkedInt'] = train_df.Embarked.fillna('NoEntry').map({'NoEntry':0, 'S':0, 'C':1, 'Q':2}).astype(int)
train_df['AgeFill'] = train_df.Age
#Initialise the salutation
train_df['Salutation'] = -1

#Set the salutation for each person
for i in range(0,len(salutations)):
    train_df['Salutation'] = np.where(train_df['Name'].str.contains(salutations[i]),salutationIds[i],train_df['Salutation'])

#Fill in missing ages based on salutation and class
for i in range(0,3):
    for j in range(0,len(salutations)):
        train_df.loc[((train_df.Age.isnull()) & (train_df['Salutation'] == j) & (train_df['Pclass'] == i+1), 'AgeFill')] = salutation_ages_pclass[i,j]


#Fill in missing ages based on class and gender
for i in range(0,2):
    for j in range (0,3):
        train_df.loc[((train_df.Age.isnull()) & (train_df.Gender == i) & (train_df.Pclass == j+1), 'AgeFill')] = median_ages[i,j]

train_df['AgeIsNull'] = pd.isnull(train_df.Age).astype(int)

#Calculate familysize from parents/children + siblings/spouses
train_df['FamilySize'] = train_df.Parch + train_df.SibSp
#Calculate a value based on age and class
train_df['Age*Class'] = train_df.AgeFill * train_df.Pclass

#Drop unnecessary columns
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId', 'Salutation'], axis=1)

#Drop fare column as this column is biased
train_df = train_df.drop(['Fare'], axis=1)

train_data = train_df.values

#------------------TRAIN DATA END----------------------#

#------------------TEST DATA START---------------------#
test_df = pd.read_csv("test.csv", header=0)

test_df['Gender'] = test_df.Sex.map({'female':0, 'male':1}).astype(int)
test_df['EmbarkedInt'] = test_df.Embarked.fillna('NoEntry').map({'NoEntry':-1, 'S':0, 'C':1, 'Q':2}).astype(int)
test_df['AgeFill'] = test_df.Age
#Initialise the salutation
test_df['Salutation'] = -1

print test_df.info()
#Set the salutation for each person
for i in range(0,len(salutations)):
    test_df['Salutation'] = np.where(test_df['Name'].str.contains(salutations[i]),salutationIds[i],test_df['Salutation'])


#Fill in missing ages based on salutation and class
for i in range(0,3):
    for j in range(0,len(salutations)):
        test_df.loc[((test_df.Age.isnull()) & (test_df['Salutation'] == j) & (test_df['Pclass'] == i+1), 'AgeFill')] = salutation_ages_pclass[i,j]


#Fill in missing ages based on class and gender
for i in range(0,2):
    for j in range (0,3):
        test_df.loc[((test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1), 'AgeFill')] = median_ages[i,j]

test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)
print test_df.info()
#Calculate familysize from parents/children + siblings/spouses
test_df['FamilySize'] = test_df.Parch + test_df.SibSp
#Calculate a value based on age and class
test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass


# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values

#Drop unnecessary columns
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId', 'Salutation'], axis=1)

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
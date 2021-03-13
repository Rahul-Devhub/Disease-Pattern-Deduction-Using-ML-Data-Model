## Tensorflow-Machine-Learning-Model-Binary-Viruses-Prediction-Dataset

Binary pattern that are frequently co-regulated. Such matrices provide expression values for given disease detection based on extraction situations (the lines) and given genes (columns). 
The widely wide-spread itemset (sets of columns) extraction method allows to process tough instances (millions of lines, thousands of columns) furnished that statistics is now not too dense. 
However, expression matrices can be dense and have commonly solely have few strains w.r.t. the range of columns. Known algorithms, which include the current algorithms that compute the so-called condensed.

Introduction

Data Science is the area of learning about which includes extracting insights from widespread quantities of information by means of the use of a variety of scientific methods, algorithms, and processes.
It helps you to find out hidden patterns from the raw data. The time period Data Science has emerged due to the fact of the evolution of mathematical statistics, statistics analysis, and massive data.
Data Science is an interdisciplinary discipline that approves you to extract expertise from structured or unstructured data. Data science allows you to translate a commercial enterprise trouble into a 
lookup challenge and then translate it returned into a sensible solution.

Data Science Process

Data science technique is consisting of six key modules discovery, statistics preparation, model planning, model building, operation and transmit results. These procedures are given

Discovery

Discovery step entails obtaining facts from all the recognized internal & external sources which helps you to answer the trading..



The statistics can be:

• Logs from webservers

• Data gathered from social media

• Census datasets

• Data streamed from on line sources the use of APIs

Data Preparation

Data can have plenty of inconsistencies like lacking value, blank columns, mistaken facts layout which wants to be cleaned. You want to process, explore, and situation information earlier than modeling. The cleaner your data, the higher are your predictions.

Model Planning

In this stage, you want to decide the technique and approach to draw the relation between input variables. Planning for a pattern is carried out through the usage of exceptional statistical formulation and visualization tools. SQL evaluation services, R, and SAS/access are some of the equipment used for this purpose.

Model Building

In this step, the right pattern constructing manner starts. Here, Data scientist distributes datasets for training and testing. Techniques like association, classification, and clustering are utilized to the training data set. The model as soon as organized is examined towards the ”testing” dataset.

Operationalize

In this stage, you supply the final word base-lined pattern with reports, code, and technical documents.
Model is deployed into a real-time manufacturing surroundings after thorough testing.


Communicate Result

In this stage, the key findings are conveys to all or any contributor. This helps you to work out if the implications of the assignment are successful or a failure based totally on the inputs from the model.

Problem Statement and Domain

Medical information is critical for Doctors, Scientist and different fascinating events to acknowledge disease, sides of prognosis and engineer some element to prevent humans in dispose. Medical statistics may be a sizable developing area during which many evaluation may be completed to acknowledge and forestall disease. Our data-set consists of every year targeted disorder with appreciate to gender and age. we are going to strive to research it and visualize the records to meet some questions distinctive within the “List of Requirements”.



List of Requirements


• Age vs Disease.
Model Building

In this step, the right pattern constructing manner starts. Here, Data scientist distributes datasets for training and testing. Techniques like association, classification, and clustering are utilized to the training data set. The model as soon as organized is examined towards the ”testing” dataset.

Operationalize

In this stage, you supply the final word base-lined pattern with reports, code, and technical documents.
Model is deployed into a real-time manufacturing surroundings after thorough testing.


Communicate Result

In this stage, the key findings are conveys to all or any contributor. This helps you to work out if the implications of the assignment are successful or a failure based totally on the inputs from the model.

Problem Statement and Domain

Medical information is critical for Doctors, Scientist and different fascinating events to acknowledge disease, sides of prognosis and engineer some element to prevent humans in dispose. Medical statistics may be a sizable developing area during which many evaluation may be completed to acknowledge and forestall disease. Our data-set consists of every year targeted disorder with appreciate to gender and age. we are going to strive to research it and visualize the records to meet some questions distinctive within the “List of Requirements”.

List of Requirements


• Age vs Disease.
• No null values found.

Total Records

6,907 Records

Columns Information

•	Age (Years): Age of Patient.

•	Range: 21 - 76

•	Gender : Gender information of the patient

•	Unique Value: Male, Female

•	Diagnosis; Disease diagnosed

•	Disease Types


– Acute febrile mucocutaneous node syndrome [MCLS]

– Crimean viral haemorrhagic fever (CHF Congo virus)

– Dengue

– Encephalitis

– Febrile convulsions


– Fever of unknown origin (PUO)

– Malaria

– Meningitis

– Pneumonia

– infectious disease

– Sinusitis

– Tetanus

– typhoid

– Viral Fever

– hepatitis

– Maternal pyrexia

– Mosquito-borne virus infection

– Postprocedural fever

– Mosquito-borne fever/Chikungunya


• Year : Year on which disease is diagnosed 2015 to 2019.

Note: Dataset are already clean and it contains no null/empty values.

Data Analytics

Libraries used.

```r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency
from scipy.stats import chi2
```

Declaring and exploring DataSet.

```r
file_path = os.path.abspath('../Disease Prediction Project/Disease Data.xlsx')
csv_path = os.path.abspath('../Disease Prediction Project/DT.csv')
#	disease_data = pd.read_excel( file_path ) disease_data = pd.read_csv( csv_path ) disease_data.head() disease_data.dtypes disease_data.info()

```

Age vs Disease

```r 
plt.figure(figsize=(12,7))
plt.xlabel("Diseases")
plt.ylabel("Age")
plt.title("Disease vs Age (2015-2019)")
plt.xticks(rotation=90)
plt.plot(disease_data.DIAGNOSIS, disease_data["Age (Years)"],'.')
age_btw_20_30 = disease_data.loc[ (disease_data ["Age (Years)"]>=20) & (disease_data ["Age (Years)"]<30)].groupby(['DIAGNOSIS'])["S. No"].count()
age_btw_30_40 = disease_data.loc[ (disease_data ["Age (Years)"]>=30) & (disease_data ["Age (Years)"]<40)].groupby(['DIAGNOSIS'])["S. No"].count()
age_btw_40_50 = disease_data.loc[ (disease_data ["Age (Years)"]>=40) & (disease_data ["Age (Years)"]<50)].groupby(['DIAGNOSIS'])["S. No"].count()
age_btw_50_60 = disease_data.loc[ (disease_data ["Age (Years)"]>=50) & (disease_data ["Age (Years)"]<60)].groupby(['DIAGNOSIS'])["S. No"].count()
age_btw_60_70 = disease_data.loc[ (disease_data ["Age (Years)"]>=60) & (disease_data ["Age (Years)"]<70)].groupby(['DIAGNOSIS'])["S. No"].count()
age_btw_70_80 = disease_data.loc[ (disease_data ["Age (Years)"]>=70) & (disease_data ["Age (Years)"]<80)].groupby(['DIAGNOSIS'])["S.No"].count()
plt.figure(figsize=(12,7))
for disease in diseases_count.sort_values(ascending=False).iloc[:4].index.values:
df= pd.DataFrame(columns=['age_group','percentage'])
df = df.append( {'age_group':'20-30','percentage':( ( age_btw_20_30[disease] if disease in age_btw_20_30 else 0 )/diseases_count[disease])*100}, ignore_index=True )
df = df.append( {'age_group':'30-40','percentage':( ( age_btw_30_40[disease] if disease in age_btw_30_40 else 0 )/diseases_count[disease])*100}, ignore_index=True )
df = df.append( {'age_group':'40-50','percentage':( ( age_btw_40_50[disease] if disease in age_btw_40_50 else 0 )/diseases_count[disease])*100}, ignore_index=True )
df = df.append( {'age_group':'50-60','percentage':( ( age_btw_50_60[disease] if disease in age_btw_50_60 else 0 )/diseases_count[disease])*100}, ignore_index=True )
df = df.append( {'age_group':'60-70','percentage':( ( age_btw_60_70[disease] if disease in age_btw_60_70 else 0 )/diseases_count[disease])*100}, ignore_index=True )
df = df.append( {'age_group':'70-80','percentage':( ( age_btw_70_680[disease] if disease in age_btw_70_80 else 0 )/diseases_count[disease])*100}, ignore_index=True )
```

```r
plt.plot( 'age_group', 'percentage', data=df,marker='s' ,label='%s' % disease)
plt.xlabel("Age Groups")
plt.ylabel("Prevalence in %")
plt.title("Age Group Vs Top 4 Prevalent diseases in % (2015-2019)")
plt.legend()
```

Analysis

From the above figures we are able to see that:

20-30 age bracket suffering all told Pneumonia ,Viral Fever , enteric fever , Dengue

30-40 cohort have highest ratio of infectious disease as 25 take a while the Pneumonia ,Viral Fever , Dengue have average ratio below 20 %

40-50 age bracket have the minimal ratio of all disease under 20 %

50-60 people have the minimum ratio of typhoid under quarter-hour while Viral Fever have the most ratio as 23 %

age group 60-70 leading in both Dengue and Viral Fever age bracket 70-80 have minimum no of disease

Gender vs Disease

```r
gender_based_group = dict(tuple(disease_data.groupby('Gender')))
plt.figure(figsize=(12,7))
plt.hist(x=[gender_based_group["Male"]["DIAGNOSIS"],gender_based_group["Female"]["DIAGNOSIS"]],stacked=True, bins='auto',
label=["Male","Female"] )
plt.xticks( rotation=90)
plt.xlabel("Prevalent Diseases (2015-2019)")
plt.ylabel("No of reported cases")
plt.legend()
```

Analysis

From the above we will see that Pneumonia is highest trending disease among male and female.

Most common disease in Female

```r
plt.figure(figsize=(12,7))
sns.countplot(x= gender_based_group['Female']['DIAGNOSIS'],order=gender_based_group['Female']['DIAGNOSIS'].value_counts().index)
plt.ylabel("No of reported cases")
plt.xlabel("Prevalent Diseases (2015-2019)")
plt.xticks(rotation=90)
plt.show()
```
Analysis

From the above figures we are able to see that Pneumonia is commonest in Male.

In which year Female having the most common disease

```r
years = gender_based_group["Female"]["YEAR"].unique()
female_diseases_count = gender_based_group["Female"].groupby(['DIAGNOSIS'])["S. No"].count() top_4_female_diseaes = male_diseases_count.sort_values(ascending=False).iloc[:4].index.values print ("Top 4 most common diseases in Female for years :") [print(str(year)+" ", end = '') for year in years]
print("")
[print("'"+str(disease)+"' ", end = '') for disease in top_4_male_diseaes]
print("")
plt.figure(figsize=(12,7))
for disease in top_4_female_diseaes:
df= pd.DataFrame(columns=['year','no_of_cases'])
df = df.append( {'year':'2014','no_of_cases':gender_based_group["Female"].loc[ (gender_based_group["Female"]["DIAGNOSIS"]==disease)
& (gender_based_group["Female"]["YEAR"]==2015) ]["S. No"].count()}, ignore_index=True)
df = df.append( {'year':'2015','no_of_cases':gender_based_group["Female"].loc[ (gender_based_group["Female"]["DIAGNOSIS"]==disease)
& (gender_based_group["Female"]["YEAR"]==2016) ]["S. No"].count()}, ignore_index=True)
df = df.append( {'year':'2016','no_of_cases':gender_based_group["Female"].loc[ (gender_based_group["Female"]["DIAGNOSIS"]==disease)
& (gender_based_group["Female"]["YEAR"]==2017) ]["S. No"].count()}, ignore_index=True)
df = df.append( {'year':'2017','no_of_cases':gender_based_group["Female"].loc[ (gender_based_group["Female"]["DIAGNOSIS"]==disease)
& (gender_based_group["Female"]["YEAR"]==2018) ]["S. No"].count()}, ignore_index=True)
df = df.append( {'year':'2019','no_of_cases':gender_based_group["Female"].loc[ (gender_based_group["Female"]["DIAGNOSIS"]==disease)
&	(gender_based_group["Female"]["YEAR"]==2019) ]["S. No"].count()}, ignore_index=True) plt.plot( 'year', 'no_of_cases', data=df,marker='s' ,label='%s' % disease) plt.xlabel("Year")
plt.ylabel("No of reported Cases") plt.title("Female Trending b/w (2015-2019)") plt.legend()

```

Analysis

From the figure above we are able to see that:

• Typhoid and Pneumonia is trending in 2018
• Viral fever was trending on 2017
• Dengue was trending in 2016
In which year Male having the most common disease


```r
years = gender_based_group["Male"]["YEAR"].unique()
male_diseases_count = gender_based_group["Male"].groupby(['DIAGNOSIS'])["S. No"].count() top_4_male_diseaes = male_diseases_count.sort_values(ascending=False).iloc[:4].index.values print ("Top 4 most common diseases in Males for years :") [print(str(year)+" ", end = '') for year in years]
print("")
[print("'"+str(disease)+"' ", end = '') for disease in top_4_male_diseaes]
print("")
plt.figure(figsize=(12,7))
for disease in top_4_male_diseaes:
df= pd.DataFrame(columns=['year','no_of_cases'])
df = df.append( {'year':'2015','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2015) ]["S. No"].count()}, ignore_index=True)
df = df.append( {'year':'2016','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2016) ]["S. No"].count()}, ignore_index=True)
df = df.append( {'year':'2017','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2017) ]["S. No"].count()}, ignore_index=True)
df = df.append( {'year':'2018','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2018) ]["S. No"].count()}, ignore_index=True)
df = df.append( {'year':'2019','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2019) ]["S. No"].count()}, ignore_index=True)
plt.plot( 'year', 'no_of_cases', data=df,marker='s' ,label='%s' % disease)
plt.xlabel("Year")
plt.ylabel("No of reported Cases")
plt.title("M")
plt.legend()
plt.show()
```

Analysis

From the above figure we will see that Pneumonia and Typhoid is trending in 2018, viral fever was trending on 2017, Dengue was trending in 2016.

Male vs Female disease pattern

In this section we've tried to seek out the dependence between Gender and Diseases for this purpose we've taken top-4 most typical disease among male and feminine, Chi-Square test are performed between diseases and gender.


```r
top_4_dis_df = disease_data.loc[ (disease_data["DIAGNOSIS"] == "Pneumonia") | (disease_data["DIAGNOSIS"] == "Viral Fever") | (disease_data["DIAGNOSIS"] == "Typhoid fever") | (disease_data["DIAGNOSIS"] == "Dengue")] pd.crosstab(top_4_dis_df.Gender, top_4_dis_df.DIAGNOSIS, margins=True, margins_name="Total")
top_4_dis_matrix = pd.crosstab(top_4_dis_df.Gender, top_4_dis_df.DIAGNOSIS, margins=False).values print( top_4_dis_matrix )
stat, p, dof, expected = chi2_contingency(top_4_dis_matrix)
print('dof=%d' % dof)
print(expected)
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
print('Dependent (reject H0)')
else:
print('Independent (fail to reject H0)')
#	interpret p-value alpha = 1.0 - prob print('significance=%.3f, p=%.3f' % (alpha, p)) if p <= alpha:
print('Dependent (reject H0)')
else:
print('Independent (fail to reject H0)'

```

Analysis


It is obvious from the above pattern-square test analysis that there's no dependence between gender and diseases.

Most common disease analysis for male and female

```r
disease = 'Pneumonia'
df_male = pd.DataFrame(columns=['year','no_of_cases'])
df_male = df_male.append( {'year':'2015','no_of_cases':gender_based_group["Male"].loc[
(gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2015) ]["S. No"].count()}, ignore_index=True)
df_male = df_male.append( {'year':'2016','no_of_cases':gender_based_group["Male"].loc[
(gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2016) ]["S. No"].count()}, ignore_index=True)
df_male = df_male.append( {'year':'2017','no_of_cases':gender_based_group["Male"].loc[
(gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2017) ]["S. No"].count()}, ignore_index=True)
df_male = df_male.append( {'year':'2018','no_of_cases':gender_based_group["Male"].loc[
(gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2018) ]["S. No"].count()}, ignore_index=True)
df_male = df_male.append( {'year':'2019','no_of_cases':gender_based_group["Male"].loc[
(gender_based_group["Male"]["DIAGNOSIS"]==disease) & (gender_based_group["Male"]["YEAR"]==2019) ]["S. No"].count()}, ignore_index=True)
df_female = pd.DataFrame(columns=['year','no_of_cases'])
df_female = df_female.append( {'year':'2015','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["DIAGNOSIS"]==disease) & (gender_based_group["Female"]["YEAR"]==2015) ]["S. No"].count()}, ignore_index=True)
df_female = df_female.append( {'year':'2016','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["DIAGNOSIS"]==disease) & (gender_based_group["Female"]["YEAR"]==2016) ]["S. No"].count()}, ignore_index=True)
df_female = df_female.append( {'year':'2017','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["DIAGNOSIS"]==disease) & (gender_based_group["Female"]["YEAR"]==2017) ]["S. No"].count()}, ignore_index=True)
df_female = df_female.append( {'year':'2018','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["DIAGNOSIS"]==disease) & (gender_based_group["Female"]["YEAR"]==2018) ]["S. No"].count()}, ignore_index=True)
df_female = df_female.append( {'year':'2019','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["DIAGNOSIS"]==disease) & (gender_based_group["Female"]["YEAR"]==2019) ]["S. No"].count()}, ignore_index=True)
plt.figure(figsize=(12,7))
plt.plot( 'year', 'no_of_cases', data=df_male,marker='s' ,label='Male')
plt.plot( 'year', 'no_of_cases', data=df_female,marker='s' ,label='Female')
plt.xlabel("Year")
plt.ylabel("No of reported Cases")
plt.title("Male vs Female analysis for Pneumonia")
plt.legend()
```
Analysis

From the above figure we are able to see most vital (Pneumonia) year for Female & Male was 2018.

Crucial Year Analysis for Male and Female


```r
df_male = pd.DataFrame(columns=['year','no_of_cases'])
df_male = df_male.append( {'year':'2015','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["YEAR"]==2016)
]["S. No"].count()}, ignore_index=True)
df_male = df_male.append( {'year':'2016','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["YEAR"]==2017)
]["S. No"].count()}, ignore_index=True)
df_male = df_male.append( {'year':'2017','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["YEAR"]==2018)
]["S. No"].count()}, ignore_index=True)
df_male = df_male.append( {'year':'2018','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["YEAR"]==2019) ]["S. No"].count()}, ignore_index=True)
df_male = df_male.append( {'year':'2019','no_of_cases':gender_based_group["Male"].loc[ (gender_based_group["Male"]["YEAR"]==2020)]["S. No"].count()}, ignore_index=True)
df_female = pd.DataFrame(columns=['year','no_of_cases'])
df_female = df_female.append( {'year':'2015','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["YEAR"]==2015) ]["S. No"].count()}, ignore_index=True)
df_female = df_female.append( {'year':'2016+','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["YEAR"]==2016) ]["S. No"].count()}, ignore_index=True)
df_female = df_female.append( {'year':'2016','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["YEAR"]==2017) ]["S. No"].count()}, ignore_index=True)
df_female = df_female.append( {'year':'2018','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["YEAR"]==2018) ]["S. No"].count()}, ignore_index=True)
df_female = df_female.append( {'year':'2019','no_of_cases':gender_based_group["Female"].loc[
(gender_based_group["Female"]["YEAR"]==2019) ]["S. No"].count()}, ignore_index=True)
plt.figure(figsize=(12,7))
plt.plot( 'year', 'no_of_cases', data=df_male,marker='s' ,label='Male')
plt.plot( 'year', 'no_of_cases', data=df_female,marker='s' ,label='Female')
plt.xlabel("Year")
plt.ylabel("No of reported Cases")
plt.title("Yearly Female Vs Male analysis")
plt.legend()
```
Analysis

From the above figure we will depict that for female the foremost crucial year is 2017 and it get low in 2019 except for male it increase from 2015 to 2019.

Result and Conclusion

The dataset containing details of patients is from 2015 to 2019, with having almost 7000 unique records with relation to age and gender.

From the above analysis, we've got deduced that the foremost common disease in male/female is Pneumonia, and most vital year for male and feminine is 2017, 2018 and 2019 respectively. Congo Virus only occurs on Male and lymph gland syndrome on Female only.

At the age starting from 40 to 50 Male are more likely to chuck and starting from 60 to 70 Female are more likely to retch with these diseases.
because of this analysis we will understand that which disease is more likely to attack which gender at certain age and may help patient with rapid diagnosis.





























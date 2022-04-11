#!/usr/bin/env python
# coding: utf-8

# ## The Data
# 
# I have used a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
# 
# 
# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.
# 
# ### Problem Statement
# 
# Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off),  you have to build a model that can predict wether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. 
# 
# 
# ### Data Overview

# ----
# -----
# There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>LoanStatNew</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>loan_amnt</td>
#       <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>term</td>
#       <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>int_rate</td>
#       <td>Interest Rate on the loan</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>installment</td>
#       <td>The monthly payment owed by the borrower if the loan originates.</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>grade</td>
#       <td>LC assigned loan grade</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>sub_grade</td>
#       <td>LC assigned loan subgrade</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>emp_title</td>
#       <td>The job title supplied by the Borrower when applying for the loan.*</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>emp_length</td>
#       <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>home_ownership</td>
#       <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>annual_inc</td>
#       <td>The self-reported annual income provided by the borrower during registration.</td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td>verification_status</td>
#       <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td>issue_d</td>
#       <td>The month which the loan was funded</td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td>loan_status</td>
#       <td>Current status of the loan</td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td>purpose</td>
#       <td>A category provided by the borrower for the loan request.</td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td>title</td>
#       <td>The loan title provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td>zip_code</td>
#       <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td>addr_state</td>
#       <td>The state provided by the borrower in the loan application</td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td>dti</td>
#       <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>earliest_cr_line</td>
#       <td>The month the borrower's earliest reported credit line was opened</td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>open_acc</td>
#       <td>The number of open credit lines in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>20</th>
#       <td>pub_rec</td>
#       <td>Number of derogatory public records</td>
#     </tr>
#     <tr>
#       <th>21</th>
#       <td>revol_bal</td>
#       <td>Total credit revolving balance</td>
#     </tr>
#     <tr>
#       <th>22</th>
#       <td>revol_util</td>
#       <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
#     </tr>
#     <tr>
#       <th>23</th>
#       <td>total_acc</td>
#       <td>The total number of credit lines currently in the borrower's credit file</td>
#     </tr>
#     <tr>
#       <th>24</th>
#       <td>initial_list_status</td>
#       <td>The initial listing status of the loan. Possible values are – W, F</td>
#     </tr>
#     <tr>
#       <th>25</th>
#       <td>application_type</td>
#       <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
#     </tr>
#     <tr>
#       <th>26</th>
#       <td>mort_acc</td>
#       <td>Number of mortgage accounts.</td>
#     </tr>
#     <tr>
#       <th>27</th>
#       <td>pub_rec_bankruptcies</td>
#       <td>Number of public record bankruptcies</td>
#     </tr>
#   </tbody>
# </table>
# 
# ---
# ----
# 
# 
# Here, The "loan_status" column contains our label.

# ## Loading the data and importing necessary libraries.
# 

# In[1]:


import pandas as pd


# In[2]:


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[3]:


print(data_info.loc['revol_util']['Description'])


# In[15]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[5]:


feat_info('mort_acc')


# ## Loading the data and other imports

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('../DATA/lending_club_loan_two.csv')


# In[8]:


df.info()


# # Exploratory Data Analysis
# 
# **OVERALL GOAL: Get an understanding for which variables are important, view summary statistics, and visualize the data**
# 

# In[10]:


sns.countplot(x='loan_status',data=df)


# **Here, by looking at the label column i.e, "loan_status" it is clear that it is the case of unbalanced or inbalanced problem because we have a lot more entries of people that pay off their loans than we have people that did not pay back.**
# 

# In[6]:


plt.figure(figsize=(12,4))
sns.displot(df['loan_amnt'],kde=False,bins=30)
plt.xlim(0,45000)


# **Here we can see some spikes on 10k , 15k ,20k as these are standard loan amount that is taken.**

# **Exploring correlation between the continuous feature variables.

# In[14]:


df.corr()


# **Visualizing this correlation using a heatmap.
# 

# In[16]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)


# **Here we can notice almost perfect correlation with the "installment" feature. we want to explore this feature because we want   to make sure that we are not accidentally leaking data from our features into our label. so we always want make sure that       there is not a single feature that is perfect predictor of label because thats basically indicates that thats not really a       feature ,its probably just some duplicate information thats very similar to label.**

# In[18]:


feat_info('installment')


# In[19]:


feat_info('loan_amnt')


# In[20]:


sns.scatterplot(x='installment',y='loan_amnt',data=df,)


# **so, these two are highly correlated because the company generally uses some sort of formulas that is just a direct function of   the loan amount to figure out what the installment should be.**

# In[22]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# **Although both looks similar,but if loan amount is higer, we have a slight increase in the likelihood of it being charged off,
#   which also makes sense because its harder to pay back larger loans than it is in smaller loans.**

# In[24]:


df.groupby('loan_status')['loan_amnt'].describe()


# **This shows the quantative no behind the above box plot since, box plot is little harder to read.
#   So, here we can see charged off average price is little higher than the fully paid loan. This indicates that the averages of
#   the loans for people that arent able to pay them back are slightly higher than the averages for people that do pay off their     loans.**

# **Exploring the Grade and SubGrade columns that LendingClub attributes to the loans.**

# In[26]:


sorted(df['grade'].unique())


# In[27]:


sorted(df['sub_grade'].unique())


# In[28]:


sns.countplot(x='grade',data=df,hue='loan_status')


# In[30]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )


# In[32]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')


# **These grade are acoording to decresing order according to increasing alphabet .Grade A is best and then so on.
#   From this visulization , these worst grade categoriies it looks like the charge off rate is almost the same as the fully 
#   paid rate. so, it might be worth investigating if its even worth giving people loans, if we are going to grade them "G" or       "F".**
# 

# In[34]:


F_and_G = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(F_and_G['sub_grade'].unique())
sns.countplot(x='sub_grade',data=F_and_G,order = subgrade_order,hue='loan_status')


# **So here we can notice that for subgrade G5, the likelihood is almost same as fully paying off the loan versus being charged
#   off the loan.**

# **Right now, our label column is string and i want to change that essentially map to 1 or 0. so , creating a new column called     'load_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".**

# In[36]:


df['loan_status'].unique()


# In[37]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[38]:


df[['loan_repaid','loan_status']]


# **Creating a bar plot showing the correlation of the numeric features to the new loan_repaid column.

# In[40]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# **So, here we can see essentially the highest negative correlation(for "int_rate" column) with whether or not someone is going     to repay their loan, which a kind of make sense because maybe if you have an extremely high interest rate, then you are going   to find it harder to pay off that loan.**

# 
# # Data PreProcessing
# 
# **Goals of this section: Remove or fill any missing data. Removing unnecessary or repetitive features. Converting categorical     string features to dummy variables.**
# 
# 

# In[41]:


df.head()


# # Missing Data
# 
# **Exploring missing data columns.**

# In[7]:


len(df)


# In[8]:


missing_values=df.isnull().sum()
# actualno and columns how many point are missing


# In[46]:


missing_values


# **So, its look like mort_acc, emp title and emp length has most missing values then title and so on..**

# In[9]:


percentage_missing = 100* df.isnull().sum()/len(df)


# In[49]:


percentage_missing


# In[11]:


total_missing_values=pd.concat([missing_values,percentage_missing],axis=1,keys=["missing","percent"])


# In[13]:


total_missing_values


# **So, we will focus on mort_acc bcoz we cannot drop 10 percent data and it should be fine to drop some of these which has very     less missing values.**

# **Let's examine emp_title and emp_length to see whether it will be okay to drop them.

# In[51]:


feat_info('emp_title')
print('\n')
feat_info('emp_length')


# In[53]:


df['emp_title'].nunique()


# In[54]:


df['emp_title'].value_counts()


# **So, we can see there is a ton of unique employment title,in fact there is 173105 unqiue employment title and our dataset has     around 400000 values. So, its look like there is almost half of that are all unique employment titles and it is impossible to   convert all this to a dummy variable feature. So, its better to remove that emp_title column.**

# In[56]:


df = df.drop('emp_title',axis=1)


# In[58]:


sorted(df['emp_length'].dropna().unique())


# In[59]:


emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']


# In[60]:


plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order)


# **Quite majority of people have been working in their employment for ten plus years, which make sense, if you are taking a         loan,you are very likely to have a job otherwise you wont be able to pay it back.**

# In[62]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')


# **Now, what is interested here is the relationship between fully paid off versus charged off per employment length.
#   So, if there is an extreme difference in one of these categories of fuly paid off versus charged off, for example, maybe if 
#   someone worked less than 1 year, everyone there charge off their loan didnt pay it back then its a very important feature. 
#   But, if the ratio of this blue bar to this orange bar is essentially the same across all these employment length categories, 
#   then this isnt a very informative feature.
#   So, lets figure out the ratio betwen the fully paid vs charged off people per category of employment length.**

# In[64]:


emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']


# In[65]:


emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']


# In[66]:


emp_len = emp_co/emp_fp


# In[67]:


emp_len


# In[68]:


emp_len.plot(kind='bar')


# **If we see here, across the extremes , it looks exactly similar. So, this particular feature emp_length doesnt actualy have       some extreme differences on the charge off rates. So, it looks like what actual employment length you have, if you were to       pick someone then about approx 20 percent of them are going have not paid back their loans and we can see this in bar plot       also that since all these bars are almost the same height there really isnt that much information or differentiation between     the emp_length column, which is kind of surprising.
#   But we can see here the main differenece is that people who work for 10 years are having a slightly smaller charge of rate       than people who work less than 1  year but that difference is not extreme enough to really validate keeping this feature.
#   So, since they are so extremely similar across all employments, so we will drop that "emp_length" columns.**

# In[70]:


df = df.drop('emp_length',axis=1)


# **Checking the DataFrame to see what feature columns still have missing data.**

# In[71]:


df.isnull().sum()


# In[73]:


df['purpose'].head(10)


# In[74]:


df['title'].head(10)


# **The title column is simply a string subcategory/description of the purpose column.So, dropping the title column.**

# In[76]:


df = df.drop('title',axis=1)


# In[78]:


feat_info('mort_acc')


# In[80]:


df['mort_acc'].value_counts()


# **Seems like majority of people have zero mortage accounts and its almost 25 percent of our data.**

# **Now, 10percent of data is missing in "mort_acc" and we cannot drop the rows otherwise we are dropping 10 percent of our data     or we can drop that actual feauture depending upon its importance in this dataset and there is no right answer for that.
#   so this is one of the hardest things to do with missing data is to figure out a reasonable way to try to fill it in.
#   so, one approach is to try to figure out which of these other features that we have all the information that correlates
#   highly with this "mort_acc" and we see if we can use that to fill in our information.**
#   
#   **Generally, if the data set is large we will drop that feature or if data set is small and we have to fill the missing data
#     and in general- for categorical data we fill with mode and for numerical data we fill it with mean or median values.**

# In[81]:


print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()


# **Looks like the "total_acc" feature have some positive correlation with the "mort_acc". 
#   So, We can group the dataframe by the "total_acc" and calculate the mean value for the "mort_acc" per "total_acc" entry.**

# In[ ]:





# In[82]:


print("Mean of mort_acc column per total_acc")
df.groupby('total_acc').mean()['mort_acc']


# **Now, filling the missing "mort_acc" values based on their "total_acc" value. If the "mort_acc" is missing, then it is filled     by the missing value with the mean value corresponding to its "total_acc" value from the Series that is created above. In this   we can use an .apply() method with two columns.**

# In[84]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[85]:


total_acc_avg[2.0]


# In[86]:


def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    SO, looking up avearge value for that mortage account based on their total account
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[87]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
# basically here we are applying a function to two column of a datrame.


# In[88]:


df.isnull().sum()


# **Here, "revol_util" and the "pub_rec_bankruptcies" have missing data points, but they account for less than 0.5% of the total     data. So,removing the rows that are missing those values in those columns with dropna().**

# In[90]:


df = df.dropna()


# In[91]:


df.isnull().sum()


# ## Categorical Variables and Dummy Variables
# 
# **We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.**

# In[93]:


df.select_dtypes(['object']).columns


# **Now, here i will go through each of these categorical column and will keep some of them which will be useful and i will just
#   remove some of the column.**

# 
# **Let's now go through all the string features one by one to see what we can do with them.**
# 
# ###  "term" feature
# 

# In[121]:


df["term"].head


# In[120]:


df['term'].value_counts()


# **So, its look like a binary column - either 36 or 60 months, which means i have a couple of options here and  because its also 
#   numeric i could convert this to be either 36 as a numeric integer or 60 as  a numeric integer or can do normal mapping.**
# 

# In[122]:


df['term'] = df['term'].apply(lambda term: int(term[:3]))


# ### "grade" feature
# 
# **Since, "grade" is part of "sub_grade", so i will just drop the "grade" feature because "subgrade" has already have that         information**

# In[124]:


df = df.drop('grade',axis=1)


# **Now, converting the subgrade into dummy variables.**

# In[127]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[101]:


df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)


# In[102]:


df.columns


# In[103]:


df.select_dtypes(['object']).columns


# ### verification_status, application_type,initial_list_status,purpose
# **Converting these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and   concatenating them with the original dataframe.**

# In[105]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[ ]:





# ### home_ownership feature

# In[107]:


df['home_ownership'].value_counts()


# **Converting these to dummy variables, and replacing NONE and ANY with OTHER, so that it have just 4 categories, MORTGAGE, RENT, OWN, OTHER.**

# In[109]:


df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)


# ### address feature
# **Here, i am extracting zip code from the address in the data set. Creating a column called 'zip_code' that extracts the zip       code from the address column.**

# In[133]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# **Now, converting this zip_code column into dummy variables.**

# In[134]:


dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


# ### issue_d feature

# In[16]:


feat_info("issue_d")


# **Here, we don't know beforehand whether or not a loan would be issued when using this model, so in theory this model should not have any issue_date column otherwise this would be data leakage, So, dropping this feature.**

# In[136]:


df = df.drop('issue_d',axis=1)


# ### earliest_cr_line feature
# 
# **This appears to be a historical time stamp feature. So, Extracting the year from this feature using a .apply function, then     converting it to a numeric feature. Setting this new data to a feature column called 'earliest_cr_year'.Then dropping the       old earliest_cr_line feature.**

# In[137]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)


# In[138]:


df.select_dtypes(['object']).columns


# **We have converted all the necessary categorical column into its dummy variable.**

# ## Train Test Split

# **Importing train_test_split from sklearn.**

# In[139]:


from sklearn.model_selection import train_test_split


# **Now, i will drop the load_status column that was created earlier, since its a duplicate of the loan_repaid column. I'll use the loan_repaid column since its already in 0s and 1s.**

# In[140]:


df = df.drop('loan_status',axis=1)


# **Setting X and y variables to the .values of the features and label.**

# In[141]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# **Performing a train/test split with test_size=0.2 and a random_state of 101.**

# In[143]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# ## Normalizing the Data
# 
# **Here, i have used MinMaxScaler to normalize the feature data X_train and X_test.**

# In[144]:


from sklearn.preprocessing import MinMaxScaler


# In[145]:


scaler = MinMaxScaler()


# In[146]:


X_train = scaler.fit_transform(X_train)


# In[147]:


X_test = scaler.transform(X_test)


# # Creating the Model
# 
# **Here, i will be using following model - ANN model using keras, logistic regression model , decision tree and random forest       model.**

# In[148]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


# In[149]:


model = Sequential()


# In[153]:


X_train.shape


# **Here, we have 78 column feature. So, what i am going to do is creating my first layer match with 78 neuron.
#   and reducing no of neuron to half in each layer.**

# In[150]:


model = Sequential()

# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid')) 

# Compiling model
model.compile(loss='binary_crossentropy', optimizer='adam')


# **Fiting the model to the training data.**

# In[151]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )


# # Evaluating Model Performance.
# 
# **Plotting out the validation loss versus the training loss.**

# In[156]:


losses = pd.DataFrame(model.history.history)


# In[157]:


losses[['loss','val_loss']].plot()


# **Now, Creating predictions from the X_test and displaying it in the form of classification report and confusion matrix.**

# In[158]:


from sklearn.metrics import classification_report,confusion_matrix


# In[160]:


predictions = (model.predict(X_test) > 0.5)*1


# In[166]:


print(classification_report(y_test,predictions))


# In[162]:


confusion_matrix(y_test,predictions)


# # Checking the model -
# 
#  **Given the customer below, would you offer this person a loan?**

# In[168]:


import random
random.seed(101)
random_index = random.randint(0,len(df))

#the random seed allows us to reproduce the same random results

new_customer = df.drop('loan_repaid',axis=1).iloc[random_index]
new_customer


# In[165]:


random_ind


# In[169]:


new_customer


# In[170]:


new_customer.values


# In[171]:


new_customer.values.reshape(1,78)


# **Now before using this data i have to scale because this model is trained on scale data.**

# In[172]:


new_customer=scaler.transform(new_customer.values.reshape(1,78))


# In[173]:


new_customer


# In[177]:


predictions = (model.predict(new_customer) > 0.5)*1


# In[178]:


predictions


# **Now Cross Checking time - did this person actually end up paying back their loan?**

# In[180]:


df.iloc[random_index]['loan_repaid']


# **So, this model is performing well. Now, i will find the confusion matrix and evaluate the result using logistic regression and random forest model and then will conclude to the fonal conclusion.**
# 

# # Model evaluation by logistic regression

# In[201]:


from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()


# In[202]:


from sklearn.model_selection import train_test_split


# In[203]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# In[204]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[205]:


from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)


# In[206]:


from sklearn.linear_model import LogisticRegression


# In[207]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[208]:


predictions1 = logmodel.predict(X_test)


# In[209]:


predictions1


# In[210]:


from sklearn.metrics import classification_report,confusion_matrix


# In[211]:


print(classification_report(y_test,predictions1))
print("\n")
print(confusion_matrix(y_test,predictions1))


# # Model evaluation by Decision tree and Random forest

# In[215]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[216]:


from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)


# In[217]:


from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred=dtree.predict(X_test)


# In[218]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,pred))
print("\n")
print(confusion_matrix(y_test,pred))


# 
# #  Using random forest

# In[223]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[224]:


from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)


# In[225]:


rfc.fit(X_train,y_train)
pred2=rfc.predict(X_test)


# In[226]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,pred2))
print("\n")
print(confusion_matrix(y_test,pred2))


# # CONCLUSION

# **After running this dataset through different -different model ,let's summarize the classification report.**
# 
# **Accuracy - We are getting at around 90 percent accuracy, thats nice.**
# 
# **Now, if we recall that in the beginning of this project, what i did is i first analyzed the actual label itself and recall       that its an imbalanced label.**
#   
# **There is a lot more fully paid loans than there are charged off loans and infact, if we were to take a look at a model that     simply recalled back any loan as being fully paid, it would actually still be pretty accurate.**

# In[181]:


df["loan_repaid"].value_counts()


# In[182]:


317696/len(df)


# **Notice here that what this is indicating is that 80 percent of my points were already being predicted as loan repaid,
#   which means if i created a very simple model that simply said any loan will be repaid, i would be 80 percent accurate.**
#   
# **So, we can't be so sure by the model if it returns 80 percent accuracy beacuse it also depends on the type of dataset we have   used to create that model, either it was balanced or imbalanced dataset.** 
# 
# **So, getting 89 percent accuracy of this model is  OK but its not absolutely fantastic because of our imbalanced dataset.**
#  
# **The actual metrics we want to look at is our precision, recall and f1 score, and specially f1 score beacause it is harmonic   mean of reacll and precision.**
# 
# **So, the true notification of whether or not this model is doing well is this F1 score on this zero class whch is 0.61.
#   and is it good or bad that really depends on the entire context of the situation, whether we have a model that already
#   attempts to predict this and what its F1 score. so we need a lot more context to decide whether or not this recall on the
#   F1 score are good enough.**
# 
# 
# **Although, I can say that this accuracy is better than just kind of  default guess which should be 80 percent.So this model is 
#  definitely better than just kind of a random guess or a straight guess. So this model is performing much better.**
#  
# **At random guess we wil get 50 perceny acuracy and a straight guess of always being repaid with 80 percent accuracy and 
#  this model is getting 89 percent accuracy. So, it is performing better than both a random guess and a straight loan paid 
#  return.**
# 
# 
# **So, by concluding all the above points, overall, we can see that this model is learning something from this dataset.**

# # THANKS

# In[ ]:





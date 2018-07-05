
# coding: utf-8

# In[119]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[120]:


alldata = pd.read_csv("AAPL.csv",encoding='utf-8-sig') #Using pandas to read into the csv file and store it as a variable
print(alldata)  


# In[121]:


alldata["Date"]= pd.to_datetime(alldata["Date"])
alldata.index = alldata["Date"]
del alldata["Date"]


# In[122]:


def dist_summary(df):
   ts = pd.Series(df)
   ts.plot()
   plt.show()


# In[123]:


dist_summary(alldata['Adj Close'])


# In[124]:


#also to judge the accuracy of your model you compare difference of your prediction of Y(price) using your X (input features) <- this is what you plot, should not see any pattern for this
#plot the histogram of the residual of histogram -> should see a guassian distro .. if you see these you can say your model is correct
#residuals should be close to 0


# In[125]:


def norm(df):
    num_cols = df.select_dtypes(include=[np.number]).copy() 
    df_norm = ((num_cols-num_cols.min())/(num_cols.max()-num_cols.min())) 
    return df_norm


# In[126]:


normdata = norm(alldata)
print (normdata)


# In[127]:


#To calculate a moving 5 day average 
normdata["Moving Average"] = normdata["Adj Close"].rolling(5).mean()


# In[128]:


from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(normdata, test_size=0.25)
print("Training size: {}, Testing size: {}".format(len(trainset), len(testset)))
print("Samples: {} Features: {}".format(*trainset.shape))


# In[129]:


#Dropping NaN/Empty Cols
cleandata = normdata.dropna()
print (cleandata)


# In[130]:


from sklearn import svm, feature_selection, linear_model
df = cleandata.select_dtypes(include=[np.number]).copy()
feature_cols = df.columns.values.tolist()
feature_cols.remove('Adj Close')  #Remove adj close since its the value we want to predict based on input cols
XO = df[feature_cols]
YO = df['Adj Close']
estimator = svm.SVR(kernel="linear")
selector = feature_selection.RFE(estimator, 6, step=1)
selector = selector.fit(XO, YO)
select_features = np.array(feature_cols)[selector.ranking_ == 1].tolist()
print(select_features)


# In[138]:


X = df[select_features]
Y = df['Adj Close']
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25) 
lm = linear_model.LinearRegression()
lm.fit(trainX, trainY)
# Inspect the calculated model equations
print("Y-axis intercept {}".format(lm.intercept_))
print("Weight coefficients:")
for feat, coef in zip(select_features, lm.coef_):
    print(" {:>20}: {}".format(feat, coef))
# The value of R^2
print("R squared for the training data is {}".format(lm.score(trainX, trainY))) 
print("Score against test data: {}".format(lm.score(testX, testY)))


# In[139]:


pred_trainY = lm.predict(trainX)
plt.figure(figsize=(14, 8))
plt.plot(trainY, pred_trainY, 'o')  
plt.xlabel('Actual Prices')
plt.ylabel="Predicted Prices"
plt.title="Plot of predicted vs actual prices"
plt.show()


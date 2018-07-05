
# coding: utf-8

# # Importing the libraries and raw dataset

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import pandas as pd


# In[2]:


alldata = pd.read_csv("PremierLeague1718.csv",encoding='utf-8-sig') #Using pandas to read into the csv file and store it as a variable
#print(alldata)                                #Used to print out the raw dataset without trimming irrelvant/repeat/null data 


# # Removing missing data

# In[3]:


maxdata = alldata.dropna(axis=1,how='all')
print (alldata)


# # Extracting relevant columns

# In[4]:


reqdata = maxdata[["Date","HomeTeam","AwayTeam","FTHG","FTAG","HST","AST","HS","AS"]]
print(reqdata)


# # Describing the data

# In[5]:


reqdata.describe()


# # Storing games played as DataFrames

# In[6]:


mchg = reqdata.loc[reqdata['HomeTeam'] == 'Man City'] #Games where Manchester City was a Home Team
print(mchg)


# In[7]:


mcag = reqdata.loc[reqdata['AwayTeam'] == 'Man City'] #Games where Manchester City was an Away Team
print(mcag)


# In[8]:


muhg = reqdata.loc[reqdata['HomeTeam'] == 'Man United'] #Games where Manchester United was a Home Team
print(muhg)


# In[9]:


muag = reqdata.loc[reqdata['AwayTeam'] == 'Man United'] #Games where Manchester United was an Away Team
print(muag)


# # Manchester City’s away offense vs. Manchester United’s home defense

# In[10]:


#Generating a histogram displaying Manchester City's Away Performance
fig1 = plt.figure(1)
plt.subplot(1,1,1)
plt.hist(mcag["FTAG"].as_matrix(), bins=20, alpha=0.7, color="skyblue") #Here we are plotting MC's goals as offence
plt.legend(['Man City Away'])    
plt.ylabel("Num. of games played ")
plt.xlabel("Goals Scored")
plt.show()


# In[11]:


#Generating a histogram displaying Manchester United's Home Performance
fig1 = plt.figure(1)
plt.subplot(1,1,1)
plt.hist(muhg["FTAG"].as_matrix(), bins=20, alpha=0.7, color="red")  #For defence we plot the goals MU concede home
plt.legend(['Man United Home'])
    
plt.xlabel("Goals Conceded")
plt.ylabel("Num. of games played ")
plt.show()


# # Manchester City’s home offense vs. Manchester United’s away defense

# In[12]:


#Generating a histogram displaying Manchester City's Home Performance
fig1 = plt.figure(1)
plt.subplot(1,1,1)
plt.hist(mchg["FTHG"].as_matrix(), bins=20, alpha=0.7, color="skyblue") #Plotting goals MC scored playing home
plt.legend(['Man City Home'])    
plt.ylabel("Num. of games played ")
plt.xlabel("Goals scored")
plt.show()


# In[13]:


#Generating a histogram displaying Manchester United's Away Performance
fig1 = plt.figure(1)
plt.subplot(1,1,1)
plt.hist(muag["FTHG"].as_matrix(), bins=20, alpha=0.7, color="red") #Plotting the number of goals MU concede while away
plt.legend(['Man United Away'])
plt.xlabel("Goals conceded")
plt.ylabel("Num. of games played ")
plt.show()


# # Simulating Man City Home vs Man United Away

# In[14]:


#Generate matches for each team
def sim_poisson(nums, mean):
    dist = nr.poisson(lam = mean, size = nums)
    return list(dist)


# In[22]:


gennum = 500      #Storing the number of games to simulate between the two teams
mchome = sim_poisson(gennum, mchg["FTHG"].mean())      #Passing the number of games played and the mean number of goals the teams score as parameters for simulating a poisson distribution
muaway = sim_poisson(gennum, muag["FTAG"].mean())

combined = zip(mchome, muaway)

#[print(x) for x in combined]            #uncomment to display the result of each game simulated

MUWin = 0   #Counters for determining which team wins
MCWin = 0

for x,y in combined:                     #Logic for determining which team won based on the number of goals scored (x = MC, y = MU)
    if x > y:
        #print("MC Win")
        MCWin+=1
    elif y > x:
        #print("MU Win")
        MUWin+=1
        #print("Draw")
        

print("Games Man City Won HOME: ", MCWin)
print("Games Man United Won AWAY: ", MUWin)
print("Draws: ", abs(gennum-(MUWin+MCWin)))


# # Simulating Man City Away vs Man United Home 

# In[16]:


#also need to compare their performance by taking probability of games won as a percentage of all games played.


# In[31]:


gennum = 500      #Storing the number of games to simulate between the two teams
mcaway = sim_poisson(gennum, mcag["FTAG"].mean()) #Passing the number of games played and the mean number of goals the teams score as parameters for simulating a poisson distribution
muhome = sim_poisson(gennum, muhg["FTHG"].mean())

combined = zip(mcaway, muhome)

#[print(x) for x in combined]             #uncomment to display the result of each game simulated

MUWin = 0     #Counters for games won
MCWin = 0

for x,y in combined:   #Logic for determining which team won based on the number of goals scored (x = MC, y = MU)
    if x > y:
        #print("MC Win")
        MCWin+=1
    elif y > x:
        #print("MU Win")
        MUWin+=1
        #print("Draw")
        
 
print("Games Man City Won AWAY: ", MCWin)
print("Games Man United Won HOME: ", MUWin) 
print("Draws: ", abs(gennum-(MUWin+MCWin)))


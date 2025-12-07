"""
Linear regression
==============

We are going to look at the relationship between age and
minutes played. Start by watching the video a

..  youtube:: TnOqoeVPnXE
   :width: 640
   :height: 349

Either work through the code at the same time as watching or afterwards.
"""

#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

##############################################################################
# Opening data
# ----------------------------
# In this example we use data downloaded from `FBref <https://fbref.com/en/comps/12/2021-2022/stats/2021-2022-La-Liga-Stats>`_ on players in La Liga.
# We just use the age and minutes played columns.
# And we only take the first 20 observations, to help visualise the process.
# Download `playerstats.csv <https://github.com/soccermatics/Soccermatics/blob/main/course/lessons/lesson2/playerstats.csv>`_
# your working directory.

num_obs=20
laliga_df=pd.read_csv("playerstats.csv",delimiter=',')
minutes_model = pd.DataFrame()
minutes_model = minutes_model.assign(minutes=laliga_df['Min'][0:num_obs])
minutes_model = minutes_model.assign(age=laliga_df['Age'][0:num_obs])

# Make an age squared column so we can fir polynomial model.
minutes_model = minutes_model.assign(age_squared=np.power(laliga_df['Age'][0:num_obs],2))


##############################################################################
# Plotting the data
# ----------------------------
# Start by plotting the data.

fig,ax=plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker= '.', markersize= 10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15,40))
plt.ylim((0,3000))
plt.show()



##############################################################################
#Fitting the model
#----------------------------
#We are going to begin by doing a  straight line linear regression
# .. math::
#
#    y = b_0 + b_1 x
#
#A straight line relationship between minutes played and age.

model_fit=smf.ols(formula='minutes  ~ age   ', data=minutes_model).fit()
print(model_fit.summary())        
b=model_fit.params

##############################################################################
# Comparing the fit 
# ----------------------------
#We now use the fit to plot a line through the data.
# .. math::
#
#    y = b_0 + b_1 x
#
#where the parameters are estimated from the model fit.


#First plot the data as previously
fig,ax=plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker= '.', markersize= 10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15,40))
plt.ylim((0,3000))

#Now create the line through the data
x=np.arange(40,step=1)
y= np.mean(minutes_model['minutes'])*np.ones(40)
ax.plot(x, y, color='black')

#Show distances to line for each point
for i,a in enumerate(minutes_model['age']):
    ax.plot([a,a],[minutes_model['minutes'][i], np.mean(minutes_model['minutes']) ], color='red')
plt.show()

##############################################################################
# A model including squared terms
# ----------------------------
#We now fit the quadratic model
# .. math::
#
#    y = b_0 + b_1 x + b_2 x^2
#
#estimating the parameters from the data.

# First fit the model
model_fit=smf.ols(formula='minutes  ~ age + age_squared  ', data=minutes_model).fit()
print(model_fit.summary())        
b=model_fit.params

# Compare the fit 
fig,ax=plt.subplots(num=1)
ax.plot(minutes_model['age'], minutes_model['minutes'], linestyle='none', marker= '.', markersize= 10, color='blue')
ax.set_ylabel('Minutes played')
ax.set_xlabel('Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim((15,40))
plt.ylim((0,3000))
x=np.arange(40,step=1)
y= b[0] + b[1]*x + b[2]*x*x
ax.plot(x, y, color='black')

for i,a in enumerate(minutes_model['age']):
    ax.plot([a,a],[minutes_model['minutes'][i], b[0] + b[1]*a + b[2]*a*a], color='red')
plt.show()

##############################################################################
# Now try with all data points
# ----------------------------
# 1) Refit the model with all data points
#
# 2) Try adding a cubic term
#
# 3) Think about how well the model works. What are the limitations?


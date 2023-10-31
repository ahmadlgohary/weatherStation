import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt

df = pd.read_csv("weatherstats_toronto_normal_daily.csv")

#Y values
rain    =   df['rain_v']
snow    =   df['snow_v']
#X values
maxdew  =   df['max_dew_point_v']
mindew  =   df['min_dew_point_v']
maxhum  =   df['max_relative_humidity_v']    
minhum  =   df['min_relative_humidity_v']
maxtemp =   df['max_temperature_v']
mintemp =   df['min_temperature_v']  
maxwind =   df['max_wind_speed_v']
minwind =   df['min_wind_speed_v']
precip  =   df['precipitation_v']

cols = [maxdew, mindew, maxhum, minhum, maxtemp, mintemp, maxwind, minwind, precip]

avtemp = (maxtemp + mintemp )/len(maxtemp)
avdew = (maxdew + mindew )/len(maxdew)
avhum = (maxhum + minhum )/len(maxhum)
avwind = (maxwind + minwind )/len(maxwind)

avtemp.name = "Average Temperature"
avdew.name = "Average Dew"
avhum.name = "Average Humidity"
avwind.name = "Average Wind Speed"
precip.name = "Percipitation"

cols = [avdew,avtemp,avhum,avwind,precip]

#logistic regression rain or no rain or snow
def correlation(x,y):
    avx  = np.mean(x)
    avy  = np.mean(y)
    stx  = np.std(x)
    stdy = np.std(y)
    p = np.sum((x-avx)*(y-avy))/(stx*stdy*len(x))
    if p < 0.5 and p > -0.5:
        print(f"{x.name} has {round(p,3)} correlation")

def prob(x, theta):
    p = 1/(1+m.exp(-np.linalg.transpose(x)*theta))
    return p

def cost(y, x, theta):
    j = 0
    for i in range(len(df)):
        if y == 1:
            j += - m.log(prob(x,theta))
        elif y == 0:
            j += m.log(1-prob(x,theta))
        else:
            j += (1-y)*m.log(1-prob(x,theta))- y*m.log(prob(x,theta)) 
    return j


#print(df.columns)
#for i in cols:
#    correlation(i,rain)

"""
for col in cols:
    plt.title('rain')
    plt.scatter(i,rain)
    plt.show()
    plt.scatter(i,snow)
    plt.title('snow')
    plt.show()
"""
#add logistic regression and softmax 
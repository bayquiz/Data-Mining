# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:16:48 2020

@author: bayqu
"""

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

#read in the two files *********************************
season_original = pd.read_csv("season78.csv")
draft_original = pd.read_csv("draft78.csv")

print(season_original.info)
print(draft_original.info)

#add index field***************************************
season_original['index'] = pd.Series(range(0, 15313))
season_original.head

draft_original['index'] = pd.Series(range(0, 3642))
draft_original.head

#ISAAC noticed NaN values in [Yrs] ***********************************
draft_original["Yrs"]= draft_original["Yrs"].replace({np.NaN:0})
draft_original.head 
#confirmed that it was replaced by 0

#looking for misleading values in DRAFT **********************************
plt.figure()
plt.hist(draft_original['Pick'])
plt.title("Draft - Picks")
plt.xlabel("Draft Picks");plt.ylabel("Frequency");plt.show()
print("There are outliers in Draft Pick.. need to be corrected!")


plt.figure()
plt.hist(draft_original['Draft'])
plt.title("Draft - Draft Year")
plt.xlabel("Draft Season");plt.ylabel("Frequency / # of records");plt.show()
print("found major outliers in draft years... too many values in drafts prior to 1990!")

#Looking for misleading values in SEASON
plt.figure()
plt.hist(season_original['Season'])
plt.title("Season- Seasons")
plt.xlabel("Season / Year");plt.ylabel("Frequency / # of records");plt.show()
print("More records and stats become available in modern era...")

plt.figure()
plt.hist(season_original['WS'])
plt.title("Season - WS")
plt.xlabel("Winshares");plt.ylabel("Frequency");plt.show()
print("Easy to see the avg and way above average WS values")
print("We could bin winshare values, bad, avg, good, amazing")


#identify outliers in SEASON - WS
season_original["WS_z"] = stats.zscore(season_original['WS'])
#confirming the new column was added
season_original.head

season_outliers = season_original.query('WS_z > 3 | WS_z < -3')
season_sort = season_original.sort_values(['WS_z'], ascending=False)
print(season_sort[['Player', 'WS', 'Season', 'WS_z']].head(n=15))
print(season_sort[['Player', 'WS', 'Season', 'WS_z']].tail(n=15))
print("We only have outliers in positive direction > 3. None < -3")
print("We need to keep all values")

#identify outliers in DRAFT - PICK **************************
#NOT SURE IF EITHER OF THESE TWO ARE NECESSARY!!??
pick_count = draft_original["Pick"].value_counts()
pick_count.to_frame()

draft_original['Pick_z'] = stats.zscore(draft_original['Pick'])
draft_original.head
draft_outliers = draft_original.query('Pick_z > 3 | Pick_z < -3')
draft_sort = draft_original.sort_values(['Pick_z'], ascending=False)
print(draft_sort[[ 'Draft', 'Pick', 'Pick_z']].head(n=15))
print(draft_sort[['Draft', 'Pick', 'Pick_z']].tail(n=15))
print("We have no z values > 3 or < -3")
#PROBABLY NEED TO BIN THESE AND CHECK AGAIN

count = draft_original["Draft"].value_counts()
#this lets us know how many picks were in each seasons draft
count.to_frame()

#identify outliers in DRAFT- DRAFT
draft_original['Draft_z'] = stats.zscore(draft_original['Draft'])
draft_original.head
draft_outlier2 = draft_original.query('Draft_z > 3 | Draft_z < -3')
draft_sort2 = draft_original.sort_values(['Draft_z'], ascending=False)
print(draft_sort2[[ 'Draft', 'Pick', 'Draft_z']].head(n=15))
print(draft_sort2[['Draft', 'Draft_z']].tail(n=15))
print("We have no z values > 3 or < -3")
# Modern drafts only go 2 rounds .. roughly 60 picks
 

# fixing the Draft['Pick'] to only have 2 rounds (60) this helps normalize scale
picks_filtered = draft_original[draft_original['Pick'] > 60]
draft_original.drop(picks_filtered.index, inplace = True)
print(draft_original.head())
#this deletes the picks over 60... which arent needed and contain many NaN or 0 values

firstRound = draft_original[draft_original["Pick"] <30]
secondRound = draft_original.query('Pick > 29 & Pick < 61' )
lottery = draft_original[draft_original["Pick"] < 15]



#Robert - Binning Based on Predictive Value ************************************
season_original['Season_binned'] = pd.cut(x = season_original['Season'], bins = [1979,1979.01, 1990, 2000, 2010, 2017], labels = ["70's", "80's", "90's", "2000's", "2010's"], right = False)
draft_original['Draft_binned'] = pd.cut(x = draft_original['Draft'], bins = [1978,1979.01, 1990, 2000, 2010, 2016], labels = ["70's", "80's", "90's", "2000's", "2010's"], right = False)

seasontab = pd.crosstab(season_original['Season_binned'], season_original['Season_binned'])
seasontab.plot(kind = 'bar', stacked = True, title = 'Binning with Season')

drafttab = pd.crosstab(draft_original['Draft_binned'], draft_original['Draft_binned'])
drafttab.plot(kind = 'bar', stacked = True, title = 'Binning with Draft')

#we could look at DECADE vs AVG WS?
avgWS = season_original["WS"].mean()



#check with another histogram
plt.figure()
plt.hist(draft_original['Pick'])
plt.title("Draft- Pick after fixed for 2 rounds")
plt.xlabel("Picks");plt.ylabel("Frequency");plt.show()

plt.figure()
plt.hist(draft_original['Draft'])
plt.title("Draft- Year after fixed for 2 rounds")
plt.xlabel("Draft Year");plt.ylabel("Frequency / # of records");plt.show()

#save the changes
season_original.to_csv('new_season.csv', index=False)
draft_original.to_csv('new_draft.csv', index=False)













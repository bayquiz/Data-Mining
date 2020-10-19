# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
from sklearn import metrics

import matplotlib
matplotlib.style.use('ggplot')
from subprocess import check_output
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Remove future warning for set_value()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# DATA PREPARATION PHASE
#read in the two files
season_original = pd.read_csv("season78.csv")
draft_original = pd.read_csv("draft78.csv")

#print(season_original.info)
#print(draft_original.info)

#add index field
season_original['index'] = pd.Series(range(0, 15313))
#season_original.head

draft_original['index'] = pd.Series(range(0, 3642))
#draft_original.head
  
# Fill in blank cells for Yrs with 0s     
# IS     
draft_original["Yrs"]=draft_original["Yrs"].replace({np.NaN:0})
draft_original.to_csv("draft78.csv", sep=',', encoding='utf-8', index = False)

#looking for misleading values in DRAFT
# BQ
"""
plt.figure()
plt.hist(draft_original['Pick'], ec="black")
plt.title("Draft Original - Picks")
plt.xlabel("Number of draft picks");plt.show()

plt.figure()
plt.hist(draft_original['Draft'], ec="black")
plt.title("Draft Original - Draft Year")
plt.show()
#found major outliers in draft years... too many values in drafts
#prior to 1990

#Looking for misleading values in SEASON
# BQ
plt.figure()
plt.hist(season_original['Season'], ec="black")
plt.title("Season Original - Seasons")
plt.show()

plt.figure()
plt.hist(season_original['WS'], ec="black")
plt.title("Season Original - WS")
plt.xlabel("Winshares");plt.show()
#season_original['WS_z'].plot(kind='hist')
"""

#identify outliers in SEASON - WS
season_original["WS_z"] = stats.zscore(season_original['WS'])
#confirming the new column was added
# BQ
season_original.head

season_outliers = season_original.query('WS_z > 3 | WS_z < -3')
season_sort = season_original.sort_values(['WS_z'], ascending=False)
print(season_sort[['Player', 'WS', 'Season', 'WS_z']].head(n=15))
print(season_sort[['Player', 'WS', 'Season', 'WS_z']].tail(n=15))
print("We only have outliers in positive direction > 3. None < -3")


#identify outliers in DRAFT - PICK
# BQ
pick_count = draft_original["Pick"].value_counts()

draft_original['Pick_z'] = stats.zscore(draft_original['Pick'])
draft_original.head
draft_outliers = draft_original.query('Pick_z > 3 | Pick_z < -3')
draft_sort = draft_original.sort_values(['Pick_z'], ascending=False)
print(draft_sort[[ 'Draft', 'Pick', 'Pick_z']].head(n=15))
print(draft_sort[['Draft', 'Pick', 'Pick_z']].tail(n=15))
print("We have no z values > 3 or < -3")


count = draft_original["Draft"].value_counts()
#this lets us know how many picks were in each seasons draft

#identify outliers in DRAFT- DRAFT
# BQ
draft_original['Draft_z'] = stats.zscore(draft_original['Draft'])
draft_original.head
draft_outlier2 = draft_original.query('Draft_z > 3 | Draft_z < -3')
draft_sort2 = draft_original.sort_values(['Draft_z'], ascending=False)
print(draft_sort2[[ 'Draft', 'Pick', 'Draft_z']].head(n=15))
print(draft_sort2[['Draft', 'Draft_z']].tail(n=15))
print("We have no z values > 3 or < -3")
# Modern drafts only go 2 rounds .. roughly 60 picks

#Robert - Binning Based on Predictive Value
season_original['Season_binned'] = pd.cut(x = season_original['Season'], bins = [1979,1979.01, 1990, 2000, 2010, 2017], labels = ["70's", "80's", "90's", "2000's", "2010's"], right = False)
draft_original['Draft_binned'] = pd.cut(x = draft_original['Draft'], bins = [1978,1979.01, 1990, 2000, 2010, 2016], labels = ["70's", "80's", "90's", "2000's", "2010's"], right = False)

seasontab = pd.crosstab(season_original['Season_binned'], season_original['Season_binned'])
seasontab.plot(kind = 'bar', stacked = True, title = 'Binning with Season')

drafttab = pd.crosstab(draft_original['Draft_binned'], draft_original['Draft_binned'])
drafttab.plot(kind = 'bar', stacked = True, title = 'Binning with Draft')

#save the changes
# RL
season_original.to_csv('new_season.csv', index=False)
draft_original.to_csv('new_draft.csv', index=False)

newDraft= pd.read_csv("new_draft.csv")
newSeason = pd.read_csv("new_season.csv")

# fixing the Draft['Pick'] to only have 2 rounds (60)
# BQ
pick_filtered = newDraft[newDraft['Pick'] > 60]
newDraft.drop(pick_filtered.index, inplace = True)
print(newDraft.head())


# Classify players skill level based off winshars from zscores
# Classified a players skill level based off zscores. Zscore 
# over 3 is classified as a tier 1 player. 
# Zscore between 1 and 2 is tier 2. Zscore below 1 is tier 3. 
# IS
newSeason["Skill"] = newSeason["WS_z"]
skill = newSeason["Skill"]

for i in range(len(newSeason)):
    #print(newSeason["Skill"][i])
    temp = newSeason["WS_z"][i]

    #print(i,"1")
    if temp > 2:
        newSeason.set_value(i, "Skill", 1)

    elif temp > 1 and temp < 2:
        newSeason.set_value(i, "Skill", 2)

    else:
        newSeason.set_value(i, "Skill", 3) 

#print(newSeason["Skill"].head())
# IS
newSeason.to_csv("new_season.csv", index=False)

# Binning based off predictive value to enhance models
# IS
newSeason["Skill_bins"] = pd.cut(x = newSeason["Skill"], bins = [0, 1.1, 2.1, 3.1], labels = ["tier 1", "tier 2", "tier 3"], right = False)
skillTab = pd.crosstab(newSeason["Skill_bins"], newSeason["Skill_bins"])
skillTab.plot(kind = 'bar', stacked = True, title = 'Skill distribution')




#check with another graph
# compare how players would be re-drafted based off thier z-score
# BQ

plt.figure()
plt.hist(newSeason['WS'], ec="black")
plt.title("newseason WS")
plt.xlabel("WS");plt.ylabel("Frequency / # of records");plt.show()
plt.show()

#these are correct to show difference
plt.figure()
plt.hist(newDraft['Pick'], ec="black")
plt.title("Draft cut to 2 rounds")
plt.xlabel("Draft Picks");plt.show()


plt.figure()
plt.hist(newDraft['Draft'], ec='black')
plt.title("Draft cut to 2 rounds")
plt.xlabel("Draft Season");plt.ylabel("# of records");plt.show()
          
          
# Binning for winshare distribution
# IS
newSeason["WS_bins"] = pd.cut(x = newSeason["WS"], bins = [0, 3.1, 7.1, 10.1], labels = ["low WS", "medium WS", "High WS"], right = False)
WSTab = pd.crosstab(newSeason["WS_bins"], newSeason["WS_bins"])
WSTab.plot(kind = 'bar', stacked = True, title = 'WS distribution')

# Contingency table for Winshare distribution
# IS
crosstab_01 = pd.crosstab(newSeason["Skill_bins"], newSeason["WS_bins"])
print(crosstab_01)

#Histogram for comparison of winshare z-scores vs years
# IS
#Histogram for comparison of winshare z-scores vs years
# IS
plt.figure()
histDraft_y = newSeason[newSeason.WS_z > 3]["WS_z"]
histDraft_x = newSeason[newSeason.WS_z < 3]["WS"]
plt.hist([histDraft_y, histDraft_x], bins = 10, stacked = False)
plt.title("Histogram for win-share z-score in NBA")
plt.xlabel("WS")
plt.ylabel("z-scores")
plt.show()



#--------------------- PART 2 OF PROJECT ---------------------------------






#BAYLEE- merged DS idea
merged_nba = pd.merge(newSeason, newDraft, on='Player', how='inner')
merged_nba.to_csv("merged_nba.csv")


#now doing scatter plot of avg WS and longevity of lottery picks!
l = sns.scatterplot(x='Yrs', y='WS', hue= 'Draft_binned', data = merged_nba);


#SET UP PHASE
#BQ
#NOW REBALANCE FOR MERGED_NBA ----------------------------------------------
mergeCounts = merged_nba['Skill_bins'].value_counts()
merge_train, merge_test = train_test_split(merged_nba, test_size = .40, random_state = 7)

merge_train.shape #perfect 66.9%
merge_test.shape
merged_nba.shape

trainCounts = merge_train["Skill_bins"].value_counts()
#T3 = 6487, T2 = 827, T1 = 360, TOTAL = 7674
#T3 = 84%, T2 = 10%, T1 = 4 %
#while there are more average players in NBA, than good, and great....
# we need the percentages a little closer to get good models... 
x = (.1 * 7674 - 360) / .9 #fixing for T3 to 10%
resample = merge_train.loc[merge_train['Skill_bins'] == 'tier 1']
our_resample = resample.sample(n = 452, replace = True)

merge_train_rebal = pd.concat([merge_train, our_resample])

merge_train_rebal['Skill_bins'].value_counts()
#so now we have 15% records as tier 1 (elite players)
#T3 = 6487, T2 = 827, T1 = 812 TOTAL= 8604

#now probably need to rebalance T2 as it is only 9%
y = (.30 * 8604 - 827) / .70
nextresample = merge_train_rebal.loc[merge_train_rebal['Skill_bins'] == 'tier 2']
new_resample = nextresample.sample( n = 2506, replace = True)

merged_train_rebal = pd.concat([merge_train_rebal, new_resample])

merged_train_rebal["Skill_bins"].value_counts()
#T3 = 6487, T2 = 3333, T1 = 1290, TOTAL = 11410
#T3 = 56%, T2 = 29%, T1 = 11 %

#now doing K fold validation****
kf = KFold(n_splits = 8, shuffle = True, random_state = 2)
result = next(kf.split(merged_train_rebal["Skill_bins"]), None)
print(result)
ktrain = merged_train_rebal["Skill_bins"].iloc[result[0]]
ktest = merged_train_rebal["Skill_bins"].iloc[result[1]]
#those looks good!

stats.ttest_ind(merged_train_rebal['WS'], merge_test['WS'], equal_var = False)


#MODELING PHASE 
#BQ
# linear regression of PICK VS WINSHARE ************************************
#####BASELINE #######################
base = merged_nba.groupby(["Pick", 'Season_binned'])['WS'].mean()
base = base.reset_index()
base80s = base[base["Season_binned"] == "80\'s"]
plt.plot('Pick', 'WS', data = base80s, color = 'red', label = "The 80s")
aB80 = base80s["Pick"]
bB80 = base80s["WS"]
cB80 = np.polyfit(aB80, bB80, 1)
pB80 = np.poly1d(cB80)
plt.plot(aB80, pB80(aB80), linestyle = 'dashed' , color = 'red', label = "80 Regression")

Xbase = base80s['Pick'].values.reshape(-1, 1)
Ybase = base80s['WS'].values.reshape(-1, 1)


regressorBase = LinearRegression();
regressorBase.fit(Xbase, Ybase)
print(regressorBase.intercept_)
print(regressorBase.coef_)
##actual regression using different decades
decade = merge_train_rebal.groupby(['Pick', 'Draft_binned'])['WS'].mean()
decade = decade.reset_index()

the80s = decade[decade["Draft_binned"] == "80\'s"]
the90s = decade[decade["Draft_binned"] == "90\'s"]
the00s = decade[decade["Draft_binned"] == "2000\'s"]
the10s = decade[decade["Draft_binned"] == "2010\'s"]

plt.plot('Pick', 'WS', data = the80s, color = 'red', label = "The 80s")
plt.plot('Pick', 'WS', data = the90s, color = 'blue', label = "The 90s")
plt.plot('Pick', 'WS', data = the00s, color = 'green', label = "The 2000s")
plt.plot('Pick', 'WS', data = the10s, color = 'orange', label = "The 2010s")
plt.xlabel("Draft Pick"); plt.ylabel("Average Win Shares")
plt.title("Draft Pick vs WS by Season Decade")

#adding regression line for 80s
a80 = the80s["Pick"]
b80 = the80s["WS"]
c80 = np.polyfit(a80, b80, 1)
p80 = np.poly1d(c80)
plt.plot(a80, p80(a80), linestyle = 'dashed' , color = 'red', label = "80 Regression")
#regression line for 90s
a90 = the90s["Pick"]
b90 = the90s["WS"]
c90 = np.polyfit(a90, b90, 1)
p90 = np.poly1d(c90)
plt.plot(a90, p90(a90), linestyle = 'dashed' , color = 'blue', label = "90 Regression")
#regression line for 2000s
a00 = the00s["Pick"]
b00 = the00s["WS"]
c00 = np.polyfit(a00, b00, 1)
p00 = np.poly1d(c00)
plt.plot(a00, p00(a00), linestyle = 'dashed' , color = 'green', label = "00 Regression")
#regression line for 2010s
a10 = the10s["Pick"]
b10 = the10s["WS"]
c10 = np.polyfit(a10, b10, 1)
p10 = np.poly1d(c10)
plt.plot(a10, p10(a10), linestyle = 'dashed' , color = 'orange', label = "10 Regression")


#checking regression stats ***********************************************
#need to form same variables from the TEST set
test80s = merge_test[merge_test["Season_binned"] == "80\'s"]

#testing for the 80s
X1 = the80s['Pick'].values.reshape(-1, 1)
Y1 = the80s['WS'].values.reshape(-1, 1)

X1Test = merge_test['Pick'].values.reshape(-1, 1)
Y1Test = merge_test['WS'].values.reshape(-1, 1)

regressor2 = LinearRegression();
regressor2.fit(X1, Y1)
print(regressor2.intercept_)
print(regressor2.coef_)

y_pred2 = regressor2.predict(X1Test)
df2 = pd.DataFrame({"Actual": Y1Test.flatten(), 'Predicted': y_pred2.flatten()})
#visualizing using bar chart
df2.plot(kind = 'bar', figsize = (16, 10))
plt.grid(which= 'major', linestyle = '-', linewidth = '.5', color = 'green')
plt.grid(which = 'minor', linestyle = ':', linewidth = '.5', color = 'black')
plt.title("80's Seasons , actual vs predicted")

plt.scatter(X1Test, Y1Test, color = 'gray')
plt.plot(X1Test, y_pred2, color = 'red', linewidth = 2)
plt.show()

print("mean absolute error:", metrics.mean_absolute_error(Y1Test, y_pred2))
print("Mean squared error: ", metrics.mean_squared_error(Y1Test, y_pred2))
print("Root mean squared error: ", np.sqrt(metrics.mean_squared_error(Y1Test, y_pred2)))
print("Coef of determination : %.2f"% metrics.r2_score(Y1Test, y_pred2))

#NEW LINEAR REGRESSION MODEL 2000s draft lottery, first Round, second round
players = merged_train_rebal.groupby(['Pick', 'Player', 'Draft', 'Yrs'])["WS"].mean()
players = players.reset_index()
players = players[players["Draft"] > 1999]

lotteryPlayer = players[players["Pick"] < 14]
FirstRounder = players.query("Pick > 14 & Pick < 30")
SecondRounder = players.query("Pick > 30 & Pick < 60")

plt.plot('Pick', 'WS', data = lotteryPlayer, color = 'red', label = "The 80s")
plt.plot('Pick', 'WS', data = FirstRounder, color = 'green', label = "The 2000s")
plt.plot('Pick', 'WS', data = SecondRounder, color = 'blue', label = "The 2010s")
plt.xlabel("Draft Pick"); plt.ylabel("Average Win Shares")
plt.title("Draft Pick vs WS From 2000")

#adding regression line for lottery pick
a1 = lotteryPlayer["Pick"]
b1 = lotteryPlayer["WS"]
c1 = np.polyfit(a1, b1, 1)
p1 = np.poly1d(c1)
plt.plot(a1, p1(a1), linestyle = 'dashed' , color = 'black', label = "Player Regression")

#adding regression line for first round
a2 = FirstRounder["Pick"]
b2 = FirstRounder["WS"]
c2 = np.polyfit(a2, b2, 1)
p2 = np.poly1d(c2)
plt.plot(a2, p2(a2), linestyle = 'dashed' , color = 'black', label = "Player Regression")

#adding regression line for second round
a3 = SecondRounder["Pick"]
b3 = SecondRounder["WS"]
c3 = np.polyfit(a3, b3, 1)
p3 = np.poly1d(c3)
plt.plot(a3, p3(a3), linestyle = 'dashed' , color = 'black', label = "Player Regression")

#overall regression line not nearly as accurate
#adding regression line for player
ao = players["Pick"]
bo = players["WS"]
co = np.polyfit(ao, bo, 1)
po = np.poly1d(co)
plt.plot(ao, po(ao), linestyle = 'dashed' , color = 'yellow', label = "Player Regression")



#DECISION TREE - RL ############################################################
import numpy as np
import statsmodels.tools.tools as stattools
from sklearn.tree import DecisionTreeClassifier, export_graphviz

season_tr = merged_train_rebal.groupby(['Skill', 'Pick', 'WS'])['Season'].mean()
season_tr = season_tr.reset_index()
season_cut = pd.qcut(season_tr["Pick"], q = 4)
season_cut.value_counts()
season_tr['Pick_binned'] = pd.cut(x = season_tr['Pick'], bins = [0.999, 9, 21, 36, 61], labels = ['0 to 9', '9 to 21', '21 to 36', '36 to 60'], right = False)

season_y = season_tr['Pick_binned']

predictor_season = season_tr[['Skill', 'WS']]
season_x = predictor_season

x = np.array(season_x)
y = np.array(season_y)

season_xnames = ['Skill based', 'Winshares']
season_ynames = ['0 to 9 Pick', '9 to 21 Pick', '21 to 36 Pick', '36 to 60 Pick']

cart01 = DecisionTreeClassifier(criterion = 'gini', max_leaf_nodes = 8).fit(x, y)

export_graphviz(cart01, out_file = 'cart01.dot', feature_names = season_xnames,\
                class_names = season_ynames)


#BASELINE CLASSIFICATION - RL
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
merged_train_rebal = train_test_split(x,y, random_state = 0)

dummy = DummyClassifier(strategy = 'uniform', random_state = 1)
dummy.fit(x,y)
dummy.score(x,y)

#-------------------------------------------------------------------------------------
#C50 MODEL 1 ---------------------------------------------------------------
#lets make the target variable yrs played
ystart = merged_train_rebal.groupby(["Player", "Draft", "Pick", "Yrs"])['WS'].mean()
ystart = ystart.reset_index()

yrsCut = pd.qcut(ystart["Yrs"], q = 3)
yrsCut.value_counts()
#this cuts our yrs into 4 quartiles... (0-7)(7-11)(11-14)(14-21)
ystart['Yrs_binned'] = pd.cut(x = ystart['Yrs'], bins = [-.001, 4, 9, 22], labels = ["0 to 4", "4 to 9", "10 to 22"], right = False)
ystart["Yrs_binned"].value_counts()
YNew = ystart["Yrs_binned"]
predictor_col = ystart[['Pick', 'WS']]
XNew = predictor_col
x_names = ["draft pick", "Winshares"]
y_names = ['0-4 Seasons', "4-9 Seasons", "9-22 Seasons"]


#create C5.0
clf = tree.DecisionTreeClassifier(criterion= 'entropy', max_leaf_nodes = 5).fit(XNew, YNew)
#clf = clf.fit(X, Y)
tree.plot_tree(clf, label='all', filled=True)

#this exports the text to create the tree
export_graphviz(clf, out_file = "C:/Users/bayqu/Desktop/SPRING20/DataMining/project/part2/clf4.dot", 
                feature_names=x_names, class_names=y_names)
#predict and evaluate model
y_pred = clf.predict(XNew)
cart1_accurary = metrics.accuracy_score(YNew, y_pred)
cart1_confusion = confusion_matrix(YNew, y_pred)
print(classification_report(YNew, y_pred, target_names=y_names))

#CART MODEL BASELINE
why = merged_nba.groupby(["Player", "Draft", "Pick", "Yrs"])["WS"].mean()
why = why.reset_index()
why["Yrs_bin"] = pd.cut(x = why['Yrs'], bins = [-.001, 4, 9, 22], labels = ["0 to 4", "4 to 9", "10 to 22"], right = False)

whyNew = why["Yrs_bin"]
predict_column = why[["Pick", "WS"]]
exNew = predict_column
exNames = ["Draft Pick", "WS"]
y_names = ["0-4 seasons", "4-9 seasons", "9-22 seasons"]

cliff = tree.DecisionTreeClassifier(criterion= 'gini', max_leaf_nodes = 5).fit(XNew, YNew)
tree.plot_tree(cliff, label='all', filled=True)
y_predN = cliff.predict(exNew)
cart2_accurary = metrics.accuracy_score(whyNew, y_predN)
cart1_confusion = confusion_matrix(whyNew, y_predN)
print(classification_report(whyNew, y_predN, target_names=y_names))



#-------------------------------------------------------------------------------------
#CLUSTERING - IS

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.datasets.samples_generator import make_blobs

merge = pd.read_csv("merged_nba.csv")
mergeTrain, mergeTest = train_test_split(merge, test_size = .25, random_state=7)

# Use K-means clustering to analyze predictor variables
# K-means for Ws - Pick
print("-------------------------------------------------------------------------")
print("K-means for WS-Pick")
X = mergeTrain[["WS", "Pick"]]
Xz = pd.DataFrame(stats.zscore(X), columns=["WS", "Pick"])
kmeans01 = KMeans(n_clusters = 3).fit(Xz)
cluster = kmeans01.labels_
Cluster1 = Xz.loc[cluster == 0]
Cluster2 = Xz.loc[cluster == 1]
Cluster3 = Xz.loc[cluster == 2]
print(Cluster1.describe())
print(Cluster2.describe())
print(Cluster3.describe())
print("-------------------------------------------------------------------------")
print("K-means for WS-Pick Test Set")
XTest = mergeTest[["WS", "Pick"]]
testZ = pd.DataFrame(stats.zscore(XTest), columns = [["WS", "Pick"]])
kmeans02 = KMeans(n_clusters = 3).fit(testZ)
clusterTest = kmeans02.labels_
Cluster1T = testZ.loc[clusterTest == 0]
Cluster2T = testZ.loc[clusterTest == 1]
Cluster3T = testZ.loc[clusterTest == 2]
print(Cluster1T.describe())
print(Cluster2T.describe())
print(Cluster3T.describe())
print("----------------------------------------------------------------------")
# K-means for season-years
print("K-means for season-years")
Y = mergeTrain[["Season", "Yrs"]]
Yz = pd.DataFrame(stats.zscore(X), columns=["Season", "Yrs"])
kmeans01 = KMeans(n_clusters = 3).fit(Yz)
cluster = kmeans01.labels_
Cluster1 = Yz.loc[cluster == 0]
Cluster2 = Yz.loc[cluster == 1]
Cluster3 = Yz.loc[cluster == 2]
print(Cluster1.describe())
print(Cluster2.describe())
print(Cluster3.describe())
print("-------------------------------------------------------------------------")

# Plot WS vs Picks
# Shows that the higher draft picks (lower y values) were the ones who typically had the abnormally high WS
x = mergeTrain["WS"]
y = mergeTrain["Pick"]
plt.figure()
plt.title("WS - Pick")
plt.scatter(x, y, s=5)
plt.show()


#-------------------------------------------------------------------------------------------
# Random Forest model IS
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

merge = pd.read_csv("merged_nba.csv")

X = merge[["Pick"]]
y = merge[["Yrs"]]
# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

# random forest model creation
rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
# predictions
rfc_predict = rfc.predict(X_test)
print(rfc_predict)

from sklearn.metrics import classification_report, confusion_matrix


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print('\n')
print("=== Mean AUC Score ===")

# Find parameters to modify to improve results
from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
#max_features = [‘auto’, ‘sqrt’]

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_depth': max_depth
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(X_train, y_train)
# print results
print(rfc_random.best_params_)


# Rerun RandomForestClassifier with out improved values.
rfc = RandomForestClassifier(n_estimators=1400, max_depth=220)
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)

print(" CONFUSION MATRIX FOR RANDOM FOREST CLASSIFIER WITH NEW PARAMETERS")
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print(" CLASSIFICATION REPORT FOR RANDOM FOREST CLASSIFIER WITH NEW PARAMETERS")
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print('\n')
print("=== Mean AUC Score ===")
#-------------------------------------------------------------------------------------------



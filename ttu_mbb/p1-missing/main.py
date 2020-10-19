# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:13:25 2020

@author: bayqu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


season1920 = pd.read_csv("19_20.csv")
per40 = pd.read_csv("individ40.csv")

#remove nan values that are due to empty excel cells
season1920 = season1920.dropna()

#now we need to group players that WILL NOT BE RETURNING
leaving = season1920.loc[[1,2,5,7,10,11], :]
leaving40 = per40.loc[[0, 2, 3, 7, 8, 9], :]

#players that ARE RETURNING
returning = season1920.loc[[3,4,6,8,9], :]
returning40 = per40.loc[[1, 4, 5, 6, 10], :]

#graphing TOT MINUTES
plt.figure()
plt.bar(season1920['Player'], season1920['M-TOT'], color = ["black", "black", "red",
        "red", "black", "red", "black", "red", "red", "black", "black"])
plt.title("Total Minutes 19-20")
plt.xlabel("Player");plt.ylabel("Minutes")
plt.xticks(rotation = 30, horizontalalignment = 'center')
colors = {"Leaving": "black", "Returning": "red"}
labels= list(colors.keys())
handles = [plt.Rectangle((0,0), 1, 1, color = colors[label]) for label in labels]
plt.legend(handles, labels)
plt.show()

#graphing MPG
plt.figure()
plt.bar(season1920['Player'], season1920['M-AVG'], color = ["black", "black", "red",
        "red", "black", "red", "black", "red", "red", "black", "black"])
plt.title("Minutes Per Game 19-20")
plt.xlabel("Player");plt.ylabel("Minutes")
plt.xticks(rotation = 30, horizontalalignment = 'center')
colors = {"Leaving": "black", "Returning": "red"}
labels= list(colors.keys())
handles = [plt.Rectangle((0,0), 1, 1, color = colors[label]) for label in labels]
plt.legend(handles, labels)
plt.show()

#PPG    PPG         PPG         PPG
#lets demonstrate ppg ------------------------------------------------------
#avg ppg last season was 71.9
lpoints = leaving["P-AVG"]
rpoints = returning["P-AVG"]
leavingppg = lpoints.sum()
returnppg = rpoints.sum() #losing 45 ppg next season
print(leavingppg)

#now turn into pie chart
plt.figure()
pielabel1 = "Kyler Edwards \n 11.4", "Terrence Shannon Jr. \n 9.8", "Kevin McCullar \n 6.0", "Clarence Nadolny \n 2.0 ", "Avery Benson \n 1.8", "LOST POINTS \n 45.0"
points = [11.4, 9.8, 6, 2, 1.8, 45.3]
colors = ["#fb0909", "#f95656", "#fb7373", "#fbc5c5", "#f99b9b", "#6d6b6b"]
explode = [.05, .05, .05, .05, .05, .1]
plt.pie(points, colors = colors, explode = explode, labels = pielabel1, autopct='%1.1f%%', shadow = True, 
        startangle = 140)
plt.title("Points Per Game Breakdown", bbox={'facecolor':'0.8', 'pad':5})
plt.axis('equal')
plt.show()


#DEMONSTRATE PPG PER 40 MINUTES______________________________________________
leavingppg40 = leaving40['PTS']
returningppg40 = returning40['PTS']

tot40ppg_L= leavingppg.sum() #81
tot40ppg_R = returningppg40.sum() #63
#pie chart time!!
plt.figure()
pielabelPPG40 = "Terrence Shannon Jr.\n 16.7", "Kyler Edwards\n 13.7", "Clarence Nadolny\n 13.1", "Kevin\nMcCullar\n 13.0", "Avery Benson\n 7.4", "LEAVING PPG\n 81.0"
ppg40 = [16.7, 13.7, 13.1, 13, 7.4, 81]
colorsNew = ['#ff0000', '#ff1a1a', '#ff3333', '#ff4d4d', '#ff6666', '#999999']
plt.pie(ppg40, colors = colorsNew, explode = explode, labels = pielabelPPG40, autopct = '%1.1f%%', shadow = True, startangle = 140)
plt.title("Points Per 40 Minute Breakdown", bbox={'facecolor':'0.8', 'pad':5})
plt.axis('equal')
plt.show()

#Demonstrate TOTAL POINTS OVER THE SEASON___________________________________
#use

seasonP_R = returning['PTS']
seasonP_L = leaving['PTS']

totalP_R = seasonP_R.sum() #915
totalP_L = seasonP_L.sum() #1315
#turn into a pie chart

totalPointsLabel = "Kyler Edwards\n 354", "Terrence Shannon Jr.\n 284", "Kevin\nMcCullar\n 175", "\nClarence\nNadolny\n 47","      Avery Benson\n            55", "TOTAL POINTS LEAVING\n 1315"
totalPoints = [354, 284, 175, 47, 55, 1315]
plt.figure()
plt.pie(totalPoints, colors = colorsNew, explode = explode, labels = totalPointsLabel, 
        autopct = "%1.1f%%", shadow = True, startangle = 140)
plt.title("Total Points 19-20 Breakdown", bbox={'facecolor':'0.8', 'pad':5})
plt.axis('equal')
plt.show()

#OFFENSE        OFFENSE ########################################################
#totalOff = assists + ppg
retassist = returning['AST']
leavassist = leaving['AST']

retoffense = totalP_R + retassist
leavoffense = totalP_L + leavassist
print(retoffense.sum())
print(leavoffense.sum())

plt.figure()
pielabel2 = "Kyler Edwards", "Terrence Shannon Jr", "Kevin McCullar", "Clarence Nadolny", "Avery Benon", "NEW OFFENSE"
offense = [106.4, 38.8, 27, 15, 8.8, 357.3]
plt.pie(offense, colors = colors, explode = explode, labels = pielabel2, autopct='%1.1f%%', shadow = True, 
        startangle = 140)
plt.axis('equal')
plt.show()


#REBOUNDS       REBOUNDS        REBOUNDS +++++++++++++++++++++++++++++++++++

#lets demonstrate rebounds per game__________________________________________
lrebounds = leaving['R-AVG']
retrebounds = returning['R-AVG']
leavingrpg = lrebounds.sum()
returningrpg = retrebounds.sum()
print(leavingrpg)
print(returningrpg)

pielabel2 = "Kyler", "Terrence Shannon", "Kevin", "Clarence", "Avery", "NEW REBOUNDS"
rebounds = [4, 4.1, 3.2, 1, 1.4, 18.6 ]
plt.pie(rebounds, colors = colors, explode = explode, labels = pielabel2, autopct='%1.1f%%', shadow = True, 
        startangle = 140)
plt.axis('equal')
plt.show()




#use 

#MINUTES        MINUTES         MINUTES ########################################
#SEASON___________________________________________________________________
newmins= leaving['M-TOT']
retmins = returning['M-TOT']

subtractmins = newmins.sum()  #3600
print(subtractmins)

#lets turn this percentage into a pie chart
plt.figure()
pielabel1 = "Kyler Edwards", "Terrence Shannon Jr.", "Kevin McCullar", "Avery\nBenson", "Clarence\n Nadolny", "Minutes Available"
sizes = [1036, 681, 537, 301, 145, 3600 ]
colors = ["#fb0909", "#f95656", "#fb7373", "#f99b9b",  "#fbc5c5", "#6d6b6b"]
explode = [0, 0, 0, 0, 0, .1]
plt.pie(sizes, colors = colorsNew, explode = explode, labels = pielabel1, autopct='%1.1f%%', shadow = True, 
        startangle = 140)
plt.axis('equal')
plt.title("Total Minutes 19-20 Breakdown", bbox={'facecolor':'0.8', 'pad':5})
#plt.legend()
plt.show()

#MINUTES PER GAME__________________________________________________________
#so a game is 200  mins total. 2 20 min halves with 5 players. 40 * 5 = 200
mpgtot = 203
newmpg = leaving['M-AVG']
subtractmpg = newmpg.sum()
print(subtractmpg) #123.9

plt.figure()
pielabel1 = "Kyler", "Terrence Shannon", "Kevin", "Clarence", "Avery", "NEW MPG"
sizes2 = [33.4, 23.5, 18.5, 6, 9.7, 123.9]
explode = [0, 0, 0, 0, 0, .1]
plt.pie(sizes2, explode = explode, labels = pielabel1, autopct='%1.1f%%', shadow = True, 
        startangle = 140)
plt.axis('equal')
plt.show()





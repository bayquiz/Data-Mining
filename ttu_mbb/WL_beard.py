# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:12:09 2020

@author: bayqu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import OffsetFrom

seas16= pd.read_csv("16-17_season.csv")
seas17 = pd.read_csv("17-18_season.csv")
seas18 = pd.read_csv("18-19_season.csv")
seas19 = pd.read_csv("19-20_season.csv")

KU16 = pd.read_csv("16-17_KU.csv")
KU17 = pd.read_csv("17-18_KU.csv")
KU18 = pd.read_csv("18-19_KU.csv")
KU19 = pd.read_csv("19-20_KU.csv")


fig = plt.figure()
plt.grid(b = None)
plt.style.use('seaborn-darkgrid')
plt.plot('Game', 'Wins', data = seas16, color = 'red', label = "16-17 Season", linewidth = 4, linestyle = '-')
plt.plot('Game', 'Wins', data = seas17, color = 'black', label = "17-18 Season", linewidth = 4, linestyle = '-')
plt.plot('Game', 'Wins', data = seas18, color = 'white', label = "18-19 Season", linewidth = 4,linestyle = '-')
plt.plot('Game', 'Wins', data = seas19, color = 'gray', label = "19-20 Season", linewidth = 4, linestyle = '-')

#plt.legend()
plt.xlabel("Games"); plt.ylabel("Wins")
#plt.title("Seasons Under Chris Beard",  bbox={'facecolor':'0.8', 'pad':5})

fig.savefig("WL.png", transparent = True)
#label interesting points
#plt.annotate('National\nChampionship', xy = (31, 32), xytext = (31, 32), size = 12)
#plt.annotate('Elite Eight', xy = (36, 27), xytext = (36, 28), size = 11)


#WORKING ON GROUPED BAR CHART DISPLAYING WINS UNDER CHRIS BEARD ERA VS
#OTHER BLUE BLOOD SCHOOLS- DUKE, KENTUCKY, KANSAS, BAYLOR


#bar1 = season 16 wins[TTU, Kansas, BU, Duke, Kentucky]
bar1 = [18, 31, 27, 28, 32]
bar2 = [27, 31, 19, 29, 26]
bar3 = [31, 26, 20, 32, 30]
bar4 = [18, 28, 26, 25, 25]

fig2 = plt.figure(figsize = (6,4))
barwidth = .15
r1 = np.arange(len(bar1))
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
r4 = [x + barwidth for x in r3]

plt.bar(r1, bar1, color = '#5DADE2', width = barwidth, edgecolor= 'black', label = '16-17 Season')
plt.bar(r2, bar2, color = '#85C1E9', width = barwidth, edgecolor= 'black', label = '17-18 Season')
plt.bar(r3, bar3, color = '#AED6F1', width = barwidth, edgecolor= 'black', label = '18-19 Season')
plt.bar(r4, bar4, color = '#D6EAF8', width = barwidth, edgecolor= 'black', label = '19-20 Season')

plt.xlabel('Team', fontweight='bold')
plt.ylabel("Wins", fontweight = 'bold')
plt.xticks([r + barwidth for r in range(len(bar1))], ['TTU', 'KU', 'BU', 'Duke', 'Kentucky'])
 
# Create legend & Show graphic
plt.legend()
plt.show()
fig2.savefig("WinsVS_blue.png")



fig3 = plt.figure(figsize=(4, 3))
plt.bar(r1, bar1, color = '#ff1414', width = barwidth, edgecolor= 'black', label = '16-17 Season')
plt.bar(r2, bar2, color = '#ff4e4e', width = barwidth, edgecolor= 'black', label = '17-18 Season')
plt.bar(r3, bar3, color = '#ff7676', width = barwidth, edgecolor= 'black', label = '18-19 Season')
plt.bar(r4, bar4, color = '#ff9d9d', width = barwidth, edgecolor= 'black', label = '19-20 Season')

plt.xlabel('Team', fontweight='bold')
plt.ylabel("Wins", fontweight = 'bold')
plt.xticks([r + barwidth for r in range(len(bar1))], ['TTU', 'KU', 'BU', 'Duke', 'Kentucky'])
 
# Create legend & Show graphic
plt.legend()
plt.show()
fig3.savefig("WinsVS_red.png")

#GET RID OF LABELS BEFORE EXPORTING TO IPAD TO EDIT AND ADD


#NOW NCAA TOURNEY WINS FOR EACH TEAM THE PAST 3 SEASONS 
# ** ONLY 3 DUE TO COVID

#ncaa1 =  16-17 ncaawins [ttu, kansas, baylor, duke, kentucky]
ttuNW = 5 + 3 + 0
ttuNT = 10
kansasNW = 1 + 4 + 3
kansasNT = 11
baylorNW = 1 + 0 + 2
baylorNT = 5
dukeNW = 3 + 3 + 1
dukeNT = 10
kentuckyNW = 3 + 2 + 3
kentuckyNT = 11
uncNW = 6 + 1 + 2 # one championship
uncNT = 11
novaNW = 1 + 6 + 1
novaNT= 10
uvNW = 6 + 0 + 1
uvNT = 9


nb1 = [8, 8, 3, 7, 8, 9, 8, 7] #wins
nb2 = [10, 11, 5, 10, 11, 11, 10, 9] #total games

barheight = np.add(nb1, nb2).tolist()
r = [0, 2, 4, 6, 8, 10, 12, 14]
names = ['TTU', 'Kansas', 'Baylor', 'Duke', 'Kentucky', 'UNC', 'Nova', 'UVA']
barwide = 1

# Create win bars
fig4 = plt.figure()
#total bar on top
plt.bar(r, nb2, color='#ff4e4e', width=barwide, label = 'NCAA Tournament Games')
plt.bar([x + .05 *barwide for x in r], nb1, color = '#4d4d4d',edgecolor = 'white',  width = .5,  label = 'Games Won')

plt.xticks(r, names, fontweight = 'bold')
plt.xlabel('Team')
plt.legend()
plt.show()
fig4.savefig("NCAA.png")



#REDO THE BARS BUT WITH UNC INSTEAD SEASON WINS
#bar1 = season 16 wins[TTU, Kansas, unc, Duke, Kentucky]

bars1 = [18, 31, 33, 28, 32]
bars2 = [27, 31, 26, 29, 26]
bars3 = [31, 26, 29, 32, 30]
bars4 = [18, 28, 14, 25, 25]

fig5 = plt.figure(figsize = (6,4))
barwidth = .15
rr1 = np.arange(len(bar1))
rr2 = [x + barwidth for x in r1]
rr3 = [x + barwidth for x in r2]
rr4 = [x + barwidth for x in r3]

plt.bar(rr1, bars1, color = '#5DADE2', width = barwidth, edgecolor= 'black', label = '16-17 Season')
plt.bar(rr2, bars2, color = '#85C1E9', width = barwidth, edgecolor= 'black', label = '17-18 Season')
plt.bar(rr3, bars3, color = '#AED6F1', width = barwidth, edgecolor= 'black', label = '18-19 Season')
plt.bar(rr4, bars4, color = '#D6EAF8', width = barwidth, edgecolor= 'black', label = '19-20 Season')

plt.xlabel('Team', fontweight='bold')
plt.ylabel("Wins", fontweight = 'bold')
plt.xticks([r + barwidth for r in range(len(bar1))], ['TTU', 'KU', 'UNC', 'Duke', 'Kentucky'])
 
# Create legend & Show graphic
plt.legend()
plt.show()
fig2.savefig("WINS_unc_blue.png")


















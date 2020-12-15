# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:48:10 2020

@author: Kasra
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("C:/Users/Kasra/Documents/Projects/artificial-intelligence/Projects/2_Classical Planning/result.xlsx")
df["Heuristic"].fillna(value="",inplace=True)
df["Method"] = df["Search Method"] +"_"+ df["Heuristic"]
ax = None
for m in np.unique(df["Method"]):
    if "greedy" in m:
        style = ':'
    elif "astar" in m:
        style = '--'
    else:
        style = '-'
    df_sub = df[df["Method"]==m]
    if ax is None:
        ax = df_sub.plot(x="Actions", y="Plan length", label=m, style=style, logy=True)
        plt.ylabel("Search Time")
    else:
        df_sub.plot(x="Actions", y="Plan length", label=m, style=style, ax=ax)
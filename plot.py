import os

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

font_path = os.path.abspath(os.path.dirname(__file__) + './config/NanumGothicBold.ttf')
font_name = fm.FontProperties(fname=font_path, size=15).get_name()
print(font_name)
plt.rc('font', family=font_name)

mpl.rcParams['axes.unicode_minus'] = False

array = [[15, 2, 17],
         [2, 7, 8],
         [28, 10, 142]]

df_cm = pd.DataFrame(array, index=[' ', ' ', ' '], columns=[' ', ' ', ' '])


plt.figure(figsize=(10, 8))

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g', cmap="YlGnBu")

plt.show()

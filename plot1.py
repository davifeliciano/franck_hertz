import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{siunitx}'
})

plt.style.use('ggplot')

files = glob.glob('data/T*3000mV*')
temps = [int(file.split('_')[1]) for file in files]
dfs = [pd.read_csv(file, sep='\s+', skiprows=(0, 1), index_col=0, decimal=',')
       for file in files]

for df in dfs:
    df.columns = ['I', 'U']
    df.index.names = ['Time']
    df.loc[:, 'U'] *= 2

fig, ax = plt.subplots()

for i in range(len(dfs)):
    x = gaussian_filter1d(dfs[i].loc[:, 'U'], sigma=2)
    y = gaussian_filter1d(dfs[i].loc[:, 'I'], sigma=2)

    ax.plot(x, y, label=rf'${temps[i]} \si{{\kelvin}} $')

ax.legend()
plt.show()

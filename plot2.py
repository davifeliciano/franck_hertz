import re
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

files = glob.glob('data/T_15*')
voltages = [int(re.search('VR_(.+?)mV', file).group(1)) for file in files]
dfs = [pd.read_csv(file, sep='\s+', skiprows=(0, 1), index_col=0, decimal=',')
       for file in files]

for df in dfs:
    df.columns = ['I', 'U']
    df.index.names = ['Time']
    df.loc[:, 'U'] *= 2

fig, ax = plt.subplots()

ax.set(xlim=(dfs[0].iloc[0, 1], dfs[0].iloc[-1, 1]),
       xlabel=r'$ U_a $ (\si{\volt})',
       ylabel=r'$ I_s $ (\si{\ampere})')

for i in range(len(dfs)):
    x = gaussian_filter1d(dfs[i].loc[:, 'U'], sigma=2)
    y = gaussian_filter1d(dfs[i].loc[:, 'I'], sigma=2)
    ax.plot(x, y, label=rf'$ U_s = {voltages[i]} \si{{\milli\volt}} $')

handles, labels = ax.get_legend_handles_labels()

handles_labels_voltages = [[h, l, v] for h, l, v in zip(handles, labels, voltages)]
handles_labels_voltages.sort(key=lambda list: list[2])

handles = [hv[0] for hv in handles_labels_voltages]
labels = [hv[1] for hv in handles_labels_voltages]

ax.legend(handles, labels)

fig.savefig('plots/constant_temp.png', dpi=300)
plt.show()

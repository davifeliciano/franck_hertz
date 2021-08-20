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

# importing files into dataframes
files = glob.glob('data/T_15*')
voltages = [int(re.search('VR_(.+?)mV', file).group(1)) for file in files]
dfs = [pd.read_csv(file, sep='\s+', skiprows=(0, 1), index_col=0, decimal=',')
       for file in files]

# relabeling columns and index
for df in dfs:
    df.columns = ['I', 'U']
    df.index.names = ['Time']
    df.loc[:, 'U'] *= 2

# plot of current in terms of drive voltage
fig, ax = plt.subplots()

ax.set(xlim=(dfs[0].iloc[0, 1], dfs[0].iloc[-1, 1]),
       xlabel=r'$ U_a $ (\si{\volt})',
       ylabel=r'$ I_s $ (\si{\ampere})')

minima_dfs = []
for i in range(len(dfs)):
    x = np.array(gaussian_filter1d(dfs[i].loc[:, 'U'], sigma=3))
    y = np.array(gaussian_filter1d(dfs[i].loc[:, 'I'], sigma=3))

    minima_indexes = (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0][-2:] + 1
    minima_df = dfs[i].iloc[minima_indexes, :]
    minima_dfs.append(minima_df)

    x_minima = minima_df.loc[:, 'U']
    y_minima = minima_df.loc[:, 'I']

    color = ax.plot(x_minima, y_minima, 'o', ms=3.0)[-1].get_color()
    ax.plot(x, y, color=color, label=rf'$ U_s = {voltages[i]} \si{{\milli\volt}} $')

# sorting legend labels by stop voltage value
handles, labels = ax.get_legend_handles_labels()

handles_labels_voltages = [[h, l, v] for h, l, v in zip(handles, labels, voltages)]
handles_labels_voltages.sort(key=lambda list: list[2])

handles = [hv[0] for hv in handles_labels_voltages]
labels = [hv[1] for hv in handles_labels_voltages]

ax.legend(handles, labels)

# plot of current minima in terms of stop voltages
fig_min, ax_min = plt.subplots()

ax_min.set(xlabel=r'$ U_s $ (\si{\volt})',
           ylabel=r'$ I_s $ (\si{\ampere})')

current_mins = [[df.iloc[i, 0] for df in minima_dfs] for i in range(2)]
voltage_mins = [[df.iloc[i, 1] for df in minima_dfs] for i in range(2)]

final_dfs = []
for current_min, voltage_min in zip(current_mins, voltage_mins):
    data = [[v, i, u] for v, i, u in zip(voltages, current_min, voltage_min)]
    sorted_data = sorted(data, key=lambda list: list[0])

    x = [data[0] for data in sorted_data]
    y = [data[1] for data in sorted_data]
    u = [data[2] for data in sorted_data]

    ax_min.plot(x, y, 'o-', ms=3.0)

    df = pd.DataFrame(np.array([x, y, u]).T, columns=('U_stop', 'I', 'U_drive'))
    final_dfs.append(df)

# export data for each minimum as csv
filenames = ('first_minimum.csv', 'second_minimum.csv')
for df, filename in zip(final_dfs, filenames):
    df.to_csv(filename, decimal=',', index=None)

fig.savefig('plots/constant_temp.png', dpi=300)
fig_min.savefig('plots/current_minima.png', dpi=300)
plt.show()

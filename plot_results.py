
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot median reward from log file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-l','--log', type=str, help='name of plot', default=None)
parser.add_argument('-f','--files', nargs='+', help='files to plot and compare', required=True)
args = parser.parse_args()

fig = plt.figure('{}'.format('None' if args.log is None else args.log))
fig.suptitle('{}'.format('None' if args.log is None else args.log))
x = np.arange(100)
y_all = []
plots = []
for i, each in enumerate(args.files):
    y = np.array([])
    with open(each) as f:
        for line in f.readlines():
            chunks = line.strip().split()
            if chunks[0]=='step':
                y = np.append(y, [float(chunks[5])])
        y_all.append(np.copy(y))
y_all = np.vstack(y_all)
# y_mean = np.mean(y_all,axis=0)
y_25per = np.percentile(y_all,25,axis=0)
y_50per = np.percentile(y_all,50,axis=0)
y_75per = np.percentile(y_all,75,axis=0)
# plt.plot(x, y_mean, linewidth=0.5, label='average')
plt.plot(x, y_50per, linewidth=0.5, label='50 percentile')
plt.plot(x, y_25per, linewidth=0.5, label='25th percentile')
plt.plot(x, y_75per, linewidth=0.5, label='75th percentile')

plt.xlabel('Training episode')
plt.ylabel('Median cumulative reward / episode')
plt.show()

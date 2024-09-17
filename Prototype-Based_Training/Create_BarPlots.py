import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

layers_names = [
    "Base Line",
    "1st Stage",
    "2nd Stage",
    "3rd Stage",
    "4th Stage",
    "5th Stage",
    "6th Stage",
    "7th Stage",
    "8th Stage",
    "9th Stage",
    "10th Stage",
    "11th Stage",
    "12th Stage",
    "13th Stage"
]

accuracy_values = [
    0.84211,
    0.85646,
    0.87560,
    0.88517,
    0.90431,
    0.90909,
    0.91388,
    0.90431,
    0.92823,
    0.90431,
    0.89952,
    0.91866,
    0.92345,
    0.92823    
]

layers_names.reverse()
accuracy_values.reverse()

accuracy_values = [round(i * 100, 2) for i in accuracy_values]

fig = plt.figure(figsize = (20, 15))
 
# creating the bar plot
bars = plt.barh(layers_names, accuracy_values)

ax = plt.gca()

ax.bar_label(bars)
# for i, v in enumerate(accuracy_values):
#     ax.text(v+0.5, i-0.12, str(v), color='blue', fontweight='bold')

# ax.tick_params(axis='x', labelrotation = 90)

plt.xlim([84, 95.5])
 
plt.ylabel("Fine Tuning Stages")
plt.xlabel("Accuracy (%)")

import tikzplotlib
tikzplotlib.save("/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/oi.tex")

plt.savefig("/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/oi.png")

import sys
sys.exit()

num_prot_NonCovid = [
    54,
    54,
    61,
    50,
    21,
    16,
    9,
    45,
    10,
    47,
    40,
    40,
    19,
    9
]

num_prot_Covid = [
    53,
    47,
    41,
    43,
    37,
    29,
    15,
    42,
    20,
    42,
    45,
    46,
    27,
    9
]

layers_names.reverse()
num_prot_NonCovid.reverse()
num_prot_Covid.reverse()

ind = np.arange(len(layers_names))
width = 0.4



fig, ax = plt.subplots()
ax.barh(ind + width, num_prot_NonCovid, width, color='chartreuse', label='Non-Covid')
ax.barh(ind, num_prot_Covid, width, color='red', label='Covid')

ax.set(yticks=ind + 0.5*width, yticklabels=layers_names, ylim=[2*width - 1, len(layers_names)])
ax.legend(loc='lower right')

plt.ylabel("Fine Tuning Stages")
plt.xlabel("Number of Prototypes")

for bars in ax.containers:
    ax.bar_label(bars)

plt.xlim([8, ])

import tikzplotlib
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
        
tikzplotlib_fix_ncols(plt.gcf())
tikzplotlib.save("/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/oi.tex")

plt.savefig("/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/oi.png")
import matplotlib.pyplot as plt
import numpy as np

sleep_score = 97
color_text  = '#3E738D'
color_deep  = "#044A9A"
color_light = "#1878CF"
color_rem   = "#9D0FB1"
color_awake = "#EB79D2"

plt.rcParams['figure.facecolor'] = "white" #cycler(color="#F0F2F6")
size_of_groups=[25,25,35,15]
wedgeprops = {"linewidth": 1, "edgecolor": "white"}
plt.close('all')
plt.figure()
plt.pie(size_of_groups, 
        colors=[color_deep, color_light, color_rem, color_awake], 
        startangle=90,
        counterclock=False,
        wedgeprops=wedgeprops)
my_circle=plt.Circle( (0,0), 0.8, color="white")
p=plt.gcf()
p.gca().add_artist(my_circle)
if sleep_score is None:
    plt.text(0, 0, "No data", fontsize=30, color=color_text,
             horizontalalignment='center')
else:
    plt.text(0, 0.2, (str(sleep_score) + '/100'), fontsize=30, color=color_text,
             horizontalalignment='center')
    plt.text(0, -.2, 'Quality', fontsize=20, color=color_text,
             horizontalalignment='center')
    plt.text(0, -.5, 'Good', fontsize=20, color=color_text,
             horizontalalignment='center')
# plt.savefig("assets/donut_sleep.png", transparent=True)
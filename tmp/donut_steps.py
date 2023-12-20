import matplotlib.pyplot as plt
import numpy as np


steps_score = 89
color_st_background = "white"
color_text = '#3E738D'
plt.rcParams['figure.facecolor'] = color_st_background #cycler(color="#F0F2F6")
size_of_groups=[steps_score, 100-steps_score]
wedgeprops = {"linewidth": 1, "edgecolor": "white"}
plt.close('all')
plt.figure()
plt.pie(size_of_groups, 
        colors=['green', "#e8e8e8"], 
        startangle=90,
        counterclock=False,
        wedgeprops=wedgeprops)
my_circle=plt.Circle( (0,0), 0.8, color=color_st_background)
plt.text(0, 0, (str(steps_score) + '%'), fontsize=40, color=color_text,
         horizontalalignment='center')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.savefig("assets/donut_steps.png", transparent=True)
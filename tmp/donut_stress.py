import matplotlib.pyplot as plt
import numpy as np

stress_score = 37
color_st_background = "white"
color_text      = '#3E738D'
color_rest      = "#4594F3"
color_low       = "#FFAF54"
color_medium    = "#F97516"
color_high      = "#DD5809"
 
plt.rcParams['figure.facecolor'] = color_st_background #cycler(color="#F0F2F6")
size_of_groups=[25,25,35,15]
wedgeprops = {"linewidth": 1, "edgecolor": "white"}
plt.close('all')
plt.figure()
plt.pie(size_of_groups, 
        colors=[color_rest, color_low, color_medium, color_high], 
        startangle=90,
        counterclock=False,
        wedgeprops=wedgeprops)
my_circle=plt.Circle( (0,0), 0.85, color=color_st_background)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.text(0, 0.2, (str(stress_score)), fontsize=40, color=color_text,
         horizontalalignment='center',
         verticalalignment='center')
plt.text(0, -.3, 'Overall', fontsize=30, color=color_text,
         horizontalalignment='center')
plt.savefig("assets/donut_stress.png", transparent=True)
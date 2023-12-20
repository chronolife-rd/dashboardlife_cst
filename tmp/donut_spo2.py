import matplotlib.pyplot as plt
import numpy as np

score = 93
lowest_score = 89
color_st_background = "white"
color_text = '#3E738D'
color_green      = "#17A444"
color_low       = "#F8CB4B"
color_medium    = "#F77517"
color_high      = "#CE4A14"

plt.rcParams['figure.facecolor'] = color_st_background #cycler(color="#F0F2F6")
# create data
size_of_groups=[4,23,23,23,23,4]
wedgeprops = {"linewidth": 1, "edgecolor": "white"}
plt.close('all')
plt.figure()
plt.pie(size_of_groups, 
        colors=[color_st_background, color_green, color_low, color_medium, color_high, color_st_background], 
        startangle=270,
        wedgeprops=wedgeprops)
my_circle=plt.Circle( (0,0), 0.85, color=color_st_background)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.text(0, 0.20, (str(score) + '%'), fontsize=40, color=color_text,
         horizontalalignment='center')
plt.text(0, -0.20, 'Lowest', fontsize=20, color=color_text,
         horizontalalignment='center')
plt.text(0, -0.50, (str(lowest_score) + '%'), fontsize=20, color=color_text,
         horizontalalignment='center')

# setting the axes projection as polar
plt.axes(projection = 'polar')
radius = 0.75

score_reshape = ((100 - 0)/(100-60)) * (score - 60)
deg = (270-(4/100*360)) - score_reshape/100*(92/100*360)
rad = np.deg2rad(deg)

if score < 70:
    color_spo2_score = color_high
elif 70 <= score < 80:
    color_spo2_score = color_medium
elif 80 <= score < 90:
    color_spo2_score = color_low
elif 90 <= score <= 100:
    color_spo2_score = color_green
    
plt.polar(rad, radius, '.', markersize=75, color=color_spo2_score)
plt.polar(rad, 1, '.', color=color_st_background)
plt.axis('off')
plt.savefig("assets/donut_spo2.png", transparent=True)
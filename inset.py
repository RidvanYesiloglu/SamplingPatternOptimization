# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:46:35 2020

@author: ridvan
"""
import matplotlib.pyplot as plt
import numpy as np

#from matplotlib.transforms import blended_transform_factory
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#
#fig, ax = plt.subplots() 
#ax.imshow(np.arange(256)*np.ones((256,256))/256.0)
#ax.axis('off')
#fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
#
#transform = blended_transform_factory(fig.transFigure, ax.transAxes)
#axins = zoomed_inset_axes(ax,2.5,bbox_to_anchor=(0, -0.2, 1, 0.5), bbox_transform=transform, loc=8, borderpad=0)
#axins.imshow(np.arange(256)*np.ones((256,256))/256.0)
#
#x1, x2, y1, y2 = 50, 60, 150, 160 # specify the limits
#axins.set_xlim(x1, x2) # apply the x-limits
#axins.set_ylim(y1, y2) # apply the y-limits
#axins.axis('off')
#mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")



fig, ax = plt.subplots() 
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.imshow(np.arange(256)*np.ones((256,256))/256.0)

t = "Number of taken pixels: 2134\nUndersampling rate: 0.125\nNumber of symmetric pixels: 122"
#plt.text(4, 1, t, ha='left', rotation=15, wrap=True)
#plt.text(6, 5, t, ha='left', rotation=15, wrap=True)
#plt.text(5, 5, t, ha='right', rotation=-15, wrap=True)
#plt.text(5, 10, t, fontsize=18, style='oblique', ha='center', va='top', wrap=True)
#plt.text(3, 4, t, family='serif', style='italic', ha='right', wrap=True)
plt.text(128, 350, t, ha='center', rotation=0, wrap=True)

plt.show()
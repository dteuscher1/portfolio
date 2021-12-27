---
date: "2021-12-23T00:00:00Z"
external_link: ""
image:
  caption: Photo by Yulian Alexeyev on Unsplash
  focal_point: Smart
summary: Analysis that combines of machine learning and statistical modeling methods to create irrigation zones.
tags:
- Neural Networks
- Deep Learning
- Machine Learning
- Bayesian Statistics
- Spatial Statistics
title: Creating Irrigation Zones using Bayesian Neural Networks
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
---

This project was used as my Master's project at BYU. I will provide information about the details and will include some technical information, but not everything will be included here. I have a formal report and it is currently in progress of being published. When it is published, the article title and the journal it is published it will be provided. Until this happens, if you are interested in further detail about the project, feel free to contact me and I can share what I have currently before it is published.

# Background 

The management of agricultural fields (i.e. farming) in the era of data science has evolved to use spatial mapping, remote sensing, soil and terrain measurements, weather measurements and other data sources to improve the quantity and quality of crops. The access to rich amounts of data provides opportunities for analysis that can inform decision making such as the amount of water and fertilizer to apply at any given time at any given location. The management of agricultural fields using advanced data analytics is referred to, collectively, as precision agriculture. Broadly, precision agriculture attempts to use the spatial variability within a field to manage individual crop locations rather than treat a field as spatially and temporally homogeneous.  Given that agriculture fields often occupy multiple acres of space, precision agriculture has been shown to outperform basic farming techniques by increasing crop yield.


Crop yield is heavily driven by the volumetric water content (VWC; the ratio of the volume of water to the unit volume of soil) and, in arid regions, irrigation is a practiced method for controlling and adjusting the VWC. As such, variable rate irrigation (VRI) is a practice within precision agriculture focused on using data to adjust the amount of water applied throughout the field according to spatial and temporal variations. One approach to VRI is to partition the field into management zones (or irrigation zones) wherein irrigation rates are adjusted within each management zones rather than utilizing a constant irrigation rate throughout the entire field 

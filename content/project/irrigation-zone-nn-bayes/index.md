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

Effective irrigation zones should partition the agricultural field based on VWC. However, obtaining VWC is a labor intensive process and is often only sparsely measured for the entire field. As an example, Figure 1 below shows 66 VWC measurements averaged over four time periods between April and September 2019 for an agricultural field of winter wheat in Rexburg, Idaho. Although the VWC was recorded at 66 different locations, the available data do not adequately cover the field to the point where precise irrigation decisions and zones can be defined. 

`{{< figure src="satellite_points.jpeg" title="Figure 1">}}`


Rather than using the VWC directly, agricultural scientists have begun to use alternative, more easily obtained, data to inform irrigation zones. Examples of such possible data for the Rexburg field considered in this research are shown in Figure 2 and include historical yields, normalized difference vegetation index (NDVI) and elevation (to name a few). These alternative data to VWC were recorded for over 5000 unique locations in the field using simple remote sensing mounted on drones or tractors making these data readily available to farmers.

`{{< figure src="field.jpg" title="Figure 2">}}`

The primary issue of using alternative data to VWC such as that in Figure 2 to define VRI zones is that such data is non-linearly related to VWC and, hence, may result less water efficient zones than ones defined by VWC. Figure 3 below shows scatterplots of a few of the variables against the VWC for the Rexburg field demonstrating clear non-linear relationships. In fact, the scatterplots in Figure 3 show a highly complex relationship between these alternative data and VWC suggesting that zones defined on these data may vary considerably from zones defined on VWC.

`{{< figure src="field.jpg" title="Figure 3">}}`

# Proposed Model

# Model Specification

# Parameter Estimation

# Results

# Conclusions
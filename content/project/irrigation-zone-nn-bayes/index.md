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

This project was used as my Master's project at BYU with the help of Dr. Matt Heaton and Dr. Neil Hansen. I will provide information about the details and will include some technical information, but not everything will be included here. I have a formal report and it is currently in progress of being published. When it is published, the article title and the journal it is published it will be provided. Until this happens, if you are interested in further detail about the project, feel free to contact me and I can share what I have currently before it is published.

# Background 

The management of agricultural fields (i.e. farming) in the era of data science has evolved to use spatial mapping, remote sensing, soil and terrain measurements, weather measurements and other data sources to improve the quantity and quality of crops. The access to rich amounts of data provides opportunities for analysis that can inform decision making such as the amount of water and fertilizer to apply at any given time at any given location. The management of agricultural fields using advanced data analytics is referred to, collectively, as precision agriculture. Broadly, precision agriculture attempts to use the spatial variability within a field to manage individual crop locations rather than treat a field as spatially and temporally homogeneous.  Given that agriculture fields often occupy multiple acres of space, precision agriculture has been shown to outperform basic farming techniques by increasing crop yield.


Crop yield is heavily driven by the volumetric water content (VWC; the ratio of the volume of water to the unit volume of soil) and, in arid regions, irrigation is a practiced method for controlling and adjusting the VWC. As such, variable rate irrigation (VRI) is a practice within precision agriculture focused on using data to adjust the amount of water applied throughout the field according to spatial and temporal variations. One approach to VRI is to partition the field into management zones (or irrigation zones) wherein irrigation rates are adjusted within each management zones rather than utilizing a constant irrigation rate throughout the entire field 

Effective irrigation zones should partition the agricultural field based on VWC. However, obtaining VWC is a labor intensive process and is often only sparsely measured for the entire field. As an example, Figure 1 below shows 66 VWC measurements averaged over four time periods between April and September 2019 for an agricultural field of winter wheat in Rexburg, Idaho. Although the VWC was recorded at 66 different locations, the available data do not adequately cover the field to the point where precise irrigation decisions and zones can be defined. 

`{{< figure src="satellite_points.jpeg" title="Figure 1">}}`


Rather than using the VWC directly, agricultural scientists have begun to use alternative, more easily obtained, data to inform irrigation zones. Examples of such possible data for the Rexburg field considered in this research are shown in Figure 2 and include historical yields, normalized difference vegetation index (NDVI) and elevation (to name a few). These alternative data to VWC were recorded for over 5000 unique locations in the field using simple remote sensing mounted on drones or tractors making these data readily available to farmers.

`{{< figure src="field.jpg" title="Figure 2">}}`

The primary issue of using alternative data to VWC such as that in Figure 2 to define VRI zones is that such data is non-linearly related to VWC and, hence, may result less water efficient zones than ones defined by VWC. Figure 3 below shows scatterplots of a few of the variables against the VWC for the Rexburg field demonstrating clear non-linear relationships. In fact, the scatterplots in Figure 3 show a highly complex relationship between these alternative data and VWC suggesting that zones defined on these data may vary considerably from zones defined on VWC.

`{{< figure src="nonlinear.jpg" title="Figure 3">}}`

The goals of this project and analysis are to use easily to obtain covariates to create irrigation zones that are related to VWC, while accounting for the spatial correlation that exists in the field and modeling the complex relationship that exists between the covariates and VWC. The proposed model integrates deep learning into a statistical modeling framework for irrigation zones to capture the complex relationship between the covariate data and the response variable.

# Proposed Model

Deep learning is a subfield of machine learning focused on using highly flexible algorithms called neural networks to learn complex relationships between covariates and response variables. However, from a statistical perspective, deep learning algorithms do not inherently account for uncertainty in the predictions and are often so complex that they are uninterpretable. The model proposed embeds a deep neural network into
a statistical model and estimates associated parameters using a Bayesian paradigm allowing for uncertainty quantification in the resulting fit. Further, we implement Bayesian versions of partial dependence plots to give some interpretability to the deep neural network between the covariates and the response variable.

With a deep neural network accounting for the complex relationships between the
covariates and response, our model also creates smooth irrigation zones by incorporating spatial correlation into our model. Specifically, we use spatial basis functions distinctly from the neural network portion of the model thereby successfully merging deep learning into a spatial modeling framework. Additionally, the spatial basis functions increase computational efficiency of the Markov chain Monte Carlo algorithm to sample from the posterior distribution of all model parameters.

The primary quantity of interest in this research is the irrigation zone of each location on the field - not necessarily the VWC. Admittedly, we could predict the VWC first and then create irrigation zones by splitting based on predicted VWC quantiles. However, here we first delineate zones based on the quantiles of the observed VWC and then perform modeling for the associated zone. Figure 4 below shows the discretized zones, which represents our response variable.

`{{< figure src="field_zones.jpg" title="Figure 4">}}`

While this represents loss in information for VWC, we advocate discretizing the response in this way for a few reasons. First, our data is unique in that we have the exact VWC measurements. However, given the cost, obtaining exact VWC data is exceptionally rare in agricultural practice. Rather, most agriculture fields are able to only collect ambiguous VWC levels of “low” or “high.” By transforming our response variable from VWC to irrigation zone, our statistical model will be more aligned with typical data available in other crop fields. That is, by discretization, exact VWC need not be recorded to follow this same methodology to delineate irrigation zones in other fields and our research will be more widely applicable. 

Second, VRI technology is typically done by applying “less than average” water to some locations and “more than average” to others. Hence, what is required to implement VRI is knowledge of which locations are “low” and which are “high” rather than the exact VWC.

Given the discretization here to irrigation zones corresponding to VWC levels, the response variable is thus an ordered multinomial response. Under this
discretization, we build our model using a latent variable approach to facilitate Bayesian computation. While others have developed statistical models for neural networks, to our knowledge, this is the first attempt at building a deep spatial statistical model for a non-Gaussian response.

The next sections will outline the model that is used and how the parameters were estimated, followed by the results of the model and the conclusions that can be drawn from it. 

**Note that there is a lot of background information provided in order to understand the problem and the approach we took, but I will provide less of the details and justifications in the next sections and simply explain what we did.**

# Model Specification

Let $Y(\mathbf{s}) \in \{1,\dots,R\}$ denote the irrigation zone for location $\mathbf{s} = (s_1, s_2)' \in \mathbb{R}^2$ where $R$ is the number of desired irrigation zones for a field.  Given the current state of variable rate irrigation systems, usually $R$ will only be as high as 4 or 5. Due to the difficulty of performing spatial analysis on a discrete scale, we augment $Y(\mathbf{s})$ with a latent variable $Z(\mathbf{s}) \in \mathbb{R}$ such that

\begin{align}
    Y(\mathbf{s}) = \sum_{r=1}^R r {1}(c_{r-1} \leq Z(\mathbf{s}) < c_{r})
    \label{responseMod}
\end{align}

where ${1}(\cdot)$ is an indicator function and $c_0 = -\infty < c_1 = 0 < c_2< \dots < c_R = \infty$ are cut points used to determine the probability that location $\mathbf{s}$ belongs to a specific zone.

Figure 5 below illustrates the benefit of using the latent variable, $Z(\mathbf{s})$. The cut points, $c_1$ and $c_2$, are the same for each location, but the probabilities of being in a specific zone are different depending on the mean of $Z(\mathbf{s})$, so the probabilities of being in each irrigation zone is different for each location.

`{{< figure src="normal_dist.jpg" title="Figure 5">}}`

With this specification, we can assume that 
\begin{align}
    Z(\mathbf{s}) &\sim \mathcal{N}\left(f_L(\mathbf{X}(\mathbf{s})) + w(\mathbf{s}), 1\right)
    \label{latentMod}
\end{align}

where $f_L(\mathbf{X}(\mathbf{s}))$ is the univariate output function of an $L$-layer feed forward neural network (FFNN) with input covariates $\mathbf{X}(\mathbf{s}) = (X_1(\mathbf{s}),\dots, X_P(\mathbf{s}))'$ and $w(\mathbf{s})$ is a spatial random effect.

This specific model is used because it accounts for the challenges previously mentioned. The FFNN can model the complex relationship between the covariates and the associated irrigation zones and the spatial random effect, $w(\mathbf{s})$, smoothes the predicted zones so they can be implemented by VRI. 
# Parameter Estimation

# Results

# Conclusions
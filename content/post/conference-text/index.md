---
date: "2022-02-12T00:00:00Z"
external_link: ""
image:
  caption: Photo from The Church of Jesus Christ of Latter-Day Saints
  focal_point: Smart
summary: Webscraping and analysis of text from talks from General Conferences of the Church of Jesus Christ of Latter-Day Saints with a Shiny app to explore the talks.
tags:
- Text analysis
- Text processing
- Webscraping
- Shiny
title: Text Analysis and Processing of General Conference Talks from The Church of Jesus Christ of Latter-Day Saints
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
---


This project takes text from talks from General Conferenece of the Church of Latter-Day Saints and processes and analyzes the text. The processed data can be explored with this [Shiny app](https://david-teuscher.shinyapps.io/conference-analysis-app/). The code for scraping the data and the analysis can be found [here](https://github.com/skylerg022/conference-church-of-jesus-christ). 

Every 6 months The Church of Jesus Christ of Latter-Data Saints holds their General Conference, where church leaders speak for approximately 10 hours of the course of two days. The messages vary from speaker to speaker and from conference to conference. The text for these talks are available from as early as April 1971 through the most recent conference in October 2021. With the large amount of text from a variety of speakers, I was interested in how the usage of words and topics changed over time as well as common words for each speaker and conference. 

The project is a combination for work from [Skyler Gray](https://github.com/skylerg022), [Daniel Garrett](https://github.com/GarrettDaniel), [Mckay Gerratt](https://github.com/germckay), and myself. 

The data was web scraped from the official site of The Church of Jesus Christ of Latter-Day Saints using Beautiful Soup and Selenium in Python. There is a page with a similar format for each conference. An example of the page to be scraped is [here](https://www.churchofjesuschrist.org/study/general-conference/1971/04?lang=eng). On this page, it lists the title for each talk, the speaker's name, and possibly a small excerpt from the talk. For each talk, we extracted the speaker's name as well as the `href` elements that corresponded to the link to the actual talk. 

After collecting all of the links, we looped through all of the links and pulled the title of the talk and the text of the talk. The resulting information was saved into a pandas DataFrame that contained the year and month of the conference, the speaker's name, the talk title, and the text of the talk. Afterwards, the data was written to a .csv file to be used in R for further analysis. 

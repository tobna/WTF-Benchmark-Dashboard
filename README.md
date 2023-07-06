# What Transformer to Favor: Website
This is the repository that hosts the interactive website accompanying the paper [What Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers] (Link coming soon), a benchmark of over 30 different efficient vision trainsformers.

For the benchmark's code visit [this repository](https://gitfront.io/r/user-5921586/dmRcCBtFqbtK/WhatTransformerToFavor/).

## Technical Background
The website is created using a local [Dash app](https://dash.plot.ly/). 
It can be converted into a static website using a [Makefile](Makefile), as it only uses [clientside callbacks](https://dash.plotly.com/clientside-callbacks).
This version automatically deployed to github pages using github actions. 

## Requirements
The requirements are listed in [requirements.txt](requirements.txt). 
To install them, run
```commandline
pip3 install -r requirements.txt
```

## Local Deployment
To run this project locally, start the Dash server by running
```commandline
python3 app.py
```
and then visit http://127.0.0.1:8050 in your browser.


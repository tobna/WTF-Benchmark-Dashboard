# Which Transformer to Favor: Website
This is the repository that hosts the interactive website accompanying the paper [Which Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers](https://arxiv.org/abs/2308.09372), a benchmark of over 30 different efficient vision trainsformers.

For the benchmark's code visit [this repository](https://github.com/tobna/WhatTransformerToFavor).

This implementation was inspired by the [interactive covid dashboard](https://github.com/covid19-dash/covid-dashboard), which is hosted in the same way.

## Technical Background
The website is created using a local [Dash app](https://dash.plot.ly/). 
It can be converted into a static website using a [Makefile](Makefile), as it only uses [clientside callbacks](https://dash.plotly.com/clientside-callbacks).
This version is automatically deployed to github pages using github actions. [Link](https://transformer-benchmark.github.io/) 

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

## License
We release this code under the [MIT License](LICENSE).

## Citation
If you use the data or code in this repository for your project, please cite our paper:
```BibTeX
@misc{Nauen2023WTFBenchmark,
      title={Which Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers}, 
      author={Tobias Christian Nauen and Sebastian Palacio and Andreas Dengel},
      year={2023},
      eprint={2308.09372},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


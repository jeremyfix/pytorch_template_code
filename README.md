# Template base code for pytorch

This repository contains a template base code for a complete pytorch pipeline.

This is a template because it works on fake data but aims to illustrate some pythonic syntax allowing the pipeline to be modular.

More specifically, this template base code aims to target :

- modularity : allowing to change models/optimizers/ .. and their hyperparameters easily
- reproducibility : saving the commit id, ensuring every run saves its assets in a different directory, recording a summary card for every experiment, building a virtual environnement

For the last point, if you ever got a good model as an orphane pytorch tensor whithout being able to remember in which conditions, with which parameters and so on you got, you see what I mean. 



# GammaGMM

`GammaGMM` is a GitHub repository containing the **gammaGMM** [1] algorithm. It refers to the paper titled *Estimating the Contamination Factor's Distribution in Unsupervised Anomaly Detection*.

Check out the pdf here: [[pdf](https://openreview.net/pdf?id=pf3NihScj1)].

## Abstract

Anomaly detection methods identify examples that do not follow the expected behaviour, typically in an unsupervised fashion, by assigning real-valued anomaly scores to the examples based on various heuristics. These scores need to be transformed into actual predictions by thresholding so that the proportion of examples marked as anomalies equals the expected proportion of anomalies, called contamination factor. Unfortunately, there are no good methods for estimating the contamination factor itself. We address this need from a **Bayesian perspective**, introducing a method for estimating the posterior distribution of the contamination factor for a given unlabeled dataset. We leverage several anomaly detectors to capture the basic notion of anomalousness and estimate the contamination using a specific mixture formulation. Empirically on 22 datasets, we show that the estimated distribution is **well-calibrated** and that setting the threshold using the posterior mean improves the detectors’ performance over several alternative methods

## Contents and usage

The repository contains:
- gammaGMM.py, a function that allows to get samples from the contamination factor's posterior distribution;
- Notebook.ipynb, a notebook showing how to use gammaGMM on an artificial 2D dataset;
- results, a folder that contains the samples that we obtained after running our code along with the true contamination factors;
- online_supplement, a pdf with the online supplementary material.

To use gammaGMM, import the github repository or simply download the files. You can find the benchmark datasets at this [[link](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)]. Alternatively, feel free to use directly our results (i.e., the samples from the posterior) that you can find inside the results folder.


## EXample-wise ConfidEncE of anomaly Detectors (ExCeeD)

Given a dataset with attributes **X**, an unsupervised anomaly detector assigns to each example an anomaly score, representing its degree of anomalousness. Thus, the first step of gammaGMM is to use a set of M unsupervised detectors (passed as input by the user) to transform the data into an M dimensional score space. Then, it sets a DPGMM model on this score space. Each component of the DPGMM is ordered using our proposed ordering criterium. By measuring how anomalous the components are (jointly), we derive the contamination factor's posterior.

Given a training dataset **X** and the user-specified hyperparameters p0 and phigh, the code can be used as in the Notebook file.

## Dependencies

The `gammaGMM` function requires the following python packages to be used:
- [Python 3.9](http://www.python.org)
- [Numpy 1.21.0](http://www.numpy.org)
- [Pandas 1.4.1](https://pandas.pydata.org/)
- [PyOD 1.1.0](https://pyod.readthedocs.io/en/latest/install.html)


## Contact

Contact the author of the paper: [lorenzo.perini@kuleuven.be](mailto:lorenzo.perini@kuleuven.be).


## References

[1] Perini, L., Burkner, P., Klami, A.: *Estimating the Contamination Factor's Distribution in Unsupervised Anomaly Detection.* In: The Fortieth International Conference on Machine Learning (ICML) 2023.

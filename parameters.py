from collections import OrderedDict
import numpy as np
import os


### parameters
DPI = 300
oneDocument = False



nRepeats = 1
nCV = 10
nInnerCV = 10


basePath = "/home/aydin/results/radPreFilter"
ncpus = 16

dList = [ "Desmoid", "Lipo", "Liver", "Melanoma", "GIST", "CRLM", "HN"]
dList = sorted(dList)


CT_datasets = ['HN', 'GIST', 'CRLM', 'Melanoma']
MR_datasets = ['Lipo', 'Desmoid', 'Liver']


imagePath = "/data/WORCDatabase/Data/worc"
HNPath = "/data/WORCDatabase/Data/worc/HN"
featuresPath = os.path.join(basePath, "features")
cachePath = os.path.join(basePath, "cache")
trackingPath = os.path.join(basePath, "mlrun")



filterNames = ["original_", "wavelet-", "squareroot_", "square_", "logarithm_",
        "log-sigma-", "lbp-3D", "gradient_", "exponential_"]


fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "FeatureSet":  filterNames + ["all"],
        "N": [1,2,4,8,16,32],
        "Methods": {
            "ET": {},
            "LASSO": {"C": [1.0]},
            "Anova": {},
            "Bhattacharyya": {},
            "MRMRe": {},
            "None": {}
        }
    }
})



clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "Constant": {},
            "RBFSVM": {"C":np.logspace(-6, 6, 7, base = 2.0), "gamma":["auto"]},
            "RandomForest": {"n_estimators": [50, 125, 250]},
            "LogisticRegression": {"C": np.logspace(-6, 6, 7, base = 2.0) },
            "NeuralNetwork": {"layer_1": [4, 16, 64], "layer_2": [4, 16, 64], "layer_3": [4, 16, 64]},
            "NaiveBayes": {}
        }
    }
})







#

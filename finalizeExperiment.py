#!/usr/bin/python3

from functools import partial
from datetime import datetime
import pandas as pd
from joblib import parallel_backend, Parallel, delayed, load, dump
import random
import numpy as np

#import pycm
from sklearn.calibration import CalibratedClassifierCV
import shutil
import pathlib
import os
import math
import random
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time
import copy
import random
import pickle
import tempfile
import itertools
import multiprocessing
import socket
from glob import glob
from collections import OrderedDict
import logging
import mlflow
from typing import Dict, Any
import hashlib
import json

from pprint import pprint
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel
#from skfeature.function.statistical_based import f_score, t_score
# from skfeature.function.similarity_based import reliefF
from sklearn.feature_selection import mutual_info_classif

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from sklearn.utils import resample

from mlflow import log_metric, log_param, log_artifact, log_dict, log_image
from mlflow.tracking.client import MlflowClient


#from loadData import *
from helpers import *
#from utils import *
from parameters import *
from extraFeatureSelections import *
from featureScoring import *
from nestedExperiment import *


### parameters
TrackingPath = "/home/aydin/results/radPreFilter/mlrun"


def getResults (expName):
    mlflow.set_tracking_uri(trackingPath)
    if os.path.exists("./results/"+str(expName)+".feather") == False:
        results = []
        current_experiment = dict(mlflow.get_experiment_by_name(expName))
        experiment_id = current_experiment['experiment_id']
        runs = MlflowClient().search_runs(experiment_ids=experiment_id, max_results=50000)
        for r in runs:
            row = r.data.metrics
            row["UUID"] = r.info.run_uuid
            row["Model"] = r.data.tags["Version"]
            row["Parameter"] = r.data.tags["pID"]

            row["Parameter"] = row["Parameter"]
            row["Model"] = row["Model"]

            row["FSel"], row["Clf"] = row["Model"].split("_")
            row["Dataset"] = d

            row["nFeatures"] = eval(row["Parameter"])[row["FSel"]]["nFeatures"]
            row["featureSet"] = eval(row["Parameter"])[row["FSel"]]["featureSet"]

            row["Path"] = os.path.join(trackingPath,  str(experiment_id), str(r.info.run_uuid), "artifacts")
            results.append(row)

            # read timings
            apath = os.path.join(row["Path"], "timings.json")
            with open(apath) as f:
                expData = json.load(f)
            row.update(expData)

            # read AUCs
            apath = os.path.join(row["Path"], "aucStats.json")
            with open(apath) as f:
                aucData = json.load(f)
            row.update(aucData)

        results = pd.DataFrame(results)
        print ("Pickling results")
        pickle.dump (results, open("./results/"+str(expName)+".feather","wb"))
    else:
        print ("Restoring results")
        results = pickle.load(open("./results/"+str(expName)+".feather", "rb"))

    return results



@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UserWarning)
def executeExperiment (fselExperiments, clfExperiments, X_train, y_train, X_test, y_test, test_index, dataID):
    mlflow.set_tracking_uri(TrackingPath)

    # experiment B
    raceOK = False
    while raceOK == False:
        try:
            mlflow.set_experiment(dataID + "_final")
            raceOK = True
        except:
            time.sleep(0.5)
            pass

    stats = {}
    for i, fExp in enumerate(fselExperiments):
        np.random.seed(i)
        random.seed(i)
        for j, cExp in enumerate(clfExperiments):
            timings = {}

            # fake
            k = "final"

            # only take those features we need
            X_train, y_train = selectFeatureSubset (X_train, y_train, fExp)
            X_test, y_test = selectFeatureSubset (X_test, y_test, fExp)

            foldStats = {}
            foldStats["features"] = []
            foldStats["params"] = {}
            foldStats["params"].update(fExp)
            foldStats["params"].update(cExp)
            #foldStats["params"].update({"Experiment": "B"})
            run_name = getRunID (foldStats["params"])

            current_experiment = dict(mlflow.get_experiment_by_name(dataID + "_final"))
            experiment_id = current_experiment['experiment_id']

            # log what we do next
            with open(os.path.join(TrackingPath, "curExperiments.txt"), "a") as f:
                f.write("(RUN) " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + str(fExp) + "+" + str(cExp) + "+final\n")


            expVersion = '_'.join([k for k in foldStats["params"] if "Experiment" not in k])
            pID = str(foldStats["params"])

            # register run in mlflow now
            run_id = getRunID (foldStats["params"])
            mlflow.start_run(run_name = run_id, tags = {"Version": expVersion, "pID": pID})
            # this is stupid, but well, log a file with name=runid
            log_dict(foldStats["params"], run_id+".ID")

            fselector = createFSel (fExp)
            with np.errstate(divide='ignore',invalid='ignore'):
                timeFSStart = time.time()
                fselector.fit (X_train.copy(), y_train.copy())
                timeFSEnd = time.time()
                timings["Fsel_Time_Fold_" + str(k)] = timeFSEnd - timeFSStart
            feature_idx = fselector.get_support()
            selected_feature_names = X_train.columns[feature_idx].copy()
            all_feature_names = X_train.columns.copy()

            #log_dict({"Index": feature_idx.tolist()}, "FIndex_"+str(k)+".json")

            # log also 0-1
            fpat = np.zeros(X_train.shape[1])
            for j,f in enumerate(feature_idx):
                fpat[j] = int(f)

            # apply selector-- now the data is numpy, not pandas, lost its names
            X_fs_train = fselector.transform (X_train)
            y_fs_train = y_train

            X_fs_test = fselector.transform (X_test)
            y_fs_test = y_test

            # check if we have any features
            if X_fs_train.shape[1] > 0:
                classifier = createClf (cExp)

                timeClfStart = time.time()
                classifier.fit (X_fs_train, y_fs_train)
                timeClfEnd = time.time()
                timings["Clf_Time_Fold_" + str(k)] = timeClfEnd - timeClfStart

                y_pred = classifier.predict_proba (X_fs_test)
                y_pred = y_pred[:,1]
                foldStats["fold_"+str(k)], df, acc = testModel (y_pred, y_fs_test, idx = test_index, fold = k)
            else:
                y_pred = y_test*0 + 1
                foldStats["fold_"+str(k)], df, acc = testModel (y_pred, y_fs_test, idx = test_index, fold = k)


            stats[str(i)+"_"+str(j)] = logMetrics (foldStats)
            log_dict(timings, "timings.json")
            mlflow.end_run()
            with open(os.path.join(TrackingPath, "curExperiments.txt"), "a") as f:
                f.write("(DONE)" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + str(fExp) + "+" + str(cExp) + "\n")




def executeExperiments (z):
    fselExperiments, clfExperiments, trainData_X, trainData_y, testData_X, testData_y, test_index, d = z
    executeExperiment ([fselExperiments], [clfExperiments], trainData_X, trainData_y, testData_X, testData_y, test_index, d)



if __name__ == "__main__":
    print ("Hi.")

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    # load data first
    # load old things

    statsfile = os.path.join("./results", "trainStats.joblib")
    trainStats = load(open(statsfile, "rb"))

    allStats = {}
    for d in dList:
        data = loadData(d, drop = False).reset_index(drop = True)
        print ("\tLoaded ", d, "with shape", data.shape)

        # preprocess at the beginning
        p = data["Patient"]
        y = data["Target"]
        X = data.drop(["Patient", "Target"], axis = 1).copy()
        X, y = preprocessData (X, y)
        data = X.copy()
        data["Target"] = y
        data["Patient"] = p
        print ("\tLoaded ", d, "with shape", data.shape)

        try:
            mlflow.set_tracking_uri(TrackingPath)
            mlflow.create_experiment(d+"_final")
            mlflow.set_experiment(d+"_final")
            time.sleep(3)
        except:
            pass

        print ("\nExecuting", d)
        besties = []
        clList = []
        kfolds = RepeatedStratifiedKFold(n_splits = nCV, n_repeats = 1, random_state = 42*1)
        for j, (train_index, test_index) in enumerate(kfolds.split(data["Patient"], data["Target"])):
            # avoid race conditions later

            # save outer fold just to be sure
            allStats[d + "_train_" + str(j)] = train_index
            allStats[d + "_test_" + str(j)] = test_index

            assert ((allStats[d + "_train_" + str(j)] == trainStats[d + "_train_" + str(j)]).all)
            assert ((allStats[d + "_test_" + str(j)] == trainStats[d + "_test_" + str(j)]).all)

            trainData_X = data.iloc[train_index].reset_index(drop = True)
            trainData_y = data["Target"].iloc[train_index].reset_index(drop = True)
            testData_X = data.iloc[test_index].reset_index(drop = True)
            testData_y = data["Target"].iloc[test_index].reset_index(drop = True)
            trainData_X = trainData_X.drop(["Patient", "Target"], axis = 1)
            testData_X = testData_X.drop(["Patient", "Target"], axis = 1)

            results = getResults (d+"_" + str(j))
            for fset in set(results["featureSet"]):
                fsetresults = results.query("featureSet == @fset").sort_values("AUC", ascending = False).reset_index(drop = True).copy()
                params = eval(fsetresults.iloc[0]["Parameter"])
                fsel = fsetresults.iloc[0]["FSel"]
                fsel = [(fsel, params[fsel])]
                clf = fsetresults.iloc[0]["Clf"]
                clf = [(clf, params[clf])]

                clList.append( (fsel, clf, trainData_X, trainData_y, testData_X, testData_y, test_index, d))

        # execute
        ncpus = 24
        with parallel_backend("loky", inner_max_num_threads=1):
            fv = Parallel (n_jobs = ncpus)(delayed(executeExperiments)(c) for c in clList)

    # statsfile = os.path.join("./results", "trainStats.joblib")
    # p_fsel = dump(allStats, statsfile)

#

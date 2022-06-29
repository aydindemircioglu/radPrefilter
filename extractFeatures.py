import random
import pandas as pd
from radiomics import featureextractor
import json
import argparse
import logging
import math
import numpy as np

from helpers import *
from parameters import *


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # keep everyone happy
        self.parser.add_argument('-f', type=str, default="dummy", help='dummy parameter')

        self.initialized = True


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt



internalCache = {}
def cache_read_csv (csvFile):
    try:
        f = internalCache[csvFile]
    except:
        f = pd.read_csv(csvFile)
        internalCache[csvFile] = f.copy()
    return f



def getRadiomicFeaturesFromDataset (data, dataID, config = None):
    featureList = {}
    params = os.path.join("config", dataID  + ".yaml")

    featureList = {}
    for i, (idx, row) in enumerate(data.iterrows()):
        patID = row["Patient"]
        if patID in ["Lipo-044", "Desmoid-001"]:
            print ("SHOULD NOT HAPPEN: ### Ignoring ", patID)
            continue

        os.makedirs (cachePath, exist_ok = True)
        cacheFile = os.path.join(cachePath, "rad_" + dataID + "_" + patID + ".csv")
        if os.path.exists(cacheFile) == True:
            f = cache_read_csv(cacheFile)
        else:
            fImage, fMask = row["Image"], row["mask"]
            print ("Masks", fImage, fMask)
            try:
                extractor = featureextractor.RadiomicsFeatureExtractor(params)
                print ("\t ### Extracting ", patID)
                result = extractor.execute(fImage, fMask)
                # save it
                f = pd.DataFrame({k:[str(result[k])] for k in result})
                f.to_csv (cacheFile, index = False)
            except Exception as e:
                f = pd.DataFrame([{"ERROR": patID}])
                print ("#### GOT AN ERROR!", e)
                print(f)
                #raise Exception ("really wrong here?")

        # if we had a problem, we ignore it
        if f.shape[1] > 1:
            f["Patient"] = patID
            f["Diagnosis"] = row["Diagnosis"]
            featureList[patID] = f
        else:
            print(f)
            #raise Exception ("Something wrong here?")

    df = pd.concat([pd.DataFrame(featureList[patID]) for patID in featureList])
    df.index = df.Patient

    # remove those keys we do not need here
    df = df.drop([k for k in df.keys() if "diagnostics_" in k], axis = 1)
    return df




def extractFeatures (data, dataID):
    fName = os.path.join(featuresPath, "rad_" + str(dataID) + ".csv")
    if os.path.exists(fName) == True:
        # just load
        print ("Loading from cache:", fName)
        df = pd.read_csv(fName)
        return df
    df = getRadiomicFeaturesFromDataset (data, dataID, config = None)

    # save
    os.makedirs (featuresPath, exist_ok = True)
    df.to_csv(fName, index = False)
    pass



if __name__ == "__main__":
    print ("Hi.")

    # interprete command line options
    opt = BaseOptions().parse()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    for dataID in dList:
        print ("### Processing", dataID)
        data = getData (dataID)
        extractFeatures (data, dataID)

    # series = ["original_", "wavelet-", "squareroot_", "square_", "logarithm_",
    #     "log-sigma-", "lbp-3D", "gradient_", "exponential_"]
    # for s in series:
    #     z = pd.read_csv("/home/aydin/results/deepPreFilter/cache/rad_CRLM_CRLM-021.csv")
    #     sKeys = [k for k in z if s in k]
    #     oKeys = [k for k in z if "original_" in k]
    #     sKeys = list(set(sKeys) | set(oKeys))
    #     df = z[sKeys]
    #     print (s, df.shape)
#

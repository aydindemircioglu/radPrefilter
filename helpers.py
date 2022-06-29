from sklearn.metrics import roc_curve, auc, roc_auc_score
from math import sqrt
import numpy as np
import scipy.stats
from scipy import stats
import shutil
import os
from glob import glob
from pprint import pprint
from skimage.measure import label
import pandas as pd
from parameters import *
import cProfile
import pstats
from functools import wraps
import numpy as np
import random



def loadData (d, drop = True):
    # has no diagnosis, but target and patientID
    blacklist = pd.read_csv("./data/blacklist.csv").T.values[0]
    X = pd.read_csv(os.path.join(featuresPath, "rad_" + d + ".csv"))
    X["Target"] = X["Diagnosis"]
    X = X.drop(["Diagnosis"], axis = 1).reset_index(drop = True).copy()
    X = X.query("Patient not in @blacklist").copy()
    if drop == True:
        X = X.drop(["Patient"], axis = 1).reset_index(drop = True).copy()
    return X


def filterData (data, filter = None):
    if filter is not None:
        sKeys = [k for k in data if "original_" in k]
        for f in filter.split("+"):
            fKeys = [k for k in data if f in k]
            sKeys = list(set(sKeys) | set(fKeys))
        data = data[sKeys]
        print ("### Filtered data shape:", data.shape)
    return data.copy()


def getData (dataID):
    # load data first
    data = pd.read_csv("./data/pinfo_" + dataID + ".csv")

    # fix this
    if dataID == "HN":
        data["Diagnosis"] = data["Tstage"]
        data = data.drop (["Tstage"], axis = 1).copy()

    # add path to data
    for i, (idx, row) in enumerate(data.iterrows()):
        image, mask = getImageAndMask (dataID, row["Patient"])
        data.at[idx, "Image"] = image
        data.at[idx, "mask"] = mask
    print ("### Data shape", data.shape)

    # make sure we shuffle it and shuffle it the same
    np.random.seed(111)
    random.seed(111)
    data = data.sample(frac=1)

    return data



def getImageAndMask (d, patName):
    image = os.path.join(imagePath, patName, "image.nii.gz")
    if d in ["HN"]:
        # nifti only exists with CT, so PET will be ignored
        cands = glob(os.path.join(HNPath, patName + "*/**/image.nii.gz"), recursive = True)
        if len(cands) != 1:
            print ("Error with ", patName)
            print ("Checked", os.path.join(HNPath, patName + "*/**/image.nii.gz"))
            pprint(cands)
            raise Exception ("Cannot find image.")
        image = cands[0]
        cands = glob(os.path.join(HNPath, patName + "*/**/mask_GTV-1.nii.gz"), recursive = True)
        if len(cands) != 1:
            print ("Error with ", patName)
            pprint(cands)
            raise Exception ("Cannot find mask.")
        mask = cands[0]
    if d in ["Desmoid", "GIST", "Lipo", "Liver"]:
        mask = os.path.join(imagePath, patName, "segmentation.nii.gz")
    if d in ["CRLM"]:
        mask = os.path.join(imagePath, patName, "segmentation_lesion0_RAD.nii.gz")
    if d in ["Melanoma"]:
        mask = os.path.join(imagePath, patName, "segmentation_lesion0.nii.gz")

    # special cases
    if patName == "GIST-018":
        image = os.path.join(imagePath, patName, "image_lesion_0.nii.gz")
        mask = os.path.join(imagePath, patName, "segmentation_lesion_0.nii.gz")
    if patName == "Lipo-073":
        mask = os.path.join(imagePath, patName, "segmentation_Lipoma.nii.gz")

    if os.path.exists(image) == False:
        print ("Missing", image)
    if os.path.exists(mask) == False:
        print ("Missing", mask)
    return image, mask


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    largestCC = 255*largestCC
    return np.asarray(largestCC, dtype = np.uint8)

# find bounding box around mask
def getMaskExtend (sliceMask):
    def fix (v, b):
        v = 0 if v < 0 else v
        v = b-1 if v > b-1 else v
        return v
    x0 = np.min(np.where(np.sum(sliceMask,axis=0))) - 16
    x1 = np.max(np.where(np.sum(sliceMask,axis=0))) + 16
    y0 = np.min(np.where(np.sum(sliceMask,axis=1))) - 16
    y1 = np.max(np.where(np.sum(sliceMask,axis=1))) + 16
    x0, x1, y0, y1 = fix(x0, sliceMask.shape[1]), fix(x1, sliceMask.shape[1]), fix(y0, sliceMask.shape[0]), fix(y1, sliceMask.shape[0])
    return x0, x1, y0, y1

def extractMaskAndImage (sliceMask, slice):
    x0, x1, y0, y1 = getMaskExtend (sliceMask)
    return sliceMask[y0:y1, x0:x1], slice[y0:y1, x0:x1]



def findOptimalCutoff (fpr, tpr, threshold, verbose = False):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    fpr, tpr, threshold

    Returns
    -------
    list type, with optimal cutoff value

    """

    # own way
    minDistance = 2
    bestPoint = (2,-1)
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p

    if verbose == True:
        print ("BEST")
        print (minDistance)
        print (bestPoint)
    sensitivity = bestPoint[1]
    specificity = 1 - bestPoint[0]
    return sensitivity, specificity


# https://towardsdatascience.com/how-to-profile-your-code-in-python-e70c834fad89
def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.

    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/

    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout

    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner




# https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
  """Return an axes of confidence bands using a simple approach.

  Notes
  -----
  .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
  .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

  References
  ----------
  .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
     http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

  """
  if ax is None:
      ax = plt.gca()

  ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
  ax.fill_between(x2, y2 + ci, y2 - ci, color="#111111", edgecolor=['none'], alpha =0.15)

  return ax


# Modeling with Numpy
def equation(a, b):
  """Return a 1D polynomial."""
  return np.polyval(a, b)



def full_extent(ax, pad=0.0):
  """Get the full extent of an axes, including axes labels, tick labels, and
  titles."""
  # For text objects, we need to draw the figure first, otherwise the extents
  # are undefined.
  ax.figure.canvas.draw()
  items = ax.get_xticklabels() + ax.get_yticklabels()
  items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
  items += [ax, ax.title]
  bbox = Bbox.union([item.get_window_extent() for item in items])
  return bbox.expanded(1.0 + pad, 1.0 + pad)


#


# The effect of preprocessing filters on predictive performance in radiomics

This is the code for the paper.


# Setup

Install all requirements by

$ pip install -r requirements.txt


## Experiment

Use the WORCDatabase to download all scans.
The folder  has to have the following structure:
  ./CRLM-001/
    ...
  ./Melanoma-052

The HCC dataset is different. Download and unzip it into a folder,
say ./HCC, e.g., ./HCC/HN1417_20190403_PET should exist.
then specify this directory in the parameters.py file

1. copy down WORC data into /data/WORC/
2. make sure the .csv for WORC is downloaded
3. we want to remove those with slice thickness > 2.0*median:
   `python3 ./processData.py`
   it wil also compute some infos about the data.
4. extract all features
   `python3 ./extractFeatures.py`
5. do the inner 10-fold CV:
   `python3 ./nestedExperiment.py`
6. retrain the best model, ie, apply outer 10-fold CV:
   `python3 ./finalizeExperiment.py`

## Evaluation

All output from the evaluation is cached in ./results. Therefore, the experiment
does not need to be executed to evaluate it.
   ``python3 ./evaluate.py``


# License

## Note: All data was previously published and have their own license. Please refer to the respective publications.

Other code is licensed with the MIT license:

Copyright (c) 2022 Aydin Demircioglu Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

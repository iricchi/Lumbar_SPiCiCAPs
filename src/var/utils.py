#!/usr/bin/python

"""
This script is used as utility functions collection for the AR model.

Author: Ilaria Ricchi
"""
import numpy as np
from scipy.signal import butter, filtfilt, cheby2, lfilter


def temporal_filter(timeseries, Wn, TR=2.5):
    """ img is the input img loaded with nibabel, Wn is the widnow for the bandpass:
    [0.01,0.13]
    """
    b,a = cheby2(4, 30, Wn, 'bandpass',fs=1/TR)
    filtered_data = filtfilt(b,a,timeseries)
    
    return filtered_data


def normalize_ts(TS, direction, demean=True, hp_filt=False, lp_filt=False, TR=2.5):
    
    # Preproc data
    mean = np.mean(TS, axis=direction)
    shape = (1, mean.shape[0]) if direction == 0 else (mean.shape[0], 1)
    
    # demean 
    if demean:
        TS -= mean.reshape(shape)
        
    # zscore the data
    std = np.std(TS,axis=direction)
    TS[std>0] /= std.reshape(shape)[std>0]
    
    if hp_filt:
        TS = temporal_filter(TS, [0.11,0.199],TR)
    elif lp_filt:
        TS = temporal_filter(TS, [0.001,0.05],TR)        

    return TS, mean, std


def ar_mls(TS,p, demean=False, direction=1, HP=False, LP=False, TR=2.5):
    """
    This function identifies an autoregressive (AR) model from multivariate time series using the multivariate least square approach.

    - Inputs:
        TS = origninal time courses encoded in a matrix of size (k=numb of variables/ROIs, T=num time points)
        p = order of the AR model to be identified
        demean = set to False cause already demeaned (in general) otherwise required
        direction set to 1 for our data
        TR =2.5 , not needed in general (no temporal filtering is wanted, but for testing...)

    The AR model of order p reads:

    TS(:,t) = w + \sum_i=1^p A_i * TS(:,t-i) + e(t),     Eq. (1)

    where w is an intercept of size (T,1) and it is supposed to be 0 if timecourses are centered
          A_i are the matrices of size k*k linking T(t,:) and T(t-i,:)
          e(t) is the error vector of size (T,1) following a centered multivariate gaussian distribution

    Eq. (1) can be written for all t \in [p+1...T]. Concatenating these
     (T-p) equations yields the following matrix form:

     Y = B*Z+E,    Eq. (2)

     where: Y = [TS(p+1,:) TS(p+2,:) ... TS(T,:)] is a matrix of size (k,T-p)

            B = [w A_1 ... A_p] is a matrix of size (k,k*p+1) that
            gathers unknown parameters of the model

                |    1             1           ...        1     |
                | TS(p,:)'     TS(p+1,:)'      ...    TS(T-1,:)'|
                |TS(p-1,:)'     TS(p,:)'       ...    TS(T-2,:)'|
            Z = |    .             .          .           .     |
                |    .             .            .         .     |
                |    .             .              .       .     |
                | TS(1,:)'      TS(2,:)'       ...    TS(T-p,:)'|

            is a matrix of size (k*p+1,T-p) that is directly built from the input TS.

            E = [e(p+1) e(p+2) ... e(T)] is a matrix of size (k,T-p)
            containing the residuals of the multivariate AR model.


     Output

     'Y'             Matrix variables directly built from TS (see Eq. (2))
     'Z'             Matrix variables directly built from TS (see Eq. (2))
     'B'             Matrix containing AR model parameters
     'E'             Residuals of the AR model

     Note that the most important output is B which contains the parameters of
     the AR model. E is also of interest when autoregressive randomization is
     to be performed. Y and Z are provided for information and testing
     purposes.

     Documenttion written by Raphael Liegeois and CBIG under MIT license:   https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

     Code adapted in python by Ilaria Ricchi.

    """

    if isinstance(TS, list):
        x = np.array([]).reshape(TS[0].shape[0], 0)
        y = np.array([]).reshape(TS[0].shape[0], 0)

        for i, matrix in enumerate(TS):

            matrix = np.asarray(matrix)

            # normalize matrices
            matrix, _, std = normalize_ts(matrix,direction,demean,HP,LP,TR)

            if (std==0).all():
                print(f'Skipping subject {i:.0f}, empty time series')

            # concatenate matrices
            x_temp = matrix[:, 1:]
            y_temp = matrix[:, :-1]
            x = np.concatenate((x, x_temp), axis=1)
            y = np.concatenate((y, y_temp), axis=1)

        Y = x
        Z = y
        
    else:

        TS, _, _ = normalize_ts(TS,direction,demean,HP,LP,TR)
        k, T = TS.shapeRe

        # Split data 
        #     Y : Array-like
        #         Time-series data from t:1->T
        #     Z : Array-like
        #         Time-series data from t:0->T-1
    
        Y =  TS[:, p:]
        Z =  TS[:, :T-p]
    
    assert Y.shape == Z.shape
    
    B = (Y @ Z.T) @ (np.linalg.inv(Z @ Z.T))
    
    # Computing residuals
    E = Y - B@Z
    
    return Y,Z,B,E
    

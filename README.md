# Deep Learning with Wide-Field and Pupil Data

This repository contains a deep learning pipeline that predicts pupil state from wide-field calcium imaging data.

## Wide-Field Analysis

### [#01_registration](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2301_registration)
Register wide-field images to the Allen Brain Atlas using Affine transformation.

### [#02_hemodynamic_correction](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2302_hemodynamic_correction)
Apply hemodynamic correction using the following steps:
1. Apply singular value decomposition (SVD) to 470-nm excitation images to obtain spatial components
2. Extract corresponding temporal components from 405-nm excitation images using the spatial components
3. Perform linear regression between the two temporal components
4. Subtract the estimated calcium-independent 470-nm component from the original 470-nm temporal component
5. The resulting residual represents the calcium-dependent, hemodynamics-corrected temporal component

### [#03_ica](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2303_ica)
Apply SVD to hemodynamic-corrected images, followed by independent component analysis (ICA) to the SVD temporal components.

## Pupil Analysis

### [#04_pupil_diameter](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2304_pupil_diameter)
Process pupil videos and extract diameter measurements:
1. Trim MP4 files to focus on the eye area for DeepLabCut (DLC) analysis
2. Calculate pupil diameter from DLC output CSV files

## Deep Learning

### [#05_deep_learning](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2305_deep_learning)
Train and evaluate the prediction model:
1. Create datasets combining ICA temporal components and pupil diameter time courses
2. Train a recurrent neural network (RNN) model
3. Calculate feature importance using permutation importance and DeepSHAP analysis
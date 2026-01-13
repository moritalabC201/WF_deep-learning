# Deep learning with wide-field and pupil data
Our deep learning model predicts pupil state from wide-field calcium imaging data.

## Wide-field analysis

### [#01_registration](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2301_registration)
we register wide-field images to Allen brain atlas with Affine transformation.

### [#02_hemodynamic_correction](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2302_hemodynamic_correction)
We apply singular value decomposition (SVD) to the 470-nm excitation images to obtain spatial components, then use these spatial components to extract the corresponding temporal components from the 405-nm excitation images. Next, we perform linear regression between the two temporal components and subtract the estimated calcium-independent 470-nm component from the original 470-nm temporal component. The resulting residual is treated as the calcium-dependent, hemodynamics-corrected temporal component.

### [#03_ica](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2303_ica)
We apply SVD to hemodynamic-corrected image and independent component analysis (ICA) to SVD temporal components.


## Pupil analysis

### [#04_pupil_diameter](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2304_pupil_diameter)
We trim mp4 files around eye area for DeepLabCut (DLC) analysis, and calclate diameter from csv file DLC output.


## Deep learning

### [#05_deep_learning](https://github.com/moritalabC201/WF_deep-learning/tree/main/%2305_deep_learning)
We make datasets with time courses from ICA temporal components and pupil diameter, then input it into recurrent neural network (RNN) model. We calculate feature importance by applying permutation importance and DeepSHAP to RNN model.
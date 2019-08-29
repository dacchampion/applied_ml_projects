# Two ML projects
 
1. Pover-T test was a project hosted by Driven Data with the aim to classify households as pover or not. The best model achieved was using Gradient Boosting with hyper parameter optimization by Grid Search and Cross Validation.
 __Source Code: Poverty_T.ipynb__
2. Nuclei Segmentation, this was a project hosted by Kaggle where different strategies were tried
 * U-Net with tensorflow, achieved average results leadfing to the discovery of the heterogeneous image clusters.
 __Source Code: NucleiSegmentation_tf_unet.ipynb__
 * U-Net with keras + preprocessing + image augmentation + morphological operations, improved performance but still far from the top leaderboards
 __Source Code: NucleiSegmentation.ipynb__
 * Yolo detections, project to implement Yolo model to detect nucleis then some post process steps to get the final segmentation result, due to time constraints that step was not implemented but Yolo was succesfully implemented.
 __Source Code: Inside yolo_nuclei_detection folder__

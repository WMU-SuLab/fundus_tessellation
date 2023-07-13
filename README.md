# fundus_tessellation
This project uses color fundus photography to determine the severity of fundus tessellation, mainly using convnext as the backbone, on which improvements are made to achieve the final result.

**./code is the core code involved.**
- code
  - fund_detect: model of macular centroid localization
  - swtf_tf_sgm: model of fundus tessellation segmentation (Aided in judging grade 0 and grade 1)
  - model.py: convnext
  ******
  - five_class_indices.json:label mapping multiclassification
  - five_class_train.py:training procession of multiclassification
  - five_class_utils.py:some functions of multiclassification
  - five_class_predict.py:reasoning procession of multiclassification
  ******
  - roi_class_indices.json:label mapping binaryclassification
  - roi_class_train.py:training procession of binaryclassification
  - roi_class_utils.py:some functions of binaryclassification
  - roi_class_predict.py:reasoning procession of binaryclassification
  ******
  - Roi.py:reasoning procession just use grade1 to grade4 roi_model
  - RoiCoarseModel.py:reasoning procession use grade1 to grade4 roi_model & a coarse model of fundus tessellation/no fundus tessellation
  - RoiCoarseModelSeg.py:reasoning procession not only use grade1 to grade4 roi_model and a coarse model, but also a segmentation to aid in judge grade0 and grade1


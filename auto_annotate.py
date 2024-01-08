# Description: This file is used to test the auto_annotate function in the annotator.py file.
from ultralytics.data.annotator import auto_annotate

datapath = "gear_classfication/images"
output_dirs = "gear_classfication/labels_mask"
model_path = "model/four_object_dection_model/weights/best.pt"

auto_annotate(data=datapath, det_model=model_path, sam_model='sam_b.pt',output_dir=output_dirs) 
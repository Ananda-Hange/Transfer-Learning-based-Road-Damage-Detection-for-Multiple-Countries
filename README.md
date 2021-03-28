# Transfer-Learning-based-Road-Damage-Detection-for-Multiple-Countries

## 1. Overview:

Creating accurate Machine Learning Models which are capable of recognizing and localizing multiple objects in a single image remained a core challenge in computer vision. But, with recent advancements in Deep Learning, Object Detection applications are easier to develop than ever before.
Many municipalities and road authorities seek to implement automated evaluation of road damage. However, they often lack technology, know-how, and funds to afford state-of-the art equipment for data collection and analysis of road damages. Although some countries, like Japan, have developed less expensive and readily available Smartphone-based methods for automatic road condition monitoring, other countries still struggle to find efficient solutions.
This work makes the following contributions in this context. Firstly, Analyze a large-scale heterogeneous road damage dataset comprising 26620 images collected from multiple countries using smartphones. Secondly, Preprocessing on labeled images. Thirdly, we do transfer learning with state of the art object detection models (3 models).Lastly, build end to end pipeline (deployment) with the best model.

## 2. Task: 
1. Road Damage Detection 
2. Fine tune pretrained models to our custom dataset.

## 3. Data:
  The proposed dataset builds upon the recently introduced RDD-2019 dataset.
  
  The dataset is publicly available on the website (https://github.com/sekilab/RoadDamageDetector)

### Dataset Contains:

1. Images  which contains images from all three countries namely Czech, India and Japan
2. XML files which contain bounding box dimension with label in PASCAL format 
3. Label file which contains labels.

## Data Preperation :
    1. Data preperation stage basically have two stages firstly , Resize image according to model we are going to use and secondly, Augmentation step.
    2. Every model have their own way to prepare dataset Like Faster R-CNN and SSD MobileNet has siimmilar data preperation steps and it shown in Data_prep_for_SSD_and_faster_RCNN.ipynb file
    3. Simmilarly for YOLOv3 model all data prepesration steps are shoen in data_prep_with_augmrntation_YoloV3.ipynb file.
  
## Training Models:
    training models explain step by step in training_inference_Faster_RCNN_inception_resenet_640x640.ipynb for Faster RCNN ,For SSD MobilNet Training_inference_SSD_mobilenet_60k_steps.ipynb and YoloV3_training_inference.ipynb
    
## Final pipeline:
    After training all models mention above we comapere them and will choose best model for pipeline,which is shown in final_pieline.ipynb file.
    
## Deployment:
    Final model deployment is explain step by step in Deploy_model.ipynb  file.
 
 







# Target_Classify-and-Detect
## Require：
- python 3.x  
- numpy  
- pytorch  
- opencv-python  
- scikit-image  

## Train:
- Unzip the tiny_vid/tiny_vid.zip in the tiny_vid directory  
- bash trainModel.sh  or ./trainModel.sh  
- training log will be stored in ./LOG dir  
- the best torch model will be stored in ./storedModels dir


## Predict
- python predict.py  
- the predicted label and bbox will be stored in ./predictFile/classname_predict.txt  
- each line is formatted as 'label x_min y_min x_max y_max'  
- the images with plotted bbox will be store in ./predictFile/classname_predict dir  

## Plot training curve
- Using 'bash dataProcess.sh XXX.log' command to process the log file  
- python plot.py




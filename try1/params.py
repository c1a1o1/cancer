'''
Created on Nov 16, 2016

@author: botpi
'''

def param(patient):
    if patient==1:
        return { "cv1_size": 5
               , "cv2_size": 5
               , "cv1_channels": 4
               , "cv2_channels": 8
               , "hidden": 4
               , "img_resize": 16
               , "learning_rate": 0.0005
               , "dropout": 0.5
              }
    elif patient==2:
        return { "cv1_size": 5
               , "cv2_size": 5
               , "cv1_channels": 16
               , "cv2_channels": 8
               , "hidden": 25
               , "img_resize": 16
               , "learning_rate": 0.01
               , "dropout": 0.5
              }
    elif patient==3:
        return { "cv1_size": 5
               , "cv2_size": 5
               , "cv1_channels": 4
               , "cv2_channels": 4
               , "hidden": 4
               , "img_resize": 16
               , "learning_rate": 0.005
               , "dropout": 0.5
              }
    else:
        return {}
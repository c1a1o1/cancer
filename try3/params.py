'''
Created on Nov 16, 2016

@author: botpi
'''

def get():
    return { "image_channels": 1
    	   , "cv1_size": 5
           , "cv2_size": 5
           , "cv1_channels": 4
           , "cv2_channels": 8
           , "hidden": 4
           , "img_height": 512
           , "img_width": 100
           , "learning_rate": 0.0005
           , "dropout": 0.5
          }

from Models.Conv2d_Triplet_Model import *

model = SeparateAnchorModel()
model.Model_Init()
model.TripletLossTrain()
model.Model_Close()

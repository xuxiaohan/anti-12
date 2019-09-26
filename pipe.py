from myusercf import pipe_usercf
from preprocessing import pipe_preprocessing
from recall_sample import pipe_recall_sample
from data import pipe_data
from baseline import pipe_baseline
from others import pipe_others
from others666 import pipe_others666
from model import pipe_model

#
# """
#            myusercf(xx-yy,xx-zz,yy-zz,yy-yy,zz-yy,zz-zz)   ->
# data  ->   baseline                                        ->      recall_sample   -> preprocessing  -> model
#            others                                          ->
# """
#
print("pipe_data")
pipe_data()
print("pipe_baseline")
pipe_baseline()
print("pipe_usercf")
pipe_usercf()
print("pipe_others")
pipe_others()
print("pipe_others666")
pipe_others666()
print("pipe_recall_sample")
pipe_recall_sample()
print("pipe_preprocessing")
pipe_preprocessing()
print("pipe_model")
pipe_model()

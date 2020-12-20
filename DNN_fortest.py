from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  
import numpy as np 
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

tf.keras.backend.set_session(sess)

def DNN( model,best_hiddenlayer,best_epoch,best_batchsize,best_unitslist ) :

  with open( "small_size_test_data.txt") as txtfile:
    rows1 = csv.reader(txtfile)
    data1 = []
    label1 = []
    for row in rows1:
        data1.append(row[0:41])
        label1.append(row[41])
  normal_data = data1 
  normal_label = label1
  y_normalOneHot = np_utils.to_categorical(normal_label) 

  with open( "small_size_test-21_data.txt") as txtfile:
    rows2 = csv.reader(txtfile)
    data2 = []
    label2 = []
    for row in rows2:
        data2.append(row[0:41])
        label2.append(row[41])
  test_21_data = data2
  test_21_label = label2
  y_test_21_OneHot = np_utils.to_categorical(test_21_label) 


  scores_1 = model.evaluate(np.array(normal_data), np.array(y_normalOneHot))  
  scores_2 = model.evaluate(np.array(test_21_data),np.array(y_test_21_OneHot))
  normaltest_acc = scores_1[1]
  test_21_acc = scores_2[1]
  fp = open('update_model_predict.txt', "a" )
  fp.write("for test: ")
  fp.write( str(normaltest_acc))
  fp.write("   for test-21: ")
  fp.write( str(test_21_acc))
  fp.write( "," )
  fp.write(str(best_hiddenlayer))
  fp.write( "," )
  fp.write(str(best_epoch))
  fp.write( "," )
  fp.write(str(best_batchsize))
  fp.write(",")
  fp.write(str(best_unitslist))
  fp.write("\n")
  fp.close()
        
  return normaltest_acc,test_21_acc


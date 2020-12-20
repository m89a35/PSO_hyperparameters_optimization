from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  
import numpy as np 
import csv
import sys
import os 
num_of_model = int(sys.argv[1])
import tensorflow as tf 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

tf.keras.backend.set_session(sess)


def DNN( hidenlayer ,unitslist ,epochnum , batchsizenum) :

  with open( "small_size_train_data.txt") as txtfile:
    rows = csv.reader(txtfile)
    data = []
    label = []
    for row in rows:
        data.append(row[0:41])
        label.append(row[41])
  train_data = data 
  train_label = label
  y_TrainOneHot = np_utils.to_categorical(train_label) 
  y_TestOneHot = np_utils.to_categorical(train_label) 


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
# train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2) 
  avg_val_loss = 0.0
  avg_loss = 0.0
  avg_acc = 0.0
  avg_val_acc = 0.0
  avg_test_acc = 0.0
  test_acclist = []
  losslist = []
  val_losslist = []
  acclist = []
  val_acclist = []
  model = Sequential()
  for i in range(1) :

    model.add(Dense(units=41, input_dim=41, kernel_initializer='normal', activation='relu'))
    for j in range( hidenlayer ) :
        model.add(Dense(units = unitslist[j], kernel_initializer='normal', activation = 'relu'))

    # model.add(Dense(units=128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=5, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # print("generation = ", i)
    train_history = model.fit(x=np.array(train_data), y=np.array(y_TrainOneHot), epochs=epochnum , batch_size=batchsizenum, verbose=2) 
    # scores = model.evaluate(test_data, y_TestOneHot)  
    # val_losslist.append(train_history.history['val_loss'][epochnum-1])
    # avg_val_loss = avg_val_loss + train_history.history['val_loss'][epochnum-1]
    losslist.append(train_history.history['loss'][epochnum-1])
    avg_loss = avg_loss + train_history.history['loss'][epochnum-1]
    acclist.append(train_history.history['acc'][epochnum-1])
    avg_acc = avg_acc + train_history.history['acc'][epochnum-1]
    # val_acclist.append(train_history.history['val_acc'][epochnum-1])
    # avg_val_acc = avg_val_acc + train_history.history['val_acc'][epochnum-1]
    # test_acclist.append(scores[1])
    # avg_test_acc = avg_test_acc + scores[1]

  fp = open("every_time_result.txt", "a")
  # fp.write(str(avg_test_acc))
  # fp.write(",")
  # fp.write(str(epochnum))
  # fp.write( "," )
  # fp.write(str(batchsizenum))
  # fp.write(",")
  fp.write(str(avg_loss) )
  fp.write(",")
  fp.write(str(avg_val_loss ))
  fp.write(",")
  fp.write(str(avg_acc) )
  fp.write(",")
  fp.write(str(avg_val_acc) )
  fp.write( "," )
  fp.write(str(hidenlayer))
  fp.write(",")
  fp.write(str(unitslist))
  fp.write("\n")

  fp.close()
  return avg_loss, model


with open("current_parameter.txt","r" ) as  txtfile:
    rows = csv.reader(txtfile)
    unitslist = []
    for row in rows:
        hidenlayer  = int(row[0])
        for i in range(hidenlayer) :
            unitslist.append(int(row[1+i]))
        epochnum = int(row[hidenlayer+1])
        batchsizenum = int(row[hidenlayer+2])

loss,model = DNN(hidenlayer,unitslist,epochnum,batchsizenum)

fp = open("losslist.txt","a")
fp.write(str(loss))
fp.write(",")
fp.close()

model.save('current_model/model'+str(num_of_model)+'.h5')


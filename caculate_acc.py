import DNN_fortest
import sys
import tensorflow as tf
import csv
model_num = sys.argv[1]

with open("current_parameter.txt","r" ) as  txtfile:
    rows = csv.reader(txtfile)
    unitslist = []
    for row in rows:
        hidenlayer  = int(row[0])
        for i in range(hidenlayer) :
            unitslist.append(int(row[1+i]))
        epochnum = int(row[hidenlayer+1])
        batchsizenum = int(row[hidenlayer+2])
model = tf.contrib.keras.models.load_model('current_model/model'+str(model_num)+'.h5')
normal_acc, test21_acc = DNN_fortest.DNN(model,hidenlayer,epochnum,batchsizenum,unitslist)
fp = open("acclist.txt","a")
fp.write(str(normal_acc))
fp.write(",")
fp.write(str(test21_acc))
fp.write("\n")
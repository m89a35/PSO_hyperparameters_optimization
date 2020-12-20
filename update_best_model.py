import tensorflow as tf 
import keras 
import DNN_fortest
import threading
import tensorflow as tf 
import os
import csv 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# tf.keras.backend.set_session(sess)


def caculate(model_num) :
    os.system("python caculate_acc.py " +str(model_num ) )

def caculate_acc(hiddenlayer,unitslist,epoch,batchsize) :
    open("acclist.txt","w").close()
    normal_acc = 0.0
    test21_acc = 0.0
    t_list = []
    t1 = threading.Thread(target=caculate,args = (1,))
    t_list.append(t1)
    t2 = threading.Thread(target=caculate,args = (2,))
    t_list.append(t2)
    t3 = threading.Thread(target=caculate,args = (3,))
    t_list.append(t3)
    t4 = threading.Thread(target=caculate,args = (4,))
    t_list.append(t4)
    t5 = threading.Thread(target=caculate,args = (5,))
    t_list.append(t5)
    for t in t_list:
      t.start()

    for t in t_list:
      t.join()
    with open("acclist.txt","r" ) as txtfile:
        rows = csv.reader(txtfile)
        normal_acc = 0
        test21_acc = 0
        for row in rows:
            normal_acc += float(row[0])
            test21_acc += float(row[1])
            print(normal_acc)
            print(test21_acc)
    normal_acc = normal_acc / 5.0
    test21_acc = test21_acc / 5.0
    fp = open("avg_best_sofar.txt","a")
    fp.write(str(normal_acc))
    fp.write(",")
    fp.write(str(test21_acc))
    fp.write(",")
    fp.write(str(hiddenlayer))
    fp.write(",")
    fp.write(str(unitslist))
    fp.write(",")
    fp.write(str(epoch))
    fp.write(",")
    fp.write(str(batchsize))
    fp.write("\n")
    fp.close()
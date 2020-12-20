from subprocess import call
import csv
import threading
import os
import tensorflow as tf 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# tf.keras.backend.set_session(sess)

def caculate_avg_fitness(hiddenlayer,unitslist,epoch,batchsize) :
    open("losslist.txt","w").close()
    fp = open("current_parameter.txt","w")
    fp.write(str(hiddenlayer))
    fp.write(",")
    for i in range(hiddenlayer) :
        fp.write(str(unitslist[i]))
        fp.write(",")
    fp.write(str(epoch))
    fp.write(",")
    fp.write(str(batchsize))
    fp.close()

    sum_of_fitness = 0.0
    index = 1 
    t_list=[]
    t1 = threading.Thread(target=DNN,args =  (index,))
    t_list.append(t1)
    index += 1
    t2 = threading.Thread(target=DNN,args =  (index,))
    t_list.append(t2)
    index += 1
    t3 = threading.Thread(target=DNN,args =  (index,))
    t_list.append(t3)
    index += 1
    t4 = threading.Thread(target=DNN,args =  (index,))
    t_list.append(t4)    
    index += 1
    t5 = threading.Thread(target=DNN,args =  (index,))
    t_list.append(t5)


    for t in t_list:
      t.start()
    print("ass")

    for t in t_list:
      t.join()

    with open("losslist.txt","r" ) as  txtfile:
        rows = csv.reader(txtfile)
        avg_loss = 0
        unitslist = []
        for row in rows:
            for i in range(5) :
                avg_loss = avg_loss + float(row[i])
    avg_loss = avg_loss / 5.0
    return avg_loss

def DNN(num_of_model):
    os.system("python DNN_file_fortrain.py " +str(num_of_model) )
    

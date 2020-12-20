import sys 
import random
import caculate_avg
import update_best_model
import os
import tensorflow as tf 
from datetime import datetime


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

tf.keras.backend.set_session(sess)

iteration = int(sys.argv[1])
particle_num = int(sys.argv[2])
# info of particle
particle = []
particle_v = []
particle_fitness = []
particle_best_x = []
particle_best_fitness = []
#current_iteration and evaluation
index_of_iteration = 0
evaluation_num = 0 
# search region set
output_dim = 5.0 
neural_upperbound = 200.0
#hidden max and min 
Hmax = 10.0 
Hmin  = 1.0 

#parameter of dnn 
epochnum = 40
batchsizenum = 2000

#c1,c2 set
c1 = 0.25
c2 = 1
Cmax = 0.25
Cmin = 1 

#Vmin Vmax set
Vmin = -5.0
Vmax = 5.0
#Vhidden max and min
VHmin = -2.0
VHmax = 2.0

#Wstart and Wend set
Wstart = 0.9
Wend = 0.4
W = Wstart
W = 0.5 

#info of sol 
best_sol = [] # a list [hidden,30,40 ....]
best_fitness = 999.999


# update the best sol if fitness bigger than current_best.
def update_best(sol,fitness) :
    global best_sol
    global best_fitness
    fp = open("best.txt","a" )
    
    best_sol = sol
    best_fitness = fitness
    fp.write(str(evaluation_num))
    fp.write(",")
    fp.write(str(fitness))
    fp.write(',')
    fp.write(str(best_sol))
    fp.write("\n")
    fp.close()
    hiddenlayer = int(sol[0])
    unitslist = []
    for i in range(hiddenlayer) :
        unitslist.append(int(sol[i+1]))
    update_best_model.caculate_acc(hiddenlayer,unitslist,epochnum,batchsizenum)

# update the particle best if the current particle bigger than the best_particle_sofar.
def update_particle_best(particle_num,sol,fitness) :
    global particle_best_fitness
    global particle_best_x
    particle_best_fitness[particle_num] = fitness
    particle_best_x[particle_num] = sol 
    fp = open("particle_best_update.txt","a" ) 
    fp.write(str(index_of_iteration))
    fp.write(",")
    fp.write(str(particle_num))
    fp.write(",")
    fp.write(str(particle_best_fitness[particle_num]))
    fp.write(",")
    fp.write(str(particle_best_x[particle_num]))
    fp.write("\n")

# caculate the velocity of particle.
def v_caculate(num_of_current_particle) :
    r1 = random.random()
    r2 = random.random()
    V = []
    temp = W * particle_v[num_of_current_particle][0] + c1*r1*(particle_best_x[num_of_current_particle][0] - particle[num_of_current_particle][0]) + c2 * r2 * (best_sol[0]-particle[num_of_current_particle][0])
    if temp < VHmin :
        temp = VHmin
    if temp > VHmax :
        temp = VHmax 
    
    V.append(temp)
    for i in range(1,len(particle[num_of_current_particle])) :
        temp = W * particle_v[num_of_current_particle][i] + c1*r1*(particle_best_x[num_of_current_particle][i] - particle[num_of_current_particle][i]) + c2 * r2 * (best_sol[i]-particle[num_of_current_particle][i])
        if temp < Vmin :
            temp = Vmin 
        if temp > Vmax :
            temp = Vmax  
        V.append(temp)
    # fp = open("qqqqq.txt","a")
    # fp.write(str(V))
    # fp.write("\n")
    fp.close()
    return V

# caculate the position of particle.
def x_caculate(num_of_current_particle) :
    global particle_v
    X = []
    temp = particle[num_of_current_particle][0] + particle_v[num_of_current_particle][0]
    print("QQ",temp)
    if temp > Hmax : 
        temp = Hmax 
        particle_v[num_of_current_particle][0] = 0 
    if temp < Hmin :
        temp = Hmin 
        particle_v[num_of_current_particle][0] = 0 
    temp = round(temp)
    print("WW",temp)
    X.append(temp)
    for i in range(1,len(particle[num_of_current_particle])) :
        temp = particle[num_of_current_particle][i] + particle_v[num_of_current_particle][i]

        if temp > neural_upperbound :
            temp = neural_upperbound
            particle_v[num_of_current_particle][i] = 0 
        if temp < output_dim :
            temp = output_dim
            particle_v[num_of_current_particle][i] = 0 
        temp = round(temp)
        X.append(temp)
    return X


# get random solution
def random_sol() :
    """
    get a random solution in solution space 
    return a list [hiddenlayernum, neuralnum * 10 ] 
    """
    sol = []
    hiddenlayernum = float(random.randint(1,10))
    sol.append(hiddenlayernum)
    for i in range(10) :
        sol.append(float(random.randint(output_dim,neural_upperbound)))
    return sol 

# get random velocity.
def random_v():
    sol = []
    hiddenlayernum = float(random.randint(1,10))
    sol.append(hiddenlayernum)
    for i in range(10) :
        sol.append(float(random.randint(-1*neural_upperbound,neural_upperbound)))
    return sol

# caculate the fitness of position 
def fitness_caculate(sol ) :
    global evaluation_num
    hiddenlayer = int(sol[0])
    unitslist = []
    for i in range(hiddenlayer) :
        unitslist.append(int(sol[i+1]))
    fp = open("sol.txt","a")
    fp.write(str(sol))
    fp.write("\n")
    fp.close()

    fitness = 0.0 
    fitness = caculate_avg.caculate_avg_fitness(hiddenlayer ,unitslist ,epochnum , batchsizenum)
    evaluation_num += 1
    return fitness

# initialize every particle and their position, velocity.
def Initialization() :
    global particle

    global particle_v
    for i in range(particle_num) :
        #random X V 
        temp = random_sol()
        temp[0] = i+1
        particle.append(temp)
        particle_v.append(random_v())
    for i in range(particle_num) :
        #set to personal best
        particle_best_x.append(list(particle[i]))
        particle_best_fitness.append(999.999)

# caculate the fitness and update the best_sol
def evaluation() :
    global particle_fitness
    particle_fitness = []
    for i in range(particle_num) :
        particle_fitness.append(fitness_caculate( particle[i] ))
        if particle_fitness[i] < particle_best_fitness[i] :
            update_particle_best(i,particle[i],particle_fitness[i])
        if particle_fitness[i] < best_fitness :
            update_best(particle[i],particle_fitness[i])


# update the velocity and position. ( can control if we need linear decrease? )
def determination() :
    global particle
    global particle_v
    global c1
    global c2
    global W 
  
    for i in range(particle_num) :
        #update V and X 
        particle_v[i] =  v_caculate(i)
        particle[i] = x_caculate(i)

    #linear decrease
    # W = Wstart - (Wstart-Wend)/iteration*index_of_iteration
    # c1 = Cmax + (Cmin - Cmax) / iteration * index_of_iteration
    # c2 = Cmin + (Cmax-Cmin) / iteration * index_of_iteration



      
          
#main :
start = datetime.now()
Initialization()
end = datetime.now()
fp = open("time.txt","a")
fp.write(str((end-start).seconds))
fp.write("\n")
fp.close()
for i in range( iteration ) :
    start = datetime.now()
    fp = open("iteration.txt","a")
    fp.write(str(i)+":"+str(evaluation_num)+"\n")
    fp.close()
    evaluation()
    determination()
    print(best_fitness)
    index_of_iteration += 1
    end = datetime.now()
    fp = open("time.txt","a")
    fp.write(str((end-start).seconds))
    fp.write("\n")
    fp.close()






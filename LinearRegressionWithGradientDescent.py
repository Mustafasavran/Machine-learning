import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
#Lineer Regression with one variable without vectorization



#to compute error
#hatayı göstermek için
def getError(t0,t1,x,y):
	addition=0.0
	for i in range(len(x)):
		addition=addition+((t0+t1*x[i])-y[i])**2
	return (1/(2*len(x)))*addition
#
def gradientDescent(x,y):
    length=len(x)
    #learnig rate is 0.0001 initial thetas are 0
    #öğrenme katsayısı 0.0001 başlangıç tetalarımız 0
    lrate=0.0001
    teta0=0
    teta1=0
	
    print(teta0)
    for i in range(1000):

        addtheta0,addtheta1=getAdditions(teta0,teta1,x,y)
        print("ERROR: ",str(getError(teta0,teta1,x,y)))
        teta0=teta0-(lrate*addtheta0)/length
        teta1=teta1-((1/length)*0.0001*addtheta1)
        
		
    return teta0,teta1


def getAdditions(teta0,teta1,x,y):
    addition0=0.0
    addition1=0.0
    for j in range(len(x)):
        addition0+=(teta0+teta1*x[j])-y[j]
        addition1=addition0*x[j]
    return addition0,addition1
#plot data points and our line using matplotlib
#matplotlib kullanarak doğrumuzu çizdik.
def plot(t1,t0):
    
    a=np.arange(150)
    plt.scatter(x,y)
    plt.scatter(a,t1*a+t0)
    plt.show()
#to read and get data using pandas
#pandas kullanarak verileri okuduk ve değerleri aldık
def get_attribütes_from_file(filename):# 
    file=pd.read_csv(filename)
    x=file[file.columns.values[0]]
    y=file[file.columns.values[1]]
    return x,y
x,y=get_attribütes_from_file("linear.csv")

teta0,teta1=gradientDescent(x,y)
plot(teta1,teta0)

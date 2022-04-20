
from cmath import e, pi
from turtle import end_fill
import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from sympy import det

def lineCount(filename):
    file = open(filename, "r")
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()
    return line_count


# return distribution 
def get_RMS(col):
    squaredCol = pd.DataFrame(np.array(col**2))
    summedCol = squaredCol.sum()
    r, c = trainDF.shape
    rmsVal = math.sqrt(summedCol/r)
    return rmsVal


def openFile(filename):
    with open(filename, "r") as readfile:
        # Reads the first line to skip the col names to move indicator
        firstRow = readfile.readline()
        # Split the values in the 1st col (TO DO: future cleanup to use this instead of manually typing cols)
        columns = firstRow.split()
        
        # Read the whole file then split and put each value in a dataframe
        data = readfile.read()
        bigData = data.split()
        r = lineCount(filename) - 1
        df = pd.DataFrame(np.array(bigData).reshape(r,9))

        # Assigning names to columns
        df.columns = ['var', 'm00', 'mu02', 'mu11', 'mu20', 'mu03', 'mu12', 'mu21', 'mu30']
        # Changing the datatype of the numbers (String - > float)
        col_Type = {'m00':float, 'mu02':float, 'mu11':float, 'mu20':float, 'mu03':float, 'mu12':float, 'mu21':float, 'mu30':float}
        df = df.astype(col_Type)

    # Return the dataframe/table from the read file
    return df  

# Creates confusion matrix
def create_CM(y_test,pred, classID):
    cm = list(confusion_matrix(y_test, pred))

    class_names = ['a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'x', 'z']
    #creating confusion matrix dataframe
    cmat_df = pd.DataFrame(cm, index=class_names, columns=class_names) 

    #calculating error 1. Adding all numbers except the diagonal of the matrix
    r1 = sum(cmat_df.iloc[0][1:])+sum(cmat_df.iloc[0][:0])
    r2 = sum(cmat_df.iloc[1][2:])+sum(cmat_df.iloc[1][:1])
    r3 = sum(cmat_df.iloc[2][3:])+sum(cmat_df.iloc[2][:2])
    r4 = sum(cmat_df.iloc[3][4:])+sum(cmat_df.iloc[3][:3])
    r5 = sum(cmat_df.iloc[4][5:])+sum(cmat_df.iloc[4][:4])
    r6 = sum(cmat_df.iloc[5][6:])+sum(cmat_df.iloc[5][:5])
    r7 = sum(cmat_df.iloc[6][7:])+sum(cmat_df.iloc[6][:6])
    r8 = sum(cmat_df.iloc[7][8:])+sum(cmat_df.iloc[7][:7])
    r9 = sum(cmat_df.iloc[8][9:])+sum(cmat_df.iloc[8][:8])
    r10 = sum(cmat_df.iloc[9][10:])+sum(cmat_df.iloc[9][:9])
    cmat_df['error1'] = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10]

    #calculating error 2
    c1 = sum(cmat_df['a'][1:])+sum(cmat_df['a'][:0])
    c2 = sum(cmat_df['c'][2:])+sum(cmat_df['c'][:1])
    c3 = sum(cmat_df['e'][3:])+sum(cmat_df['e'][:2])
    c4 = sum(cmat_df['m'][4:])+sum(cmat_df['m'][:3])
    c5 = sum(cmat_df['n'][5:])+sum(cmat_df['n'][:4])
    c6 = sum(cmat_df['o'][6:])+sum(cmat_df['o'][:5])
    c7 = sum(cmat_df['r'][7:])+sum(cmat_df['r'][:6])
    c8 = sum(cmat_df['s'][8:])+sum(cmat_df['s'][:7])
    c9 = sum(cmat_df['x'][9:])+sum(cmat_df['x'][:8])
    c10 = sum(cmat_df['z'][10:])+sum(cmat_df['z'][:9])
    c11 = sum(cmat_df['error1'][11:])+sum(cmat_df['error1'][:10])
    cmat_df.loc[len(cmat_df.index)] = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]

    #changing index names
    index_names = ['a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'x', 'z', 'error2']
    cmat_df.index = index_names

    # Calculating stats
    if classID == 1:
        error = (c11/(len(df))) * 100
    # else:
    #     error = (c11/(len(df2))) * 100
    
    correct = 100 - error 

    # #printing confusion matrix and result
    # if classID == 1:
    #     print("eval1 cm:")
    # else:
    #     print('eval2 cm')

    print(cmat_df)
    print("Classification Results: ", "{:.2f}".format(correct),"% correct", "{:.2f}".format(error), "% error")

# opens the weight files and puts into a dataframe
def openW(fileW):
    with open(fileW, "r") as readfile:
        firstRow = readfile.readline().split()
        c = len(firstRow)
        r = lineCount(fileW)
        data = readfile.read()
        bigData = data.split()
        df = pd.DataFrame(firstRow)
        df = df.append(bigData)
        df = pd.DataFrame(np.array(df).reshape(r,c))
        df = df.astype(float)
    return df 


# Calls the function to open a file and save the dataframe output to a variable
df = openFile('eval1dat.txt')
# df2 = openFile('eval2dat.txt')
trainDF = openFile('traindat.txt')

# # Getting RMS values for each moments
m00_RMS = get_RMS(trainDF['m00'])
mu02_RMS = get_RMS(trainDF['mu02'])
mu11_RMS = get_RMS(trainDF['mu11'])
mu20_RMS = get_RMS(trainDF['mu20'])
mu03_RMS = get_RMS(trainDF['mu03'])
mu12_RMS = get_RMS(trainDF['mu12'])
mu21_RMS = get_RMS(trainDF['mu21'])
mu30_RMS = get_RMS(trainDF['mu30'])

RMS_df = pd.DataFrame(np.array([m00_RMS, mu02_RMS, mu11_RMS, mu20_RMS, mu03_RMS, mu12_RMS, mu21_RMS, mu30_RMS]).reshape(1,8))
RMS_df.columns = ['m00', 'mu02', 'mu11', 'mu20', 'mu03', 'mu12', 'mu21', 'mu30']
# # Normalizing eval1, eval2, and training data
for column in df[['m00', 'mu02', 'mu11', 'mu20', 'mu03', 'mu12', 'mu21', 'mu30']]:
    df[column] = df[column].div(RMS_df.at[0,column])


for column in trainDF[['m00', 'mu02', 'mu11', 'mu20', 'mu03', 'mu12', 'mu21', 'mu30']]:
    
    trainDF[column] = trainDF[column].div(RMS_df.at[0,column])


# Just taking in moment features and ignore var columns. Saves var columns in separate variable
print(trainDF)
X_train = trainDF.drop(columns = ['var']).values

X_train = np.array(X_train)
y_train = trainDF['var']
X_test = df.drop(columns = ['var']).values
y_test = df['var']
print(X_train)

def one_hot(y_train):
    one_hot_Y = np.zeros((y_train.size, 10))
    for i in range(len(y_train)):
        if y_train[i] == 'a':
            one_hot_Y[i,0] = 1
        elif y_train[i] == 'c':
            one_hot_Y[i,1] = 1
        elif y_train[i] == 'e':
            one_hot_Y[i,2] = 1
        elif y_train[i] == 'm':
            one_hot_Y[i,3] = 1
        elif y_train[i] == 'n':
            one_hot_Y[i,4] = 1
        elif y_train[i] == 'o':
            one_hot_Y[i,5] = 1
        elif y_train[i] == 'r':
            one_hot_Y[i,6] = 1
        elif y_train[i] == 's':
            one_hot_Y[i,7] = 1
        elif y_train[i] == 'x':
            one_hot_Y[i,8] = 1
        elif y_train[i] == 'z':
            one_hot_Y[i,9] = 1
    print(type(one_hot_Y))
    one_hot_Y.astype(float)
    return one_hot_Y
    

def derivative_func(net, function):
    if function == "Sigmoid":
        # for i in range(len(net)):
        #         net[i] = 2 * (1/(1+e**(-2*net[i]))) * (1 - (1/(1+e**(-2*net[i]))))
        newNet = 2 * (1/(1+e**(-2*net))) * (1-(1/(1+e**(-2*net))))
    return newNet
        


# Creating a layer
class Layer_Dense:
    def __init__(self, bias, n_neurons, weight_file):
        if weight_file == 1:
            self.weights = openW("wji.txt").T

            # self.weights = openW("wji.txt").T
            
        elif weight_file ==2:
            self.weights = openW("wkj.txt").T
        
        self.biases = np.zeros((1,n_neurons)) * bias
        # print(self.biases)
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# output from activation function
class Activation_func:
    def forward(self, inputs, function):
        if function == "Sigmoid":
            # for i in range(len(inputs)):
            #     inputs[i] = 1/(1+e**(-2*inputs[i]))
            inputs = 1/(1+e**(-2*inputs))
        elif function == "ReLU":
            inputs = np.maximum(0,inputs)
        elif function == "Softmax":
            exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
            inputs = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = inputs

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.sum(sample_losses)
        return data_loss

class Loss_MeanSquareError(Loss):
    def forward(self, y_pred, y_true):
        loss = 0.5 * np.sum((y_pred - y_true)**2, axis = 1) 
        return loss

# delta k to calculate the change in weights
def deltaK_calc(zk,tk,netk,function):
    deriv = derivative_func(netk,function)

    sum = np.subtract(tk,zk)
    return (sum * deriv).T

df = openFile('eval1dat.txt')
print(df.values)

# layer1 = Layer_Dense(0, 4, 1)
# layer2 = Layer_Dense(0, 10, 2)
# activation1 = Activation_func()
# activation2 = Activation_func()

# layer1.forward(X_train)

# activation1.forward(layer1.output,"Sigmoid")

# layer2.forward(activation1.output)
# netk = layer2.output
# print(layer2)
# activation2.forward(layer2.output,"Sigmoid")



# loss_function = Loss_MeanSquareError()
# loss = loss_function.calculate(activation2.output, one_hot(y_train))

# deltaK = deltaK_calc(activation1.output,one_hot(y_train),netk,"Sigmoid")



# one_hot(y_train)


# one_hot_Y = np.zeros((y_train.size, 10))
# print(one_hot_Y)


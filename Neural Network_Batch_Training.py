
from cmath import e, pi
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

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
testDF = openFile('eval1dat.txt')
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
for column in testDF[['m00', 'mu02', 'mu11', 'mu20', 'mu03', 'mu12', 'mu21', 'mu30']]:
    testDF[column] = testDF[column].div(RMS_df.at[0,column])


for column in trainDF[['m00', 'mu02', 'mu11', 'mu20', 'mu03', 'mu12', 'mu21', 'mu30']]:
    
    trainDF[column] = trainDF[column].div(RMS_df.at[0,column])

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
    one_hot_Y.astype(float)
    return one_hot_Y
    

# Just taking in moment features and ignore var columns. Saves var columns in separate variable
x_train = trainDF.drop(columns = ['var']).values
y_train = one_hot(trainDF['var'])

x_test = testDF.drop(columns = ['var']).values
y_test = one_hot(testDF['var'])



# Sigmoid Function
def act(x):
    return 1/(1+np.exp(-x))


x = x_train;
y = y_train
# parameters
input_size = x.shape[1] # original input + bias
hidden_size = 10
output_size = 10
alpha = 0.1    # learning rate

# weights
w1 = np.random.randn(input_size, hidden_size)
w2 = np.random.randn(hidden_size, output_size)

loss = 0;
loss_arr = [];
loss_idx = [];
# training
for iter in range(10000):
    if(iter%100 == 0 and iter != 0):
        alpha *= 0.9;
        loss_arr.append(loss);
        loss_idx.append(iter);
        # print("{} iter loss is {}.4f".format(iter,loss));
    # forward
    z1 = np.dot(x, w1)
    a1 = act(z1)
    z2 = np.dot(a1, w2)
    Y = act(z2)

    # Calcluate loss
    data = (Y-y);
    loss = np.sum(data**2/data.size);

    # back propagation
    delta2 = (Y-y) * (Y * (1-Y))
    delta1 = np.dot(delta2, w2.T) * (a1 * (1-a1))
    w2 -= alpha * np.dot(a1.T, delta2)
    w1 -= alpha * np.dot(x.T, delta1)

# test
z1 = np.dot(x, w1)
a1 = act(z1)
z2 = np.dot(a1, w2)
Y = act(z2)
print ("Ouput after training...")
res = np.round(Y,0);
for index in range(10):
    print(res[index*10,:])


#/////////Test running.......
z1 = np.dot(x_test, w1)
a1 = act(z1)
z2 = np.dot(a1, w2)
Y = act(z2)

res = np.round(Y,1);
predicted = np.argmax(res, axis=1)

actual = np.argmax(y_test,axis=1);

predict_series = pd.Series(predicted,name='predicted')
actual_series = pd.Series(actual, name="actual")

confusion_matrix = pd.crosstab(actual_series, predict_series)
confusion_matrix=confusion_matrix.set_axis(['a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'x', 'z'], axis=0)
confusion_matrix=confusion_matrix.set_axis(['a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'x', 'z'], axis=1)
print ("confusion_matrix on testData...")
print(confusion_matrix)


# Plot loss function
plt.plot(loss_idx, loss_arr,)
plt.title("Mean Square Error ...")
plt.show();

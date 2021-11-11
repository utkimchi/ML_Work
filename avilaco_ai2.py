import pandas as pd, numpy as np, os, matplotlib.pyplot as plt

#Normalize Values Z-Score : ( val - mean ) / std
#df - dataframe
def preProcess(df,cmean,cstd):
    include = ['Age', 'Vehicle_Age_0', 'Vehicle_Age_1', 'Vehicle_Age_2',  'Annual_Premium', 'Vintage']

    col_mean = cmean
    col_std = cstd

    if cmean or cstd:
        for c in df.columns:
            if c in include:
                df[c] = df[c].astype(float)
                df[c] = (df[c] - col_mean[c]) / col_std[c]
    else:
        for c in df.columns:
            if c in include:
                df[c] = df[c].astype(float)
                col_mean[c] = df[c].mean()
                col_std[c] = df[c].std()
                df[c] = (df[c] - col_mean[c]) / col_std[c]
    return(df,col_mean,col_std)

#Gradient Descent for Ridge Logistic Regression
#td = training data ; lr = learning rate ; rp = regularization parameter ; act_pred = actual response

def gd_log_reg(td,lrn_rts,reg_params,act_resp,reg_type):
    num_te = len(td)
    num_cols = len(td.columns)
    weights = {}
    #All cominations of learning rates & regualrization parameters
    for lr in lrn_rts:
        for rp in reg_params:
            name = str(lr) + " " + str(rp)
            print("Checking", name)
            max_iter = 2500
            iter = 0
            w_vect = np.array([0] * (num_cols)).T
            norm_grad = 9999999
            converge_max = 10**-3
            alpha_n_coeff = lr / num_te

            while iter < max_iter and converge_max < norm_grad:
                predy = 1 / (1 + np.exp(-td.dot(w_vect)))
                t_MSE = np.dot(act_resp - predy, td)
                grad = alpha_n_coeff * t_MSE

                w_vect = w_vect + grad
                #L2 Norm
                if reg_type == "L2":
                    for i in range(1,num_cols):
                        w_vect[i] = w_vect[i] - (lr * rp * w_vect[i])
                else:
                    for i in  range(1,num_cols):
                        mval = abs(w_vect[i]) - lr * rp 
                        if mval > 0:
                            w_vect[i] = (w_vect[i] / abs(w_vect[i])) * mval
                        else:
                            w_vect[i] = 0
                iter += 1
                norm_grad = np.linalg.norm(grad)

            weights[name] = w_vect
    return(weights)

def classify(probabilities):
    classifications = []
    for p in probabilities:
        if p >= 0.5:
            classifications.append(1)
        else:
            classifications.append(0)
    return classifications

#Accuracy correct preds / total trials
def accuracy(tc,rc):
    correct = 0
    for i in range(len(tc)):
        if tc[i] == rc[i]:
            correct+=1
    return correct / (len(tc))


def logLoss(df,weights,response,reg_param,reg_type):
    if reg_type == "L2":
        reg = reg_param * sum(weights**2)
    else:
        reg = reg_param * sum(np.abs(weights))
    cf = 1 / len(df)
    predy = 1 / (1 + np.exp(-df.dot(weights)))
    reg = reg_param * sum(weights**2)
    loss = cf * sum((-response * np.log(predy) - (1-response)*np.log(1-predy))) + reg
    clss = classify(predy)
    return loss, clss

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df1 = pd.read_csv('IA2-train.csv')
print("Checking DF")
print(df1.head())
print(df1.columns)
print("Normalizing")
df1,df1_means,df1_stds = preProcess(df1,{},{})

df1_resp = df1['Response']
df1 = df1.drop(columns = ['Response'])

df2 = pd.read_csv('IA2-dev.csv')
#Train using means and stds from 1
df2,_,_ = preProcess(df2,df1_means,df1_stds)
df2_resp = df2['Response']
df2 = df2.drop(columns = ['Response'])

learning_rates = [0.01]
regularization_params = [10**3,10**2,10**1,10**0,10**-1,10**-2,10**-3,10**-4]
#regularization_params = [10**0,10**-1,10**-2]

df1weights = gd_log_reg(df1,learning_rates,regularization_params,df1_resp,"L2")
train_log_losses = {}
val_log_losses = {}
train_acc = {}
val_acc = {}


for key,ws in df1weights.items():
    reg_param = float(key.split(" ")[1])

    train_lossval,train_class = logLoss(df1,ws,df1_resp,reg_param,"L2")
    train_acc = accuracy(train_class,df1_resp)
    train_log_losses[str(reg_param)] = train_acc

    val_lossval,val_class = logLoss(df2,ws,df2_resp,reg_param,"L2")
    val_acc = accuracy(val_class,df2_resp)
    val_log_losses[str(reg_param)] = val_acc



print("Training Losses")
for k,v in train_log_losses.items():
    print(k," ",str(v))
    print("Validation Accuracy")
    print(val_log_losses[k])



#1a) Plotting the training and validation accuracy


# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# x = range(0,1000)
# y = range(0,1)
# ax1.scatter(train_log_losses.keys(), train_log_losses.values(), s=20, c='b', marker="s", label='training')
# ax1.scatter(val_log_losses.keys(), val_log_losses.values(), s=15, c='r', marker="o", label='validation')
# plt.xlabel("Regularization Parameter")
# plt.ylabel("Accuracy")
# plt.legend(loc='lower right')
# plt.show()

#1b) Our best value was 0.1 so we chose 0.01 and 1.0

best_params = [10**0,10**-1,10**-2]

for l in best_params:
    name = str('0.01') + " " + str(l)
    print("Best weights for", str(l))
    bw = {}
    cols = list(df1.columns)
    for i in range(len(df1.columns)):
        bw[cols[i]] = df1weights[name][i]
    a = sorted(bw, key=bw.get)
    
    i = 0
    top5 = []

    for i in range(5):
        print(a[i],":",bw[a[i]])


# #1c) Sparsity

print("Sparsity values")
sps = []
for l in regularization_params:
    name = str('0.01') + " " + str(l)
    sparsity = 0
    for w in df1weights[name]:
        if w <= 0.00001:
            sparsity+=1
    print("Lambda",str(l),":",sparsity)
    sps.append(sparsity)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(regularization_params, sps, s=20, c='b', marker="s", label='training')
plt.xlabel("Regularization Parameter")
plt.ylabel("Sparsity")
plt.legend(loc='lower right')
plt.show()


#2a) Validation And Training Accuracy

df1weights_lasso = gd_log_reg(df1,learning_rates,regularization_params,df1_resp,"L1")

train_log_losses = {}
val_log_losses = {}
train_acc = {}
val_acc = {}


for key,ws in df1weights_lasso.items():
    reg_param = float(key.split(" ")[1])

    train_lossval,train_class = logLoss(df1,ws,df1_resp,reg_param,"L1")
    train_acc = accuracy(train_class,df1_resp)
    train_log_losses[str(reg_param)] = train_acc

    val_lossval,val_class = logLoss(df2,ws,df2_resp,reg_param,"L1")
    val_acc = accuracy(val_class,df2_resp)
    val_log_losses[str(reg_param)] = val_acc



print("Training Losses - L1 Reg")
for k,v in train_log_losses.items():
    print(k," ",str(v))
    print("Validation Accuracy")
    print(val_log_losses[k])

fig = plt.figure()
ax1 = fig.add_subplot(111)
x = range(0,1000)
y = range(0,1)
ax1.scatter(train_log_losses.keys(), train_log_losses.values(), s=20, c='b', marker="s", label='training')
ax1.scatter(val_log_losses.keys(), val_log_losses.values(), s=15, c='r', marker="o", label='validation')
plt.xlabel("Regularization Parameter")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()


#1b) Our best value was 0.1 so we chose 0.01 and 1.0

best_params = [10**0,10**-1,10**-2]

for l in best_params:
    name = str('0.01') + " " + str(l)
    print("Best weights for", str(l))
    bw = {}
    cols = list(df1.columns)
    for i in range(len(df1.columns)):
        bw[cols[i]] = df1weights_lasso[name][i]
    a = sorted(bw, key=bw.get)
    
    i = 0
    top5 = []

    for i in range(5):
        print(a[i],":",bw[a[i]])


#1c) Sparsity

print("Sparsity values")
sps = []
for l in regularization_params:
    name = str('0.01') + " " + str(l)
    sparsity = 0
    for w in df1weights_lasso[name]:
        if w <= 0.00001:
            sparsity+=1
    print("Lambda",str(l),":",sparsity)
    sps.append(sparsity)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(regularization_params, sps, s=20, c='b', marker="s", label='training')
plt.xlabel("Regularization Parameter")
plt.ylabel("Sparsity")
plt.legend(loc='lower right')
plt.show()

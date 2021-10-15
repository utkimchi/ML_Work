from numpy.lib.shape_base import _column_stack_dispatcher
import pandas as pd, numpy as np,os,matplotlib.pyplot as plt


def BatchGradientDescent(learning_rate,training_df,actual_y):
    mse_list = []
    
    l2_normed_loss = 10000
    max_iterations = 5000
    converge_max = 10**-3
    df = training_df
    w_vect = np.array([0] * (len(df.columns))).T
    n_training_examples = len(df)

    #Run until convergence criterion
    i=0
    while i < max_iterations and converge_max < l2_normed_loss:
        #Establish wTx(i)
        pred_y = df.dot(w_vect)
        #Calculate MSE & Loss
        t_MSE = np.dot(pred_y - actual_y, df)
        loss = (2 / n_training_examples) * t_MSE

        #Normalize Gradient to check if lower than convergence criteria
        l2_normed_loss = np.linalg.norm(loss)

        #Update weight vector
        w_vect = w_vect - (learning_rate * loss)

        #Append to learning rate vector
        mse_list.append(l2_normed_loss)
        i+=1

    iter_list = [x for x in range(i)]

    return mse_list, iter_list, w_vect


def PreProcess(df_in,nmean,nstd,norm_type,v_avg):

    df = df_in.copy()
    #Split date into 3 cols
    df[['month','day','year']] = df['date'].str.split('/',expand = True)
    df = df.drop(columns=['date'])

    #Add Dummy Variable, bias term
    df["bias_t"] = 1

    #Calculate renovation age, or just age since built
    df['age_since_renovated'] = np.where(df['yr_renovated'] == 0, df['year'].astype(int)- df['yr_built'], df['year'].astype(int) - df['yr_renovated'])

    #Drop Id?
    df = df.drop(columns=['yr_renovated'])

    #List of columns to ignore

    omit = ['price','waterfront','bias_t','ids']

    #Quartile Calculation before z-score
    if v_avg == "Q":
        for c in df.columns:
            if c not in omit:
                df[c] = df[c].astype(float)
                p_05 = df[c].quantile(0.05) # 5th quantile
                p_95 = df[c].quantile(0.95) # 95th quantile

                df[c].clip(p_05, p_95, inplace=True)

    df_nn = df.copy()
    col_mean = {}
    col_std = {}
    vavg = {}

    if norm_type == "Z":
        if nmean!=0 and nstd!=0:
            for c in df.columns:
                if c not in omit:
                    df[c] = df[c].astype(float)
                    df[c] = (df[c] - nmean[c]) / nstd[c]

        else:
        #Normalize values
            for c in df.columns:
                if c not in omit:
                    df[c] = df[c].astype(float)
                    col_mean[c] = df[c].mean()
                    col_std[c] = df[c].std()
                    df[c] = (df[c] - col_mean[c]) / col_std[c]

    elif norm_type == "Min":
        if nmean!=0 and nstd!=0:
            for c in df.columns:
                if c not in omit:
                    df[c] = df[c].astype(float)
                    df[c] = (df[c] - nmean[c]) / (nstd[c] - nmean[c])
        else:
            for c in df.columns:
                if c not in omit:
                    df[c] = df[c].astype(float)
                    col_mean[c] = df[c].min()
                    col_std[c] = df[c].max()
                    df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

    else:
        if nmean!=0 and nstd!=0 and v_avg!=0:
            for c in df.columns:
                if c not in omit:
                    df[c] = df[c].astype(float)
                    df[c] = (df[c] - v_avg[c]) / (nstd[c] - nmean[c])
        else:
            for c in df.columns:
                if c not in omit:
                    df[c] = df[c].astype(float)
                    col_mean[c] = df[c].min()
                    col_std[c] = df[c].max()
                    vavg[c] = df[c].mean()
                    df[c] = (df[c] - vavg[c]) / (col_std[c] - col_mean[c])
            

    return df,col_mean,col_std,df_nn,vavg



def ValidationMSE(weights,validation_df,price_df):
    N = len(price_df)
    MSE = sum((price_df - validation_df.dot(weights))**2) / N
    return MSE


def runGradDesc(learning_rates,df,prices):
    #MSE,Iterations,Weights
    nmv = {}
    ni = {}
    nw = {}

    for l in learning_rates:
        m,i,w = BatchGradientDescent(l,df,prices)

        #Convergence Values Pre-Determined
        if l in [10**-1,0.15,0.125,10**-2,10**-3,10**-11,10**-12,10**-13,10**-14]:
            nmv[l]= m
            ni[l] = i
            nw[l] = w

        if not np.isnan(m[-1]) and not np.isposinf(m[-1]):
            plt.plot(i,m,label = str(l))

        print(l, " - MSE:", m[-1], "- Iterations: ",i[-1])

    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    #input("Hit button to continue")
        
    return nmv,ni,nw

def calcValMSE(weights,df_val,prices_val):
    val_mse = []
    i=-1
    for w,j in weights.items():
        t_mse = ValidationMSE(j,df_val,prices_val)
        val_mse.append(t_mse)
        print(i," MSE : ", t_mse )
        i-=1
    return val_mse


###################################################################################### MAIN

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('IA1_train.csv')
df_val = pd.read_csv('IA1_dev.csv')

#Training
df,cmean,cstd,df_nn,v = PreProcess(df,0,0,"Z",0)
ids = df['id']
prices = df['price']
df = df.drop(columns = ['price','id'])

#Validation (Used mean and std from training population)
df_val,_,_,df_val_nn,v = PreProcess(df_val,cmean,cstd,"Z",0)
prices_val = df_val['price']
ids = df_val['id']
df_val = df_val.drop(columns = ['price','id'])

#Change for Learning Rates

learning_rates = [10,10**0,10**-1,10**-2,10**-3,10**-4,10**-5,10**-6]
# learning_rates = [10**-1,10**-2,10**-3]
# learning_rates = [10**-1]

norm_mse_vals, norm_iterations, norm_weights = runGradDesc(learning_rates,df,prices)

#1b)
# Calculate the MSE for the validation data
norm_val_mse = calcValMSE(norm_weights,df_val,prices_val)
col_list = df.columns

#1c)
#Printing weights for best converged solution
for i in range(len(norm_weights[10**-1])):
    print(col_list[i],":",norm_weights[10**-1][i])

# input("Hit button to continue")

# ####################################################################### PART 2
print("PART 2")
# # We use these df's: df_nn,df_val

# #Make sure all values are float
for c in df_nn.columns:
    df_nn[c] = df_nn[c].astype(float)
    df_val_nn[c] = df_val_nn[c].astype(float)

#Drop prices from non-normalized training and val sets
prices_nn = df_nn['price']
df_nn = df_nn.drop(columns = ['price','id'])
prices_val_nn = df_val_nn['price']
df_val_nn = df_val_nn.drop(columns = ['price','id'])



learning_rates_test = [10,10**0,10**-1,10**-2,10**-3,10**-4,10**-5,10**-6,10**-10]
learning_rates_nn = [10**-11,10**-12,10**-13,10**-14]

nn_mse_vals, nn_iterations, nn_weights = runGradDesc(learning_rates_test,df_nn,prices_nn)
nn_mse_vals, nn_iterations, nn_weights = runGradDesc(learning_rates_nn,df_nn,prices_nn)


# #2b)
nn_val_mse  = calcValMSE(nn_weights,df_val_nn,prices_val_nn)
col_list = df_nn.columns

#Printing weights for best converged solution
print("BEST WEIGHTS")
for i in range(len(nn_weights[10**-11])):
    print(col_list[i],":",nn_weights[10**-11][i])


df_nn_vals = pd.DataFrame([nn_weights[10**-11]],columns = col_list)

#df_nn_vals.to_csv("weights_df_nn_val10-11.csv")
#input("Hit button to continue")


# ################################################# PART 3

df_nsl15 = df.copy()
df_val_nsl15 = df_val.copy()

df_nsl15 = df_nsl15.drop(columns="sqft_living15")
df_val_nsl15 = df_val_nsl15.drop(columns="sqft_living15")

d_lr = [10**-1]


d_norm_mse_vals, d_norm_iterations, d_norm_weights = runGradDesc(d_lr,df_nsl15,prices)
d_norm_val_mse  = calcValMSE(d_norm_weights,df_val_nsl15,prices_val)

print("droppedMSE:",d_norm_val_mse)

col_list = df_nsl15.columns

#Printing weights for best converged solution
for i in range(len(d_norm_weights[10**-1])):
    print(col_list[i],":",d_norm_weights[10**-1][i])


df_print_vals = pd.DataFrame([d_norm_weights[10**-1]],columns = col_list)
df_print_vals.to_csv("weights_df_nsl15_val10-11.csv")
#input("Hit button to continue")



##############PART 4 - Feature-Engineering
#############1) Testing Normalization Techniques

df = pd.read_csv('IA1_train.csv')
df_val = pd.read_csv('IA1_dev.csv')

df_ids = df['id']
prices = df['price']
df = df.drop(columns = ['price','id'])

prices_val = df_val['price']
df_val = df_val.drop(columns = ['price','id'])

learning_rates = [10**-1,10**-2,10**-3]

#Min Normalization
df_min,cmean,cstd,df_nn,v = PreProcess(df,0,0,"Min",0)
df_min_val,_,_,df_val_nn,v = PreProcess(df_val,cmean,cstd,"Min",v)
min_mse_vals, min_iterations, min_weights = runGradDesc(learning_rates,df_min,prices)
min_val_mse  = calcValMSE(min_weights,df_min_val,prices_val)

#Z-Score Normalization
df_z,cmean,cstd,df_nn,v = PreProcess(df,0,0,"Z",0)
df_z_val,_,_,df_val_nn,v = PreProcess(df_val,cmean,cstd,"Z",v)
z_mse_vals, z_iterations, z_weights = runGradDesc(learning_rates,df_z,prices)
z_val_mse  = calcValMSE(z_weights,df_z_val,prices_val)

#Mean
df_avg,cmean,cstd,df_nn,v = PreProcess(df,0,0,"Mean",0)
df_avg_val,_,_,df_val_nn,v = PreProcess(df_val,cmean,cstd,"Mean",v)
avg_mse_vals, avg_iterations, avg_weights = runGradDesc(learning_rates,df_avg,prices)
avg_val_mse  = calcValMSE(avg_weights,df_avg_val,prices_val)

##############2) Removing Outliers


df = pd.read_csv('IA1_train.csv')
df_val = pd.read_csv('IA1_dev.csv')

df_ids = df['id']
prices_val = df_val['price']
df_val = df_val.drop(columns = ['price','id'])

learning_rates = [10**-1,0.15]

# Z-Score Outliers
df_z,cmean,cstd,df_nn,v = PreProcess(df,0,0,"Z",0)

print(df_z.shape)
for c in df_z.columns:
    if c!= "price":
        df_z.drop(df_z.loc[abs(df_z[c])>3.5].index, inplace=True)

print(df_z.shape)

prices = df_z['price']
df_z = df_z.drop(columns = ['price','id'])

#Remove the outliers "z-score" larger than 3

df_z_val,_,_,df_val_nn,v = PreProcess(df_val,cmean,cstd,"Z",v)
z_mse_vals, z_iterations, z_weights = runGradDesc(learning_rates,df_z,prices)
z_val_mse  = calcValMSE(z_weights,df_z_val,prices_val)



# Quartile 
print(df.shape)

df = df.drop(columns=['sqft_living15'])
df_val = df_val.drop(columns=['sqft_living15'])
df_z,cmean,cstd,df_nn,v = PreProcess(df,0,0,"Z","Q")
print(df.shape)
print(df_z.shape)

prices = df_z['price']
df_z = df_z.drop(columns = ["price",'id'])

df_z_val,_,_,df_val_nn,v = PreProcess(df_val,cmean,cstd,"Z",v)
z_mse_vals, z_iterations, z_weights = runGradDesc(learning_rates,df_z,prices)
z_val_mse  = calcValMSE(z_weights,df_z_val,prices_val)



############################################################### CORRELATION TESTS REMOVING SINGLE VARIABLES

df = pd.read_csv('IA1_train.csv')
df_val = pd.read_csv('IA1_dev.csv')

df_ids = df['id']
prices_val = df_val['price']
df_val = df_val.drop(columns = ['price','id'])

learning_rates = [0.15]

corr = df.corr()
print(corr[corr > 0.7])

#We see that sqft_living 15, sqft_above, grade, and sqft_lot15 all are correlated with other values

corrvals = ['sqft_living15','sqft_above','grade','sqft_lot15']
df_z,cmean,cstd,df_nn,v = PreProcess(df,0,0,"Z","Q")

prices = df_z['price']
df_z = df_z.drop(columns = ['price','id'])

mses = {}
for vals in corrvals:
    tdf = df_z.copy()
    t_val = df_val.copy()
    tdf = tdf.drop(columns=vals)
    t_val = t_val.drop(columns = vals)
    df_tdf_val,_,_,df_val_tdf,v = PreProcess(t_val,cmean,cstd,"Z",v)
    tdf_mse_vals, tdf_iterations, tdf_weights = runGradDesc(learning_rates,tdf,prices)
    tdf_val_mse  = calcValMSE(tdf_weights,df_tdf_val,prices_val)
    mses[vals] = tdf_val_mse

print(mses)



########################### KAGGLE CODE

df = pd.read_csv('PA1_train1.csv')
df_val = pd.read_csv('PA1_test1.csv')
pred_df = pd.DataFrame()

prices = df['price']
df = df.drop(columns=['id','price'])
pred_df['id'] = df_val['id']
df_val = df_val.drop(columns=['id'])


learning_rates = [.15]

df_f,cmean,cstd,df_f_nn,v = PreProcess(df,0,0,"Min","Q")
df_f_val,_,_,df_val_f,v = PreProcess(df_val,cmean,cstd,"Min",v)
df_f_mse_vals, df_f_iterations, df_f_weights = runGradDesc(learning_rates,df_f,prices)

dfans = df_f_val.dot(df_f_weights[0.15])

pred_df["price"] = dfans



pred_df.to_csv("Prediction_avilaco_a1.csv",index = False)

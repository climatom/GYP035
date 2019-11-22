# -*- coding: utf-8 -*-
"""
This file contains the functions used by students in GYP035

Created: 10/11/2019 by T. Matthews
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

def auto_regress(target,predictors,year_predict,days_behind,p_sig,verbose=True):
    
    """
    
    This function takes a Pandas Series (target)
    of weather and fits a multiple regression using 
    the predictors -- observations of other weather 
    variables made days_nehind. 
    
    Before the regression is fitted, the target and
    predictors have the seasonal cycle removed, so 
    they are 'anomalies' relative to the climatology
    
    Only predictors with a p-value <= p_sig are 
    included in the model. 
    
    Inputs:
    
        - target       : Pandas Series - variable 
                         we want to predict
        - predictors   : Pandas DataFrame - observations 
                         of all weather variables to 
                         possibly include in the (auto)
                         regression
        - year_predict : Int - the year we want to evaluate 
                         model perfoemance on. This year
                         will not be used to fit the 
                         climatology, nor to fit the
                         regression
        - days_behind  : Int - the 'lag' in the auto-
                         regression. We will use observations
                         made this many days ago to try to 
                         predict today's weather
        - p_sig        : Float - regression terms must have a
                         p-value no greater than this to be 
                         included in the regression
        - verbose      : Bool - if True function will print out
                         which variables have been included in 
                         the prediction (along with their 
                         p-values)
                        
                        
    Outputs:
    
        - predicted    : Pandas DataFrame - the modelled variable
                         for year_predict. Obs and clim also returned 
                         for convenience
                         
        - r_squared    : The square of the correlation between the 
                         observed and predicted temperature in year_predict
                         
    Notes: 
    
        - No checking of input is performed
        
        - target and predictors must be the same length
        
        - Created by T.Matthews, 09/11/2019
    """  
    
    # Define target/sim vars
    train_idx = target.index.year != year_predict
    sim_idx = target.index.year == year_predict
    train_target = target.loc[train_idx]
    train_predictors = predictors.loc[train_idx]
    sim_predictors = predictors.loc[sim_idx]
    
    # Fit climatology for the target -- using all but year_predict. Note 
    # that we return the climatology for the full period
    target_clim = fit_clim(train_target.index.dayofyear,365.25,train_target.values[:],\
                        target.index.dayofyear.values[:])
    
    # Define Y as the weather we want to try to predict
    Y = target.values[:][train_idx]-target_clim[train_idx]
    # Shift by days_beind
    Y = Y[days_behind:]
    
    # target_clim_sim is the climatology during the simulation period (less the days_behind)
    target_clim_sim = target_clim[sim_idx][days_behind:]
    
    # Now iterate over the items in predictors, remove seasonal cycles and
    # set up arrays to fit the regressions
    vs = predictors.columns
    X_fit = np.zeros((len(train_predictors)-days_behind,len(vs)))
    X_sim = np.zeros((len(sim_predictors)-days_behind,len(vs)))
    # Init vars
    col=0
    clim_sim = np.zeros((len(sim_predictors),len(vs)))
    for v in vs:
        v_clim = fit_clim(train_predictors.index.dayofyear,365.25,\
                          train_predictors[v].values[:],\
                          predictors.index.dayofyear.values[:])
        v_clim_fit = v_clim[train_idx]
        clim_sim[:,col]=v_clim[sim_idx]
        # Difference with clim and take all up to days_behind to line up with Y
        X_fit[:,col] = train_predictors[v].values[:-days_behind]-v_clim_fit[:-days_behind]
           
        # Increment the column counter
        col+=1
        
    # Include the constant term in the X_array
    X_fit = sm.add_constant(X_fit)
    
    # Filter out NaNs
    valid_idx = np.logical_and(~np.isnan(X_fit).any(axis=1),~np.isnan(Y))
    
    # Fit the model 
    mod = sm.OLS(Y[valid_idx],X_fit[valid_idx]).fit()
     
    # Figure out which vars should be included in the final model (using p_sig)
    col=1
    in_model=0
    X_sim=np.zeros((len(sim_predictors)-days_behind,len(vs)))
    final_model=""
    for v in vs:
        if verbose:
            if col==1: 
                print("-----------------------------------------------")
                print("Fitted model with all terms. Diagnostics follow")
                print("-----------------------------------------------")
                print("P-value for constant (intercept) = %.4f"%mod.pvalues[0])     
                if mod.pvalues[0]<=p_sig: print("...Constant included in model")
            print("P-value for variable: %s = %.3f"%(v,mod.pvalues[col]))
            
        if mod.pvalues[col] <= p_sig:
            X_sim[:,col-1] = sim_predictors[v].values[days_behind:]-\
            clim_sim[days_behind:,col-1]
            if verbose:
                final_model+="%.3f x %s(t-%.0f) + "%(mod.params[col],v,\
                                       days_behind)
                print("(...Including this variable in model...)")             
            in_model+=1
        col+=1
          
    print("-----------------------------------------------")              
     
    # Truncate X_sim
    X_sim=X_sim[:,:in_model]
        
    # Append constant if p-value small enough
    if mod.pvalues[0]<=p_sig:
        X_sim=np.column_stack((np.ones(len(X_sim)),X_sim))
        final_model = "%.3f + "%mod.params[0] + final_model
    Y_sim = np.dot(X_sim,mod.params[mod.pvalues<=p_sig])
    pred = pd.DataFrame(data={"sim":target_clim_sim + Y_sim,\
                              "obs":target.loc[sim_idx].values[days_behind:],\
                              "clim":target_clim_sim},\
                               index=target.index[sim_idx][days_behind:])
    r=(pred["sim"].corr(pred["obs"]))**2
    
    final_model=final_model.strip("+ ")
            
    print("\n\n*      *      *     *      *      *")    
    print("\nModelling complete!")
    print("Final model:\n\t%s(t) = %s\n"%(target.name,final_model))
    print("*      *      *     *      *      *")    
    
    return pred, r


def random_forest(target,predictors,year_predict,days_behind,min_samples,\
                  ntrees,verbose=True):
    
    """
    
    This function takes a Pandas Series (target)
    of weather and fits a random forest (an ensemble
    machine learner)
    
    Before the random forest is fitted, the climatology
    (estimated with a sinusoid) is subtracted -- leaving
    only the "anomalies" (weather). 
    
    Inputs:
    
        - target       : Pandas Series - variable 
                         we want to predict
        - predictors   : Pandas DataFrame - observations 
                         of all weather variables to 
                         possibly include in the (auto)
                         regression
        - year_predict : Int - the year we want to evaluate 
                         model perfoemance on. This year
                         will not be used to fit the 
                         climatology, nor to fit the
                         regression
        - days_behind  : Int - the 'lag' in the auto-
                         regression. We will use observations
                         made this many days ago to try to 
                         predict today's weather                         
        - min_samples  : Int - minimum sample size to allow a node 
                         to *split* (e.g. min_samples = 2 permits min 
                         of 1 sample per leaf)
        - ntrees       : Int - the number of trees in the
                         random forest
        - verbose      : Bool - if True, calculate and return the MAE 
                         (computed on year_predict)
                        
                        
    Outputs:
    
        - out          : Pandas DataFrame - the modelled variable
                         for year_predict. Obs and clim also returned 
                         for convenience
                         
                         
    Notes: 
    
        - Minimal checking of input is performed
        
        - target and predictors must be the same length
        
        - Created by T.Matthews, 18/11/2019
    """  
    
    # Check that the dates match up
    assert (predictors.index == target.index).all(),\
    "Predictors must have same date-time as the target variable!"
            
    # Compute clims on all but year_test; return clims for full period;
    # return anoms for the full period
    vs=predictors.columns
    train_idx=target.index.year!=year_predict
    test_idx=target.index.year==year_predict
    clims={}
    anoms={}
    for v in vs:
        clims[v]=fit_clim(predictors.index[train_idx].dayofyear,\
                                   365.25,predictors[v].loc[train_idx],\
                                   target.index.dayofyear)
            
        anoms[v]=predictors[v]-clims[v]
        
    # Put in dictionaries
    clims=pd.DataFrame(data=clims,index=predictors.index)
    anoms=pd.DataFrame(data=anoms,index=predictors.index)
    
    # Now pull out X_train data (predictors[train_idx][:-days_behind])
    # and the X_test data (predictors[test_idx][:-days_behind])
    X_train=np.array(anoms.loc[train_idx][:-days_behind])
    X_test=np.array(anoms.loc[test_idx][:-days_behind])
    # Y train
    Y_train=anoms[target.name].loc[train_idx][days_behind:]
        
    # Figure out valid entries 
    valid=np.logical_and((~np.isnan(X_train)).all(axis=1),\
                             ~np.isnan(Y_train))
  
    # Pull out 2019 obs and clim -- for convenience
    obs=target.loc[test_idx][days_behind:]
    clim=clims[target.name][test_idx][days_behind:] # note: clim vs. climS
    index=clims.index[test_idx][days_behind:]
        
    # Init the random forest regressor
    rf = RandomForestRegressor(n_estimators=ntrees,\
                                   criterion="mae",\
                                   random_state=42,\
                                   min_samples_split=min_samples)
        
    # Train the model on training data
    rf.fit(X_train[valid,:], Y_train[valid])
        
    # Predict -- noting we have some NaN (so update 'valid')
    valid=(~np.isnan(X_test)).all(axis=1)
    sim_scratch=rf.predict(X_test[valid,:]) + clim[valid]
    sim=np.zeros(len(clim))*np.nan
    sim[valid]=sim_scratch
    out=pd.DataFrame(data={"sim":sim,\
                           "obs":obs,\
                           "clim":clim},index=index)
    mae=None
    if verbose:
        mae=np.mean(np.abs(out["sim"]-out["obs"]))
        print("MAE (ntrees=%.0f; min_samples=%.0f) : %.4f" % (\
              ntrees,min_samples,mae))           
    return out,mae
            
    

def fit_clim(day_of_year,n_days,variable,return_index):
    
    """
    Inputs:
    
        - day_of_year  : array-like - 1-366 indicating the
                         day of year for each observation
        - n_day        : float - number of days in year
        - variable     : array-like - the times series whose
                         climatology we are trying to model
        - return_index : array-like of (possibly repeating)
                         day-of-years for which the estimated
                         climatology will be returned
                        
                        
    Outputs:
    
        - clim         : numpy array - climatological values
                         for all items in return_index
                         
    Notes: 
    
        - Minimal checking of input is performed
        
        - Created by T.Matthews, 09/11/2019
    
    """
    
    # Get indices of valid data to fit the model 
    idx = ~np.isnan(variable)
    
    # Convert fitting data to circular scale
    ind = 2*np.pi*(day_of_year/np.float(n_days)) 
    
    # Repeat for the prediction data 
    ind_pred = 2*np.pi*(return_index/np.float(n_days))
    
    # Take cos/sin of circular indices, put in array
    # and add constant
    X = np.column_stack((np.cos(ind),np.sin(ind)))
    X = sm.add_constant(X)
    
    # Repeat for the prediction
    X_pred = np.column_stack((np.cos(ind_pred),np.sin(ind_pred)))
    X_pred = sm.add_constant(X_pred)
    
    # Check X len
    assert len(X)==len(variable), "Length mismatch: X = %.0f long;" \
    % len(X) + " Y = %.0f long" % len(variable)
    
    # Fit the model 
    mod = sm.OLS(variable[idx], X[idx]).fit()
    
    # Compute the clim for the return index
    sim = mod.predict(X_pred)
    
    # Return the sim
    return sim
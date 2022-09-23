from typing import Any,Dict,List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from prefect import task, flow
import mlflow

@task   
def load_data(path: str,unwanted_cols:List)->pd.DataFrame:
    data=pd.read_csv(path)
    data.drop(unwanted_cols,axis=1,inplace=True)
    return data

#def get_classes(target_data:pd.Series)-> List[str]:
    #return list(target_data)

@task
def split_data(input_:pd.DataFrame,output_:pd.Series,test_data_ratio:float)->Dict[str,Any]:
    X_tr,X_te,y_tr,y_te=train_test_split(input_,output_,test_size=test_data_ratio,random_state=0)
    return {'X_TRAIN':X_tr,'Y_TRAIN':y_tr,'X_TEST':X_te,'Y_TEST':y_te}

@task
def get_scaler(data:pd.DataFrame)->Any:
    # Scaling the numerical features
    scaler=StandardScaler()
    scaler.fit(data)
    
    return scaler

@task
def rescale_data(data:pd.DataFrame,scaler:Any)->pd.DataFrame:
    #scaling the numerical features
    #columns names are (annoyingly) loast after Scaling
    #(ie the dataframe is converted to numpy ndarray)
    data_rescaled=pd.DataFrame(scaler.transform(data),
                               columns=data.columns,
                               index=data.index)
    return data_rescaled

@task
def get_label(data:pd.DataFrame)->Any:
    X_train_cat_le = pd.DataFrame(index=data.index)
    
    cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}
    X_train_cat_le['cut'] = data['cut'].apply(lambda x : cut_encoder[x])

    color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}
    X_train_cat_le['color'] = data['color'].apply(lambda x : color_encoder[x])
    
    clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}
    X_train_cat_le['clarity'] = data['clarity'].apply(lambda x : clarity_encoder[x])
    
    return X_train_cat_le

@task
def get_label1(data:pd.DataFrame)->Any:
    X_test_cat_le = pd.DataFrame(index=data.index)
    
    cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}
    X_test_cat_le['cut'] = data['cut'].apply(lambda x : cut_encoder[x])

    color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}
    X_test_cat_le['color'] = data['color'].apply(lambda x : color_encoder[x])
    
    clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}
    X_test_cat_le['clarity'] = data['clarity'].apply(lambda x : clarity_encoder[x])
    
    return X_test_cat_le

@task
def find_best_model(X_train:pd.DataFrame,y_train:pd.Series,estimator:Any,parameters:List)->Any:
    #Enabling automatic MLFLOW logging for Scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)
    
    with mlflow.start_run():
        reg=GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring='neg_mean_absolute_error',
            cv=5,
            return_train_score=True,
            verbose=1
        )
        reg.fit(X_train,y_train)
        
        #Disabling autologging
        mlflow.sklearn.autolog(disable=True)
        
        return reg


#Workflow
@flow
def main(path:str='diamonds.csv',target:str='price',
         unwanted_cols:List[str]=['Unnamed: 0'],test_size:float=0.2):
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diamond price prediction logs")
    
    #Define Parameters
    DATA_PATH=path
    TARGET_COL=target
    UNWANTED_COLS=unwanted_cols
    TEST_DATA_RATIO=test_size
    

    #Load the Data
    dataframe=load_data(path=DATA_PATH,unwanted_cols=UNWANTED_COLS)
    
    #Identify the Target Variable
    target_data=dataframe[TARGET_COL]
    input_data=dataframe.drop([TARGET_COL],axis=1)
    
    #Split the data into Train and Test
    train_test_dict=split_data(input_=input_data,output_=target_data,test_data_ratio=TEST_DATA_RATIO)
    
    #Resscaling Train and Test Data on numerical features
    scaler=get_scaler(train_test_dict['X_TRAIN'].select_dtypes(include=['int64', 'float64']))
    train_test_dict['X_TRAIN_num']=rescale_data(data=train_test_dict['X_TRAIN'].select_dtypes(include=['int64', 'float64']),scaler=scaler)
    train_test_dict['X_TEST_num']=rescale_data(data=train_test_dict['X_TEST'].select_dtypes(include=['int64', 'float64']),scaler=scaler)
       
    #Labeling the train data
    label=get_label(train_test_dict['X_TRAIN'].select_dtypes(include=['object']))
    
    #Concatenating training data
    X_train_transformed = pd.concat([train_test_dict['X_TRAIN_num'], label], axis=1)
    
    #Lable encoding on test data
    label1=get_label1(train_test_dict['X_TEST'].select_dtypes(include=['object']))
    
    #Concatenating test data
    X_test_transformed=pd.concat([train_test_dict['X_TEST_num'], label1], axis=1)
    
    
    #Model training
    ESTIMATOR=KNeighborsRegressor()
    HYPERPARAMETERS = [{'n_neighbors':[i for i in range(1, 51)], 'p':[1, 2]}]
    
    regressor=find_best_model(X_train_transformed,train_test_dict['Y_TRAIN'],ESTIMATOR,HYPERPARAMETERS)
    print(regressor.best_params_)
    print(regressor.score(X_test_transformed,train_test_dict['Y_TEST']))
   
    
#Deploy the main function
from prefect.deployments import Deployment
from prefect.orion.schemas import IntervalSchedule
from datetime import timedelta

deployment=Deployment.build_from_flow(
    
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=1)),
    work_queue_name="ml"
)

deployment.apply()
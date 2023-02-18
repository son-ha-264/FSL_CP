import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn as sk
import json
import argparse
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from tqdm import tqdm



def main():

    #################################INITIALISATION#################################
    ################################################################################
    

    
    HOME = os.environ['HOME'] + '/FSL_CP/'


    #parser for optional program settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--iter', metavar='testargpars', type=int, default=100, 
     help='how many iterations per support set size (8, 16, 32, 64, 96) will be done')

    #saving parameters of each iteration (not recommended)
    parser.add_argument('--savparam', metavar='save all parameters', type=bool, default=False)
    
    #saving the dataframes of each assay from the assay list
    parser.add_argument('--savdf', metavar='save dataframes', type=bool, default=False)

    #saving the metrics
    parser.add_argument('--savmet', metavar='save metrics', type=bool, default=True)

    args = parser.parse_args()


    
    #loading initial dataset 
    df_assay = pd.read_csv(os.path.join(HOME,'FSL_CP/data/output/FINAL_LABEL_DF.csv'))
    df_feat = pd.read_csv(os.path.join(HOME,'FSL_CP/data/output/norm_CP_feature_df.csv'))
    df_map = pd.read_csv(os.path.join(HOME,'FSL_CP/data/output/assay_target_map.csv'))

    df_assay.drop(['INCHIKEY','CPD_SMILES', 'SAMPLE_KEY'],axis=1,inplace=True)
    df_feat.drop(['CPD_SMILES', 'SAMPLE_KEY', 'INCHIKEY'],axis=1,inplace=True)



    #list of asssays
    f = open(os.path.join(HOME,'FSL_CP/data/output/data_split.json'))
    datajs = json.load(f)
    ls_assay = datajs['test']

    #ls_assay = ["688267"]#, "600886", "737826", "737824_1", "737825", "1495405", "737053", "737400","736947", "752347", 
    #"752496", "752509", "752594", "809095", "845173", "845196", "954338", "845206"]

    


    #list of support set sizes
    ls_sss = [8, 16, 32, 64, 96]



    #hyperparameters 
    parameters = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    'max_depth': np.arange(1, 11, 2),
    'n_estimators': np.arange(50, 550, 50),
    'subsample': [0.5, 1]
    }



    #dictionary for metric outputs
    output_auc = {'ASSAY_ID':[],'ASSAY_CHEMBL_ID':[], 8:[],16:[],32:[],64:[],96:[]}
    output_daucpr = {'ASSAY_ID':[],'ASSAY_CHEMBL_ID':[], 8:[],16:[],32:[],64:[],96:[]}



    #total numbers of iterations that are done
    total_iter_range = args.iter

    print('number of iterations per support set size will be: ', total_iter_range)
    



    ################################################################################
    ################################################################################


    ###################################FUNCTIONS####################################
    ################################################################################


    #function to shape the original dataframe to our likings
    def assay_df(assaydf,featuredf,assays,ass_iter,save):

        assaydf['ASSAY'] = assaydf['ASSAY'].astype('string')
        
        temp_df = assaydf[assaydf['ASSAY'] == assays[ass_iter]]
        temp_df = temp_df.set_index(['NUM_ROW_CP_FEATURES'])
        
        df_one_assay=temp_df.join(featuredf, lsuffix='left_', rsuffix='_right')
        df_one_assay.drop(['VIEWS','ASSAY'],axis=1,inplace=True)

        
        if save == True:
            df_one_assay.to_csv(os.path.join(HOME,f'FSL_CP/data/output/{assays[ass_iter]}.csv'),index=False)
        
        return df_one_assay



    #function for saving best parameters as json file
    def param_json(df,iter_ass,iter_sss):
        json_name = 'params_assay_' + ls_assay[iter_ass] + '_sss_' + ls_sss[iter_sss]
        #df.mode().iloc[0,:].to_json(path_or_buf=f'../under_construction/data_output/{json_name}.json')
        df.to_json(os.path.join(HOME,f'FSL_CP/data/output/{json_name}.json'))



    #function for best parameters as csv file
    def param_csv(df,iter_ass,iter_sss):
        csv_name = 'params_assay_' + ls_assay[iter_ass] + '_sss_' + ls_sss[iter_sss]
        df.to_csv(os.path.join(HOME,f'FSL_CP/data/output/{csv_name}.csv'))



    #function for adding mean and std to output dict
    def output_dict(df, dicto, assay, it_ass):
        for i in df:
            m = np.mean(df[i].to_numpy())
            s = np.std(df[i].to_numpy())
            dicto[i].append(f'{m:.2f}' + '+/-' + f'{s:.2f}')

        dicto['ASSAY_ID'].append(assay[it_ass])
        dicto['ASSAY_CHEMBL_ID'].append(df_map[df_map['ASSAY_ID'] == assay[it_ass]].iat[0,1])

        return dicto

    

    #function for average precision score 
    def delta_auprc(true, pred):

        if type(true) != np.array:
            true = np.array(true)
        if type(pred) != np.array:
            pred = np.array(pred)

        auprc = average_precision_score(true, pred)
        baseline = np.sum(true)/len(true)
        
        return(auprc-baseline)

    

    #saving outputs as .csv file
    def save_out(output_dictionary,metric_name):

        if save_auc == True:
            df_parame = pd.DataFrame.from_dict(output_dictionary)
            df_parame.to_csv(os.path.join(HOME,f'FSL_CP/result/result_summary/xgboost_cp_{metric_name}_result_summary.csv'), index = False)

        return None



    ################################################################################
    ################################################################################



    ####################################SETTINGS####################################
    ################################################################################



    #building the model with xgboost
    model = XGBClassifier(objective='binary:logistic')



    #Save parameters
    save_param = args.savparam

    if save_param == True:
        print('all parameters will be saved')
    else:
        print('parameters will not be saved')
    
    

    #Save parameters
    save_df = args.savdf

    if save_df == True:
        print('all dataframes will be saved')
    else:
        print('dataframes will not be saved')



    #Save auc 
    save_auc = args.savmet
    
    if save_auc == True:
        print('auc and delta aucpr will be saved')
    else:
        print('auc and delta aucpr will not be saved')



    ################################################################################
    ################################################################################



    ##################################MAIN PROGRAM##################################
    ################################################################################



    #initial for-loop that starts process for every assay we chose
    for i in tqdm(range(len(ls_assay))):
        #name = ls_assay[i]

             
        
        df_testing = pd.DataFrame()
        df_testing = assay_df(df_assay,df_feat,ls_assay,i,save_df)

        df_final_auc = pd.DataFrame()
        df_final_daupcr = pd.DataFrame()

        #initialisation of the training-set
        X = df_testing.iloc[:,1:]#.to_numpy()

        y = df_testing.iloc[:,0].astype(int).to_numpy()


        #for-loop for every support set size
        for j in tqdm(range(len(ls_sss)), leave=False):


            #reset dataframe of the parameters after each finished  loop
            df_param = pd.DataFrame(columns=parameters)

            #reset the auc dataframe
            df_auc = pd.DataFrame(columns=ls_sss, index=range(total_iter_range)) 

            df_daupcr = pd.DataFrame(columns=ls_sss, index=range(total_iter_range))
                        


            for k in tqdm(range(total_iter_range), leave=False):
                


            #support set
                
                X_rest, X_support, y_rest, y_support = train_test_split(X, y, test_size=ls_sss[j], stratify=y)

                #query set
                X_unused, X_query, y_unused, y_query = train_test_split(X_rest, y_rest, test_size=32, stratify=y_rest)

                

                #for some assays with support set size of 8 it had occured that the distribution of labels is 50/50, in that case rscv doesnt work
                #therefore cv in rscv is changed to 3 (rscv uses stratifiedkfold for classification problems, the occured error refers to n_splits
                # which in our case is equal to cv)
                if ls_sss[j] == 8:
                    num_split=3
                else:
                    num_split=5



                #randomsearchcv
                rscv = RandomizedSearchCV(model, parameters, random_state=7, n_jobs=4,cv=num_split)          #n_iter=10,cv=5 by default

                search = rscv.fit(X_support,y_support)

                params = search.best_params_



                #adding parameters of the k-th run to the parameter dataframe
                if save_param == True:
                    temporary = pd.DataFrame(params,index=[k])
                    df_param = pd.concat([temporary,df_param])         



                #predicting the labels of the query set based on the best parameters from rscv
                y_pred = search.predict(X_query)



                #evaluating the model

                #auc metric
                auc = sk.metrics.roc_auc_score(y_query,y_pred)
                
                #Delta AUPRC
                delta = delta_auprc(y_query,y_pred)
                
                

                #adding auc value to auc datafrae
                df_auc.iat[k,j] = auc
                
                
                #adding delta aupcr value to delta aucpr dataframe
                df_daupcr.iat[k,j] = delta
                


            df_temp = df_auc[ls_sss[j]]
            df_final_auc = pd.concat([df_final_auc,df_temp],axis=1)
            
            df_temp2 = df_daupcr[ls_sss[j]]
            df_final_daupcr = pd.concat([df_final_daupcr,df_temp2],axis=1)
            


            #saving parameters as json file
            if save_param == True:
                param_csv(df_param,i,j)
            

        
        output_dict(df_final_auc, output_auc, ls_assay, i)
        output_dict(df_final_daupcr, output_daucpr, ls_assay, i)      
        
        save_out(output_auc,'auroc')
        save_out(output_daucpr,'daupcr')
    
    
    return None



if __name__ == '__main__':
    main() 

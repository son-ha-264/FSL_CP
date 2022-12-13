import numpy as np
import pandas as pd  
import json
import itertools

import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier





def main():

    ### Ignore unecessary warning
    from pandas.errors import DtypeWarning
    import warnings
    warnings.simplefilter(action='ignore', category=DtypeWarning)

    ### Inits
    support_set_sizes = [16]#[16, 32, 64, 96]
    query_set_size = 32
    num_repeat = 1

    ### Path inits
    cloome_embedding = torch.load('/home/son.ha/FSL_CP/data/embeddings/CLOOME_embeddings_2.pt')
    list_sample_key = pd.read_csv('/home/son.ha/FSL_CP/data/embeddings/list_sample_key_view.csv')
    label_df = pd.read_csv('/home/son.ha/FSL_CP/data/output/FINAL_LABEL_DF.csv')

    with open('/home/son.ha/FSL_CP/data/output/data_split.json') as f:
        data = json.load(f)
    test_split = data['test']

    ### Untangle label csv (each view is a datapoint)
    label_df['ASSAY'] = label_df['ASSAY'].astype(str)
    label_df = label_df[label_df['ASSAY'].isin(test_split)]
    label_df['INDEX'] = range(len(label_df))
    def _create_img_id(sample_key, views, index, assay, label):
        return([str(index)+'/'+sample_key+'-'+i+'/'+str(assay)+'/'+str(label) for i in views.split('_')])
    a = label_df.apply(lambda x: _create_img_id(x['SAMPLE_KEY'], x['VIEWS'], x['INDEX'], x['ASSAY'], x['LABEL']), axis=1)
    result = list(itertools.chain.from_iterable(a))
    untangled_df = pd.DataFrame(data={
        'df1_index':[int(i.split('/')[0]) for i in result],
        'SAMPLE_KEY_VIEW': [i.split('/')[1] for i in result],
        'ASSAY': [i.split('/')[2] for i in result],
        'LABEL': [np.float64(i.split('/')[3]) for i in result],
        })


    ### Create CLOOME df
    df = pd.DataFrame(data=cloome_embedding.numpy())
    df = pd.concat([list_sample_key, df], axis=1)

    ### Final df
    final_df = pd.merge(df,untangled_df, how='right', on='SAMPLE_KEY_VIEW')

    ### Loop through each assay
    for support_set_size in support_set_sizes: 
        temp_l = []
        for assay in test_split:
            for repeat in range(num_repeat):
                chosen_assay_df = final_df[final_df['ASSAY']==assay]
                # Random stratified sample support and query sets
                chosen_assay_df_2, support_set_df, label_not_support, label_support = train_test_split(
                                chosen_assay_df.iloc[:, 1:513], chosen_assay_df['LABEL'], test_size=support_set_size, stratify=chosen_assay_df['LABEL']
                            )
                _unused_2, query_set_df, _unused3, label_query = train_test_split(
                                chosen_assay_df_2, label_not_support, test_size=query_set_size, stratify=label_not_support
                            )
                # Fit logistic regression
                clf = LogisticRegression(random_state=0, C=1e+10, max_iter=3000).fit(support_set_df, label_support) # 1e+10 best
                pred = clf.predict(query_set_df)
                temp_l.append(roc_auc_score(label_query, pred))
                print(roc_auc_score(label_query, pred))
                break
    print(f"Mean AUC pred: {np.mean(temp_l)}")

    return None


if __name__ == '__main__':
    main()
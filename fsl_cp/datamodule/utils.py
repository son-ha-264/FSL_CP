from sklearn.model_selection import train_test_split

def task_sample(sample_method, chosen_assay_df, support_set_size, query_set_size):
    """
    Helper function for splitting a dataset (pandas dataframe) into support and query set.
    """
    assert sample_method in ['stratify', 'random']

    if sample_method == 'stratify':
        chosen_assay_df_2, support_set_df, _unused1, label_support = train_test_split(
            chosen_assay_df, chosen_assay_df['LABEL'], test_size=support_set_size, stratify=chosen_assay_df['LABEL']
        )
        _unused_2, query_set_df, _unused3, label_query = train_test_split(
            chosen_assay_df_2, chosen_assay_df_2['LABEL'], test_size=query_set_size, stratify=chosen_assay_df_2['LABEL']
        )
    elif sample_method == 'random':
        chosen_assay_df_2, support_set_df, _unused1, label_support = train_test_split(
            chosen_assay_df, chosen_assay_df['LABEL'], test_size=support_set_size, stratify=None
        )
        _unused_2, query_set_df, _unused3, label_query = train_test_split(
            chosen_assay_df_2, chosen_assay_df_2['LABEL'], test_size=query_set_size, stratify=None
                        )
    return support_set_df, query_set_df, label_support, label_query
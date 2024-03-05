from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import re 
import random

def shuffle(X, y):
    '''
    shuffle X
    '''
    idx = list(range(len(X)))
    random.shuffle(idx)
    return X.iloc[idx], y[idx]

def split_yr_mt_day(X, key):
    '''
    splits year, month, day to three different columns.
    '''
    df = pd.DataFrame()
    splited_time = X[key].str.split('-')
    df[key+"_year"] = splited_time.str[0].astype(float)
    df[key+"_month"] = splited_time.str[1].astype(float)
    df[key+"_day"] = splited_time.str[2].astype(float)
    return df

def text_parser(x):
    '''
    parses text data into list of words (bag of words). Could be improved by using nltk or spacy.
    '''
    if x is np.nan:
        return {np.nan}
    else:
        x = re.sub(r'[^\w\s]', '', x)
        stripped = x.strip(" {}[]").replace("\"", "").replace("\'", "").split(" ")
        return stripped
    
def text_ohe_parser(attribute, X, X_submission, min_freq):
    '''
    takes in a text column, conver them into list of words using text_parse. Then, we create a one hot encoding feature column if the word is used at least min_freq of times.
    '''
    text_value_counts = X_submission[attribute].apply(text_parser).explode().value_counts()
    common_texts = text_value_counts[text_value_counts > min_freq].index
    ## group the empty elements
    first_empty_key = None 
    empty_group = ["[]", "\'\'", np.nan, "{}", ""]
    for empty in empty_group:
        if empty in common_texts:
            if first_empty_key is None:
                first_empty_key = empty
            else:
                common_texts.remove(empty)
    ohe_text_dfs = []
    for word in common_texts:
        if word is first_empty_key:
            series = X[attribute].apply(lambda x: 1 if x in empty_group else 0)
        else:
            series = X[attribute].apply(lambda x: 1 if x is not np.nan and word in x else 0)
        ohe_text_dfs.append(pd.Series(series, name=f"{attribute}_{word}"))
    ohe_text_df = pd.concat(ohe_text_dfs, axis=1)
    return ohe_text_df
    
def set_parser(x):
    '''
    converts set in string to an actual set object.
    '''
    if x is np.nan:
        return {np.nan}
    else:
        x = re.sub(r'\s', '', x)
        stripped = x.strip(" {}[]").replace("\"", "").replace("\'", "").split(",")#.split("")
        # stripped = x.strip("{}").strip("[]").replace("\"", "").replace(" ", "").split(",")
        # print(stripped[0][0])
        return set(stripped)
    
def OHE4sets_in_str(attribute, X, X_submission, convert_to_str=False):
    '''
    turns pd.Series of set into one hot encoding features where all the elements will be a binary feature column (1-exists, 0-nonexistent).
    '''
    bag_of_words = set()
    bag_of_words.update(*X_submission[attribute].apply(set_parser).values)
    ## sort the set (cannot sort with np.nan)
    if np.nan in bag_of_words:
        bag_of_words.remove(np.nan)
        bag_of_words = sorted(bag_of_words) 
        bag_of_words.append(np.nan)
    else:
        bag_of_words = sorted(bag_of_words) 
    ohe4set_dfs = []
    ## group the empty elements
    first_empty_key = None 
    empty_group = ["[]", "\'\'", np.nan, "{}", ""]
    for empty in empty_group:
        if empty in bag_of_words:
            if first_empty_key is None:
                first_empty_key = empty
            else:
                bag_of_words.remove(empty)
    ## generate binary feature columns
    attribute_col = X[attribute] if not convert_to_str else X[attribute].astype(str)
    for key in bag_of_words:
        if key is first_empty_key:
            series = pd.Series(attribute_col.apply(lambda x: 1 if x in empty_group else 0), name=f'{attribute}_nans/empty')
        else:
            series = pd.Series(attribute_col.apply(lambda x: 1 if key in x else 0), name=f"{attribute}_{key}")
        ohe4set_dfs.append(series)
    ohe4set_df = pd.concat(ohe4set_dfs, axis=1)
    return ohe4set_df

def data_process_pipeline(X, X_submission, min_freq=50):
    '''
    Processes dataframe X with feature engineering pipeline. For text features, we will generate feature columns based on X_submission's text data. To elaborate, the word
    can be a feature column iff the word is used enough times in the X_submission dataframe. 
    '''
    ## Lower case all texts (for better grouping of columns)
    str_cname_sub = X_submission.columns[X_submission.dtypes == object]
    str_cname = X.columns[X.dtypes == object]
    for c in str_cname_sub:
        X_submission[c] = X_submission[c].str.lower()
    for c in str_cname:
        X[c] = X[c].str.lower()

    ## Boolean
    boolean_columns = ["host_is_superhost", 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture', 'require_guest_phone_verification']
    X_boolean = X[boolean_columns].fillna("nans").astype("category")

    ## Encoding Ordinal data
    cancellation_policy = [['strict', 'strict_14_with_grace_period', 'super_strict_30', 'super_strict_60',  'moderate', 'flexible', 'long_term']]
    encoded_cancelation_policy = OrdinalEncoder(categories=cancellation_policy).fit_transform(X[['cancellation_policy']])
    encoded_cancelation_policy = pd.Series(encoded_cancelation_policy.squeeze(), index=X.index, name='cancellation_policy')
    host_response_time = [['within an hour', 'within a few hours', 'within a day', 'a few days or more', np.nan]]
    encoded_host_response_time = OrdinalEncoder(categories=host_response_time).fit_transform(X[['host_response_time']])
    encoded_host_response_time = pd.Series(encoded_host_response_time.squeeze(), index=X.index, name="host_response_time")
    encoded_df = pd.concat([encoded_cancelation_policy, encoded_host_response_time], axis=1)

    ## extracts the numerical data
    float_cname = X.dtypes[X.dtypes == float].keys()
    float_cname = list(float_cname)
    X_extra_peop = X['extra_people'].str[1:].astype("float")
    int_cname = X.dtypes[X.dtypes == int].keys()
    int_cname = list(int_cname)[2:] ## drop id, host_id
    X_host_response_rate = X['host_response_rate'].str[:-1].astype(float)
    X_num = pd.concat([X[int_cname + float_cname], X_extra_peop, X_host_response_rate], axis=1)

    ## encode date features
    date_dfs = []
    keys = ['first_review', 'last_review', 'host_since']
    for key in keys:
        date_dfs.append(split_yr_mt_day(X, key))
    date_df = pd.concat(date_dfs, axis=1)

    ## One hot Encoding features for words that appear in long text columns. Values are binary (i.e. 1 if exists 0 else) 
    text_ohe_dfs = []
    text_attributes = ["name", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules", "host_about",]
    for attribute in text_attributes:
        text_ohe_dfs.append(text_ohe_parser(attribute, X, X_submission, min_freq=min_freq))
    text_ohe_df = pd.concat(text_ohe_dfs, axis=1)

    ## One hot Encoding features for elements that appear in a set. Values are binary (i.e. 1 if exists 0 else) 
    amenities_df = OHE4sets_in_str("amenities", X, X_submission)
    host_verification_df = OHE4sets_in_str("host_verifications", X, X_submission, convert_to_str=True)

    ## Turn object dtype into category dtype for compatibility with XGBoost and Catboost
    categorical_cname = ['state', 'country', 'property_type', 'room_type', 'city', 'neighbourhood_cleansed','host_location', 'host_neighbourhood', 'neighbourhood_group_cleansed', 'market', 'bed_type']
    X_categories = X[categorical_cname].fillna("nans").astype('category') ## fillna in order for Catboost to work? 

    ## Combine all processed data together
    rename = {
        "amenities_translationmissing:en.hosting_amenity_49": "amenities_translationmissing49",
        "amenities_translationmissing:en.hosting_amenity_50": "amenities_translationmissing50",
    }
    amenities_df = amenities_df.rename(columns=rename) ## get rid of special character ":"
    X_proccessed = pd.concat([X_num, X_boolean, encoded_df, date_df, X_categories, amenities_df, host_verification_df, text_ohe_df], axis=1)
    print(f"total number of features generated: {len(X_proccessed.columns)}")
    print(f"Length of processed data: {len(X_proccessed)}")

    return X_proccessed
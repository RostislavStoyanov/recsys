import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from itertools import product

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from models.TweetDataset import TweetDataset

orig_features = [
    'text_tokens',    ###############
    'hashtags',       #Tweet Features
    'tweet_id',       #
    'media',          #
    'links',          #
    'domains',        #
    'tweet_type',     #
    'language',       #
    'timestamp',      ###############
    'engaged_with_user_id',              ###########################
    'engaged_with_user_follower_count',  #Engaged With User Features
    'engaged_with_user_following_count', #
    'engaged_with_user_is_verified',     #
    'engaged_with_user_account_creation', ###########################
    'engaging_user_id',                  #######################
    'engaging_user_follower_count',      #Engaging User Features
    'engaging_user_following_count',     #
    'engaging_user_is_verified',         #
    'engaging_user_account_creation',    #######################
    'engagee_follows_engager',    #################### Engagement Features
    'reply',          #Target Reply
    'retweet',        #Target Retweet    
    'retweet_comment',#Target Retweet with comment
    'like',           #Target Like
                      ####################
]

target_features = orig_features[-4:]
numerical_features = ['engaged_with_user_follower_count', 'engaged_with_user_following_count', 
                      'engaging_user_follower_count', 'engaging_user_following_count', 'url_cnt',
                      'char_cnt', 'hashtag_cnt', 'Photo_cnt', 'Video_cnt', 'GIF_cnt']
categorical_features = ['language', 'engaged_with_user_id', 'engaging_user_id', 'tweet_type']



cat_target_prod = product(categorical_features, target_features)
features = []
for (cat, target) in cat_target_prod:
    features.append(cat+"_"+target+"_TE")

m = 20
MAX_LEN = 100


def load_test_data(data_path):
    return pd.read_csv(data_path+"test.csv")


def calculate_ctr(gt):
  positive = len([x for x in gt if x == 1])
  ctr = positive/float(len(gt))
  return ctr


def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


def calculate_metrics(pred, gt):
    rce = compute_rce(pred, gt)
    average_precision = average_precision_score(gt, pred)
    
    return rce, average_precision


def calculate_fair_metrics(pred,gt):
    rce = {}
    average_precision = {}
    
    for i in range(5):
      group_predictions = [p for p in pred if p[2] == i]
      group_ground_truth = [p for p in pred if p[-1] == i]
      rce[i] = compute_rce(group_predictions, group_ground_truth)
      average_precision[i] = average_precision_score(group_ground_truth, group_predictions)
  
    return (np.mean(rce.values), np.mean(average_precision.values))


def create_dataset(df, tokenizer, numerical_features=numerical_features,
                   features=features, targets=target_features, max_len=MAX_LEN):
    all_features = numerical_features + features
    
    text = df['text'].values.tolist()
    feats = df.loc[:,all_features].values
    target_values = df.loc[:, targets].values
    
    return TweetDataset(text, feats, target_values, tokenizer, max_len)


def get_results_df(pred, gt):
    results = {}
    
    for i, target in enumerate(target_features):
        curr_results = {}
        rce, avg_prec = calculate_metrics(pred[:,i], gt[:, i])

        curr_results['rce'] = rce
        curr_results['avg_prec'] = avg_prec

        results[target] = curr_results
    
    return pd.DataFrame.from_dict(results)

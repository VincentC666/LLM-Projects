from datasets import load_dataset,load_from_disk
import pandas as pd
import os

# Download dataset from Hugging Face
# data = load_dataset('Yelp/yelp_review_full')


#Save it to local data folder
# data.save_to_disk('./data/yelp_review_full')

# Check distribution for dataset
dataset = load_from_disk('data/yelp_review_full')
df_train = dataset['train'].to_pandas()
df_test = dataset['test'].to_pandas()
train_category_counts = df_train['label'].value_counts().sort_index()
test_category_counts = df_test['label'].value_counts().sort_index()
print(f'Train Data Label distribution: {train_category_counts}')
print(f'Test Data Label distribution: {test_category_counts}')


#  Eliminate rows where 'text' string length is greater than 1024
train_filter = df_train[df_train['text'].str.len()<=1024]
test_filter = df_test[df_test['text'].str.len()<=1024]

# Check max length of text in train data after Elimination
max_train = train_filter['text'].astype(str).str.len().max()
max_test = test_filter['text'].astype(str).str.len().max()
print(f'Train data Max text length: {max_train}')
print(f'Test data Max text length: {max_test}')

# Check distribution for train data after Elimination
train_filter_category_counts = train_filter['label'].value_counts().sort_index()
test_filter_category_counts = test_filter['label'].value_counts().sort_index()
print(len(train_filter))
print(train_filter_category_counts/len(train_filter))
print(test_filter_category_counts/len(test_filter))

#Save the dataset as new_train.csv and new_test.csv
output_Path = 'data/yelp_review_clean'
if not os.path.exists(output_Path):
    os.makedirs(output_Path)
train_filter.to_csv('data/yelp_review_clean/train.csv', index=False)
test_filter.to_csv('data/yelp_review_clean/test.csv', index=False)

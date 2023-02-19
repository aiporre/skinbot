import math
import os
import shutil
from random import choice

import pandas as pd

from skinbot.config import read_config
from skinbot.dataset import WoundImages
from skinbot.config import Config

C = Config()


def get_dataset_where(config1, config2):
    # find and create table of samples with fields:
    # 1. filename
    # 2. label
    # 3. dataset A or B
    # 4. original full_path
    # 5. new full_path
    root_A, root_B = config1['DATASET']['root'], config2['DATASET']['root']
    wound_images_A = WoundImages(root_A, fold_iteration=None)
    wound_images_B = WoundImages(root_B, fold_iteration=None)
    # collecting the data
    files_names = wound_images_B.image_fnames + wound_images_A.image_fnames
    sources = len(wound_images_B)*["B"]
    sources.extend(len(wound_images_A)*['A'])
    # label = self.image_fnames[index].split("_")[0]
    labels = [ff.split("_")[0].strip().lower() for ff in wound_images_B.image_fnames] + \
             [ff.split("_")[0].strip().lower() for ff in wound_images_A.image_fnames]
    for i, l in enumerate(labels):
        if l in C.labels.fixed_error_labels:
            labels[i] = C.labels.fixed_error_labels[l]
    original_paths = [os.path.join(wound_images_B.images_dir, f) for f in wound_images_B.image_fnames] \
                     + [os.path.join(wound_images_A.images_dir, f) for f in wound_images_A.image_fnames]
    newpaths = len(original_paths) *['']
    # create dataframe
    data = {
        "filename":  files_names,
        'label': labels,
        'source': sources,
        'original_path': original_paths,
        'new_path':  newpaths
    }
    return pd.DataFrame(data)

def get_images_paths(config1, config2):
    root_A, root_B = config1['DATASET']['root'], config2['DATASET']['root']
    wound_images_A = WoundImages(root_A, fold_iteration=None)
    wound_images_B = WoundImages(root_B, fold_iteration=None)
    return wound_images_A.images_dir, wound_images_B.images_dir

def get_number_a(where_dict):
    return len(where_dict[where_dict['source'] == 'A'])

def get_number_b(where_dict):
    return len(where_dict[ where_dict['source'] == 'B'])

def get_dataset_stats(where_dict):
    def get_class_ratios(df):
        _labels = df['label'].tolist()
        unique_labels = set(_labels)
        ratios = {}
        for l in unique_labels:
            xx = df[df['label'] == l]
            ratios[l] = len(xx) / len(_labels)
        return ratios
    # get the number of samples by label
    def get_class_counts(df):
        _labels = df['label'].tolist()
        unique_labels = set(_labels)
        counts = {}
        for l in unique_labels:
            xx = df[df['label'] == l]
            counts[l] = len(xx)
        return counts
    ratios_all = get_class_ratios(where_dict)
    ratios_a = get_class_ratios(where_dict[where_dict['source'] == 'A'])
    ratios_b = get_class_ratios(where_dict[where_dict['source'] == 'B'])
    labels = list(set(where_dict['label']))
    # get the number of samples in each label for each dataset
    counts_a = get_class_counts(where_dict[where_dict['source'] == 'A'])
    counts_b = get_class_counts(where_dict[where_dict['source'] == 'B'])
    counts_all = get_class_counts(where_dict)
    # get desired counts
    desired_counts_a = {}
    desired_counts_b = {}
    N = sum(counts_all.values())
    Na = sum(counts_a.values())
    Nb = sum(counts_b.values())
    Pa = Na/N
    Pb = Nb/N
    for l in labels:
        desired_counts_a[l] = math.ceil(counts_all[l]*Pa)
        desired_counts_b[l] = math.floor(counts_all[l]*Pb)
        assert desired_counts_a[l] + desired_counts_b[l] == counts_all[l]
    # get the difference between desired and actual counts
    diff_a = {}
    diff_b = {}
    for l in labels:
        diff_a[l] = desired_counts_a[l] - counts_a.get(l,0)
        diff_b[l] = desired_counts_b[l] - counts_b.get(l,0)
    ratios_a_sorted = []
    ratios_b_sorted = []
    counts_a_sorted = []
    counts_b_sorted = []
    desired_counts_a_sorted = []
    desired_counts_b_sorted = []
    diff_a_sorted = []
    diff_b_sorted = []
    ration_all_sorted = []

    for label in labels:
        ratios_a_sorted.append(ratios_a.get(label, 0))
        ratios_b_sorted.append(ratios_b.get(label, 0))
        counts_a_sorted.append(counts_a.get(label, 0))
        counts_b_sorted.append(counts_b.get(label, 0))
        desired_counts_a_sorted.append(desired_counts_a.get(label, 0))
        desired_counts_b_sorted.append(desired_counts_b.get(label, 0))
        diff_a_sorted.append(diff_a.get(label, 0))
        diff_b_sorted.append(diff_b.get(label, 0))
        ration_all_sorted.append(ratios_all.get(label, 0))

    data = {
        'label': labels,
        'ratio_A':ratios_a_sorted,
        'ratio_B': ratios_b_sorted,
        'ratio_all': ration_all_sorted,
        'count_A': counts_a_sorted,
        'count_B': counts_b_sorted,
        'desired_count_A': desired_counts_a_sorted,
        'desired_count_B': desired_counts_b_sorted,
        'diff_A': diff_a_sorted,
        'diff_B': diff_b_sorted
    }
    return pd.DataFrame(data)

def is_balanced(dataset_stats, tolerance=0.9, source='A'):
    balance_ratio_a = min(dataset_stats['ratio_A'])/max(dataset_stats['ratio_A'])
    balance_ratio_b = min(dataset_stats['ratio_B'])/max(dataset_stats['ratio_B'])
    # print('balance ratio A: ', balance_ratio_a)
    # print('balance ratio B: ', balance_ratio_b)
    # print('-----------')
    if source == 'A':
        return balance_ratio_a > tolerance
    else:
        return balance_ratio_b > tolerance

def get_lowest_label_support(where_dict, source):
    dataset_stats = get_dataset_stats(where_dict)
    column_to_read = 'ratio_'+source
    min_ratio = min(dataset_stats[column_to_read])
    candidate_indices = dataset_stats.index[dataset_stats[column_to_read] == min_ratio].tolist()
    label = dataset_stats['label'].iloc[candidate_indices[0]]
    return label

def get_highest_label_support(where_dict, source):
    dataset_stats = get_dataset_stats(where_dict)
    column_to_read = 'ratio_'+source
    min_ratio = max(dataset_stats[column_to_read])
    candidate_indices = dataset_stats.index[dataset_stats[column_to_read] == min_ratio].tolist()
    label = dataset_stats['label'].iloc[candidate_indices[0]]
    return label

def get_desired_counts(where_dict, source, label):
    dataset_stats = get_dataset_stats(where_dict)
    column_to_read = 'desired_count_'+source
    counts = dataset_stats[column_to_read].tolist()
    selected_index = dataset_stats.index[dataset_stats['label'] == label].tolist()[0]
    return counts[selected_index]

def get_desired_diff(dataset_stats, source, label):
    column_to_read = 'diff_'+source
    counts = dataset_stats[column_to_read].tolist()
    selected_index = dataset_stats.index[dataset_stats['label'] == label].tolist()[0]
    return counts[selected_index]

def get_count(where_dict, source, label):
    dataset_stats = get_dataset_stats(where_dict)
    column_to_read = 'count_'+source
    counts = dataset_stats[column_to_read].tolist()
    selected_index = dataset_stats.index[dataset_stats['label'] == label].tolist()[0]
    return counts[selected_index]


def get_random_sample_from_class(where_dict, source, label):
    # get a random sample from a class
    # return the index of the sample
    # return None if no sample is found
    # get the indices of the samples with the label
    where_dict_source = where_dict[where_dict['source'] == source]
    indices = where_dict_source.index[where_dict_source['label'] == label].tolist()
    # sample randomly one element from indices
    if len(indices) > 0:
        return choice(indices)
    else:
        return None

def swap_samples(where_dict, index_A, index_B):
    # where_dict.at[index_A, 'new_path'] = where_dict.at[index_B, 'original_path']
    # where_dict.at[index_B, 'new_path'] = where_dict.at[index_A, 'original_path']
    # where_dict.at[index_A, 'source'] = 'B'
    # where_dict.at[index_B, 'source'] = 'A'
    org_path_a= where_dict.at[index_A, 'original_path']
    org_path_b = where_dict.at[index_B, 'original_path']   
    dir_a, fname_a = os.path.split(org_path_a)
    dir_b, fname_b = os.path.split(org_path_b)
    where_dict = move_sample(where_dict, index_A, 'B', dir_a, dir_b)
    where_dict = move_sample(where_dict, index_B, 'A', dir_b, dir_a)
    return where_dict


def move_sample(where_dict, index, source, org_path, dest_path):
    new_path = where_dict.at[index, 'original_path'].replace(org_path, dest_path)
    image_fname = os.path.basename(where_dict.at[index, 'original_path'])
    # new_path = os.path.join(new_path, image_fname)
    where_dict.at[index, 'new_path'] = new_path
    where_dict.at[index, 'source'] = source
    return where_dict

def move_images(org_path, dest_path):
    if os.path.exists(org_path):
        shutil.move(org_path, dest_path)
        if os.path.exists(dest_path) and not os.path.exists(org_path):
            print('moved file!')
        else:
            print('failed moving file')
    else:
        print('Cannot move: ', org_path, ' Doesn\'t exists.')
        

config1 =  read_config('config.ini')
config2 = read_config('config_external.ini')
# config
C.set_config(config1) 

# create dictionaty
where_dict = get_dataset_where(config1, config2)
# {fname:"", path: "", "where": A or B}
N_a = get_number_a(where_dict)  # get the desired number of samples
N_b = get_number_b(where_dict)  # get the desired number of samples
dataset_stats = get_dataset_stats(where_dict)  # get sample ratio balance
source_a_path, source_b_path = get_images_paths(config1,config2)

print('dataset stats before')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(dataset_stats)
# for l in dataset_stats['label']:
#     diff = get_desired_diff(dataset_stats, 'A', l)
#     print('diff for label ', l, ' is ', diff)
#     if diff > 0:
#         # move sample from B to A
#         for i in range(diff):
#             index = get_random_sample_from_class(where_dict, 'B', l)
#             if index is not None:
#                 where_dict = move_sample(where_dict, index, 'A', source_b_path, source_a_path)
#     else:
#         # move sample from A to B
#         for i in range(abs(diff)):
#             index = get_random_sample_from_class(where_dict, 'A', l)
#             if index is not None:
#                 where_dict = move_sample(where_dict, index, 'B', source_a_path, source_b_path)

while not (is_balanced(dataset_stats, source='B')):
    # move samples around in the where_dict
    # favor lowest support for B
    label_candidate = get_lowest_label_support(where_dict, "B")
    # print('label_cand=', label_candidate)
    sample_candidate_A = get_random_sample_from_class(where_dict, "A", label_candidate)
    label_candidate = get_highest_label_support(where_dict, "B")
    sample_candidate_B = get_random_sample_from_class(where_dict, "B", label_candidate)
    # print('sample-canA = ', sample_candidate_A)
    # print('sample-canB = ', sample_candidate_B)
    where_dict = swap_samples(where_dict, sample_candidate_A, sample_candidate_B)
    dataset_stats = get_dataset_stats(where_dict)
    # repeat the same with A
print(dataset_stats)

print('dataset_stats after: ')
dataset_stats = get_dataset_stats(where_dict)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(dataset_stats)
# sava where_dict into a csv file
where_dict.to_csv('where.csv', index=False)

# apply the changes
# for index, row in where_dict.iterrows():
#     if len(row['new_path']) is not 0:
#         p1 = os.path.abspath(row['original_path'])
#         p2 = os.path.abspath(row['new_path'])
#         # move mask
#         p1_mask = p1.replace('.jpg', '_mask.png').replace('.JPG', '_mask.png')
#         p2_mask = p2.replace('.jpg', '_mask.png').replace('.JPG', '_mask.png')
#         p1_color_mask = p1.replace('.jpg', '_color_mask.png').replace('.JPG', '_color_mask.png')
#         p2_color_mask = p2.replace('.jpg', '_color_mask.png').replace('.JPG', '_color_mask.png')
#         p1_watershed_mask = p1.replace('.jpg', '_watershed_mask.png').replace('.JPG', '_watershed_mask.png')
#         p2_watershed_mask = p2.replace('.jpg', '_watershed_mask.png').replace('.JPG', '_watershed_mask.png')
#         # break
#         #shutil.move(p1, p2)
#         print('move', p1, ' to ', p2)
#         move_images(p1,p2)
#         # shutil.move(p1_mask, p2_mask)
#         print('move', p1_mask, ' to ', p2_mask)
#         move_images(p1_mask, p2_mask)
#         # shutil.move(p1_color_mask, p2_color_mask)
#         print('move', p1_color_mask, ' to ', p2_color_mask)
#         move_images(p1_color_mask, p2_color_mask)
#         # shutil.move(p1_watershed_mask, p2_watershed_mask)
#         print('move', p1_watershed_mask, ' to ', p2_watershed_mask)
#         move_images(p1_watershed_mask, p2_watershed_mask)
# 
# while not (is_balanced(dataset_stats, source='B')):
#     # move samples around in the where_dict
#     # favor lowest support for B
#     label_candidate = get_lowest_label_support(where_dict, "B")
#     sample_candidate_A = get_random_sample_from_class(where_dict, "A", label_candidate)
#     label_candidate = get_highest_label_support(where_dict, "B")
#     sample_candidate_B = get_random_sample_from_class(where_dict, "B", label_candidate)
#     where_dict = swap_samples(where_dict, sample_candidate_A, sample_candidate_B)
#     dataset_stats = get_dataset_stats(where_dict)
#     # repeat the same with A
# print(dataset_stats)
# while not(is_balanced(dataset_stats, source='A')):
#     label_candidate = get_lowest_label_support(where_dict, "A")
#     sample_candidate_B = get_random_sample_from_class(where_dict, "B", label_candidate)
#     label_candidate = get_highest_label_support(where_dict, "A")
#     sample_candidate_A = get_random_sample_from_class(where_dict, "A", label_candidate)
#     where_dict = swap_samples(where_dict, sample_candidate_A, sample_candidate_B)
#     dataset_stats = get_dataset_stats(where_dict)
#     # ... repeat
# print(dataset_stats)

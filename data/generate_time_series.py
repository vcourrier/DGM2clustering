'''
Created on Jun 17, 2020

'''
import os, sys
import numpy as np

import torch

from scipy.stats import iqr
from torch.distributions import uniform
from torch.utils.data import DataLoader

from lib.utils import *
from lib.load_imputed_data_GRUI import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/lib')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

min_time_series_len = 10

data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/" + data_dir

'''remove outlier with IQR method'''


def remove_outliers(dataset, masks):

    dataset_np = dataset.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])

    masks_np = masks.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])

    for i in range(dataset_np.shape[1]):

        masked_dataset = dataset_np[masks_np[:, i] != 0, i].numpy()

        iqr_score = iqr(masked_dataset)*1.5

        lower = np.quantile(masked_dataset, 0.25) - iqr_score

        higher = np.quantile(masked_dataset, 0.75) + iqr_score

        masks_np[(dataset_np[:, i] > higher), i] = 0

        masks_np[(dataset_np[:, i] < lower), i] = 0

        dataset_np[(dataset_np[:, i] > higher), i] = -1000

        dataset_np[(dataset_np[:, i] < lower), i] = -1000

    masks = masks_np.view(dataset.shape[0], dataset.shape[1], dataset.shape[2])

    new_dataset = dataset_np.view(dataset.shape[0], dataset.shape[1], dataset.shape[2])

    return new_dataset, masks


def remove_outliers2(dataset, masks):

    dataset_np = dataset.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])

    masks_np = masks.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])

    for i in range(dataset_np.shape[1]):

        non_masked_ids = (masks_np[:, i] != 0)

        masked_dataset = dataset_np[non_masked_ids, i]

        masked_mean = torch.mean(masked_dataset)

        masked_std = torch.sqrt(torch.mean((masked_dataset - masked_mean)**2))

        lower = masked_mean - 3*masked_std

        higher = masked_mean + 3*masked_std

        masks_np[(dataset_np[:, i] > higher) + (dataset_np[:, i] < lower), i] = 0

    new_masks = masks_np.view(dataset.shape[0], dataset.shape[1], dataset.shape[2])

    return dataset, new_masks


def standardize_dataset(training_set, test_set, mask_train):

    mean = torch.sum((training_set*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), dim=0) \
         / torch.sum(mask_train.view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), dim=0)

    train_mean = mean.expand(training_set.shape[0], training_set.shape[1], mean.shape[0])

    std = torch.sqrt(torch.sum(((training_set*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2])
                                - (train_mean*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]))**2, dim=0)
                                / torch.sum(mask_train.view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), dim=0))

    train_std = std.expand(training_set.shape[0], training_set.shape[1], std.shape[0])

    dims = (std != 0)

    training_set[:, :, dims] = (training_set[:, :, dims] - train_mean[:, :, dims])/train_std[:, :, dims]

    test_mean = mean.expand(test_set.shape[0], test_set.shape[1], mean.shape[0])

    test_std = std.expand(test_set.shape[0], test_set.shape[1], std.shape[0])

    test_set[:, :, dims] = (test_set[:, :, dims] - test_mean[:, :, dims])/test_std[:, :, dims]

    return training_set, test_set


def normalize_dataset(training_set, test_set, mask_train):

    print('normalization start!!')

    x_max = torch.max((training_set*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), axis=0)[0]

    x_min = torch.min((training_set*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), axis=0)[0]

    range = x_max - x_min

    update_data = training_set.clone().view(training_set.shape[0]*training_set.shape[1], training_set.shape[2])

    update_test_data = test_set.clone().view(test_set.shape[0]*test_set.shape[1], test_set.shape[2])

    update_data[:, range != 0] = (update_data[:, range != 0] - x_min[range != 0])/range[range != 0]

    update_test_data[:, range != 0] = (update_test_data[:, range != 0] - x_min[range != 0])/range[range != 0]

    return update_data.view(training_set.shape[0], training_set.shape[1], training_set.shape[2]), update_test_data.view(test_set.shape[0], test_set.shape[1], test_set.shape[2])


def get_features_with_one_value(masks, masks2):

    all_features = torch.sum(masks.view(masks.shape[0]*masks.shape[1], masks.shape[2]), 0)

    all_features2 = torch.sum(masks2.view(masks2.shape[0]*masks2.shape[1], masks2.shape[2]), 0)

    return (all_features > 0)*(all_features2 > 0)


def get_train_mean(data_obj, inference_len):

    train_sum = 0
    count = 0

    for id, data_dict in enumerate(data_obj["train_dataloader"]):

        for k in range(data_dict["observed_lens"].shape[0]):

            len = inference_len
            train_sum += torch.sum(data_dict["observed_data"][k, 0:len]*data_dict['observed_mask'][k, 0:len], dim=[0])

            count += torch.sum(data_dict['observed_mask'][k, 0:len], dim=0)

    train_mean = train_sum/count

    return train_mean


def check_delta_time_stamps(masks, time_stamps, exp_delta_time_stamps):

    all_ids = torch.tensor(list(range(time_stamps.shape[1]-1)))

    delta_time_stamps = torch.zeros_like(time_stamps)

    delta_time_stamps[:, all_ids + 1] = time_stamps[:, all_ids + 1] - time_stamps[:, all_ids]

    time_gap_tensors = torch.zeros(masks.shape[2], dtype=torch.float)

    res_delta_time_stamps = torch.zeros_like(masks, dtype=torch.float)

    for k in range(masks.shape[1]):

        res_delta_time_stamps[0, k] = time_gap_tensors + delta_time_stamps[0, k]

        time_gap_tensors = (1 - masks[0, k])*time_gap_tensors + (1 - masks[0, k])*delta_time_stamps[0, k]

    print('diff::', torch.norm(res_delta_time_stamps[0] - exp_delta_time_stamps[0]))


def check_remove_none(train_y, new_train_y):

    sample_id = 0

    count = 0

    for i in range(train_y[sample_id].shape[0]):
        num_nan = torch.sum(np.isnan(train_y[sample_id][i]))

        if not num_nan == train_y.shape[2]:
            for j in range(train_y[sample_id].shape[1]):
                if torch.isnan(train_y[sample_id, i, j]).item():
                    assert torch.isnan(new_train_y[sample_id, count, j])
                    continue

                else:
                    if torch.isnan(new_train_y[sample_id, count, j]).item():
                        assert torch.isnan(train_y[sample_id, i, j])
                        continue
                    else:
                        assert train_y[sample_id, i, j].item() == new_train_y[sample_id, count, j].item()
            count += 1
        else:
            continue


models_to_remove_none_time_stamps = [GRUD_method, 'DHMM_cluster_tlstm']


def parse_datasets_name(dataset_name, args) :

    def basic_collate_fn(batch, time_steps, args=args, data_type="train"):

        (
            batched_data,
            batched_mask,
            batched_origin_data,
            batched_origin_masks,
            batched_new_random_masks,
            batched_tensor_len,
            batched_time_stamps,
            batched_delta_time_stamps,
            batched_ids
        ) = zip(*batch)

        data_dict = {
            "data": torch.stack(batched_data),
            "lens": torch.tensor(batched_tensor_len),
            "origin_data": torch.stack(batched_origin_data),
            "origin_mask": torch.stack(batched_origin_masks),
            "time_stamps": torch.stack(batched_time_stamps),
            "delta_time_stamps": torch.stack(batched_delta_time_stamps),
            "new_random_mask": torch.stack(batched_new_random_masks),
            "time_steps": time_steps,
            "ids": torch.tensor(batched_ids),
            "mask": torch.stack(batched_mask)
            }

        data_dict = split_and_subsample_batch(data_dict, args, data_type=data_type)
        return data_dict

    dataset_name = args.dataset

    n_total_tp = args.timepoints + args.extrap

    max_t_extrap = 5 / args.timepoints * n_total_tp

    distribution = uniform.Uniform(torch.Tensor([0.0]), torch.Tensor([max_t_extrap]))
    time_steps_extrap = distribution.sample(torch.Size([n_total_tp-1]))[:, 0]
    time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
    time_steps_extrap = torch.sort(time_steps_extrap)[0]

    if dataset_name.startswith(climate_data_name):
        name_dir = climate_data_dir
        name_dir_base = os.path.join(os.path.join(data_folder, name_dir), dataset_name)
        time_stamps_var = 1
        name_train = name_dir_base + '/training_samples'
        name_train_mask = name_dir_base + '/training_masks'
        name_test = name_dir_base + '/test_samples'
        name_test_mask = name_dir_base + '/test_masks'
        assert_var = True
        min_time_series_len_var = True
        remove_outlier_func_var = 1
        data_train_len = climate_data_train_len

    elif dataset_name == 'physionet':
        name_dir = physionet_data_dir
        name_dir_base = os.path.join(data_folder, name_dir)
        time_stamps_var = 1
        name_train = name_dir_base + '/train_dataset_tensor'
        name_train_mask = name_dir_base + '/train_mask_tensor'
        name_test = name_dir_base + '/test_dataset_tensor'
        name_test_mask = name_dir_base + '/test_mask_tensor'
        assert_var = False
        min_time_series_len_var = True
        remove_outlier_func_var = 2
        data_train_len = physionet_data_train_len

    elif dataset_name.startswith('mimic3'):
        name_dir = mimic3_data_dir
        name_dir_base = os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name)
        time_stamps_var = 72
        name_train = name_dir_base + '/mimic3_train_tensor'
        name_train_mask = name_dir_base + '/mimic3_train_masks'
        name_test = name_dir_base + '/mimic3_test_tensor'
        name_test_mask = name_dir_base + '/mimic3_test_masks'
        assert_var = True
        min_time_series_len_var = False
        remove_outlier_func_var = 2
        data_train_len = mimic3_data_train_len

    if args.new:

        train_dataset = torch.load(name_train).type(torch.FloatTensor)

        test_dataset = torch.load(name_test).type(torch.FloatTensor)

        if time_stamps_var == 1 :
            train_y = train_dataset

            train_time_stamps = torch.tensor(list(range(train_y.shape[1])))

            train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])

            train_lens = torch.ones(train_y.shape[0], dtype=torch.long)*train_y.shape[1]

            masks_train = torch.load(name_train_mask)

            test_y = test_dataset

            test_lens = torch.ones(test_y.shape[0], dtype=torch.long)*test_y.shape[1]

            test_time_stamps = torch.tensor(list(range(test_y.shape[1])))

            test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])

            masks_test = torch.load(name_test_mask)

        elif time_stamps_var == 72 :

            train_y = train_dataset[:, :, 1:]

            single_train_time_stamp = torch.tensor(list(range(72)))

            train_time_stamps = single_train_time_stamp.view(1, 72)

            train_time_stamps = train_time_stamps.repeat(train_dataset.shape[0], 1)

            train_lens = torch.tensor(72)

            train_lens = train_lens.repeat(train_dataset.shape[0])

            masks_train = torch.load(name_train_mask)[:, :, 1:]

            test_y = test_dataset[:, :, 1:]

            single_test_time_stamp = torch.tensor(list(range(72)))

            test_time_stamps = single_test_time_stamp.view(1, 72)

            test_time_stamps = test_time_stamps.repeat(test_dataset.shape[0], 1)

            masks_test = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name)
                                    + '/mimic3_test_masks')[:, :, 1:]

            test_lens = torch.tensor(72)

            test_lens = test_lens.repeat(test_dataset.shape[0])

        if args.model in models_to_remove_none_time_stamps:

            new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(train_y, masks_train, train_time_stamps, train_lens)

            new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(test_y, masks_test, test_time_stamps, test_lens)

            check_remove_none(train_y, new_train_y)

            train_y = new_train_y

            test_y = new_test_y

        train_delta_time_stamps = train_time_stamps.clone()

        test_delta_time_stamps = test_time_stamps.clone()

        if args.model == 'DHMM_cluster_tlstm':
            train_delta_time_stamps = get_delta_time_stamps_all_dims(train_time_stamps)

            test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)

        if args.model == GRUD_method:
            train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)

            test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)

        print(torch.norm(torch.sum(masks_train, 2) - torch.sum(1-np.isnan(train_y), 2)))
        print(torch.norm(torch.sum(masks_train, [1, 2]) - torch.sum(1-np.isnan(train_y), [1, 2])))

        if assert_var :
            assert torch.sum(masks_train, dtype=torch.double) == torch.sum(1-np.isnan(train_y), dtype=torch.double)  # diff entre physionet et mimics

        all_features_not_all_missing_values = get_features_with_one_value(masks_train.clone(), masks_test.clone())

        train_y = train_y[:, :, all_features_not_all_missing_values]

        masks_train = masks_train[:, :, all_features_not_all_missing_values]

        if min_time_series_len_var :

            train_y = train_y[train_lens >= min_time_series_len]

            train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]

            if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
                train_delta_time_stamps = train_delta_time_stamps[train_lens >= min_time_series_len]
            else:
                train_delta_time_stamps = train_time_stamps.clone()

            masks_train = masks_train[train_lens >= min_time_series_len]

            train_lens = train_lens[train_lens >= min_time_series_len]

            if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
                test_delta_time_stamps = test_delta_time_stamps[test_lens >= 1]
            else:
                test_delta_time_stamps = test_time_stamps.clone()

        test_y = test_y[:, :, all_features_not_all_missing_values]

        masks_test = masks_test[:, :, all_features_not_all_missing_values]

        test_y = test_y[test_lens > 1]

        test_time_stamps = test_time_stamps[test_lens >= 1]

        masks_test = masks_test[test_lens > 1]

        test_lens = test_lens[test_lens > 1]

        train_y[train_y != train_y] = -1000

        test_y[test_y != test_y] = -1000

        args.n = train_y.shape[0] + test_y.shape[0]

        if remove_outlier_func_var == 1 :

            train_y, masks_train = remove_outliers(train_y, masks_train)
            test_y, masks_test = remove_outliers(test_y, masks_test)

        elif remove_outlier_func_var == 2 :

            train_y, masks_train = remove_outliers2(train_y, masks_train)
            test_y, masks_test = remove_outliers2(test_y, masks_test)

        origin_train_masks = masks_train.clone()

        origin_test_masks = masks_test.clone()

        random_train_masks = torch.ones_like(origin_train_masks)

        random_test_masks = torch.ones_like(origin_test_masks)

        train_y, test_y = standardize_dataset(train_y, test_y, masks_train)

        masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, data_train_len)

        masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, data_train_len)

        if args.model == 'Linear_regression':

            train_y_copy = train_y.clone()

            test_y_copy = test_y.clone()

            train_y_copy[masks_train == 0] = 0
            test_y_copy[masks_test == 0] = 0

            print(torch.norm((train_y_copy - train_y)*masks_train))

            train_y = train_y_copy

            test_y = test_y_copy

        if args.model == cluster_ODE_method or args.model == l_ODE_method:
            train_time_stamps = train_time_stamps.type(torch.float)/train_time_stamps.shape[1]
            test_time_stamps = test_time_stamps.type(torch.float)/train_time_stamps.shape[1]

        wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)

        wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        if not os.path.exists(data_folder + name_dir):
            os.makedirs(data_folder + name_dir)

        torch.save(wrapped_train_y, name_dir_base + '/dataset_train_y')

        torch.save(wrapped_test_y, name_dir_base + '/dataset_test_y')

        torch.save(time_steps_extrap, name_dir_base + '/time_steps')

    else:

        wrapped_train_y = torch.load(name_dir_base + '/dataset_train_y')

        wrapped_test_y = torch.load(name_dir_base + '/dataset_test_y')

        time_steps_extrap = torch.load(name_dir_base + '/time_steps')

        args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]

    is_missing = torch.sum(wrapped_train_y.mask) < (wrapped_train_y.mask.shape[0]*wrapped_train_y.mask.shape[1]*wrapped_train_y.mask.shape[2])

    input_dim = wrapped_train_y.data.shape[-1]

    batch_size = args.batch_size
    train_dataloader = DataLoader(wrapped_train_y, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type="train"))
    test_dataloader = DataLoader(wrapped_test_y, batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type="test"))

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader)}

    train_mean = get_train_mean(data_objects, dataset_name)

    return data_objects, time_steps_extrap, is_missing, train_mean


def basic_collate_fn(batch, args, data_type="train"):
    """
    A collate function typically used with PyTorch's DataLoader to prepare batches 
    of data for training or evaluation. This function groups the data provided in each 
    batch into a dictionary format that is suitable for further processing in your model. 
    The function takes input in the form of a batch, which is a list of individual samples, 
    where each sample is represented as a tuple of tensors. The function then constructs 
    a dictionary containing various data elements and returns it.

    """
    (   
        batched_data,
        batched_mask,
        batched_origin_data,
        batched_origin_masks,
        batched_new_random_masks,
        batched_tensor_len,
        batched_time_stamps,
        batched_delta_time_stamps,
        batched_ids
    ) = zip(*batch)

    data_dict = {
        "data": torch.stack(batched_data),
        "lens": torch.tensor(batched_tensor_len),
        "origin_data": torch.stack(batched_origin_data),
        "origin_mask": torch.stack(batched_origin_masks),
        "time_stamps": torch.stack(batched_time_stamps),
        "delta_time_stamps": torch.stack(batched_delta_time_stamps),
        "new_random_mask": torch.stack(batched_new_random_masks),
        "ids": torch.tensor(batched_ids),
        "mask": torch.stack(batched_mask)
        }

    data_dict = split_and_subsample_batch(data_dict, args, data_type=data_type)
    return data_dict


def partition_validation_set(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps):

    validation_size = int(train_y.shape[0]/8)

    all_ids = torch.randperm(train_y.shape[0])

    validation_ids = all_ids[0:validation_size]

    new_training_ids = all_ids[validation_size:]

    valid_y = train_y[validation_ids]

    valid_mask_train = masks_train[validation_ids]

    valid_origin_train_masks = origin_train_masks[validation_ids]

    valid_random_train_masks = random_train_masks[validation_ids]

    valid_train_lens = train_lens[validation_ids]

    valid_train_time_stamps = train_time_stamps[validation_ids]

    valid_train_delta_time_stamps = train_delta_time_stamps[validation_ids]

    wrapped_valid_y = MyDataset(valid_y, valid_mask_train, valid_origin_train_masks, valid_random_train_masks, valid_train_lens, valid_train_time_stamps, valid_train_delta_time_stamps)

    print('validation data size::', valid_y.shape[0])

    new_train_y = train_y[new_training_ids]

    new_masks_train = masks_train[new_training_ids]

    new_origin_train_masks = origin_train_masks[new_training_ids]

    new_random_train_masks = random_train_masks[new_training_ids]

    new_train_lens = train_lens[new_training_ids]

    new_train_time_stamps = train_time_stamps[new_training_ids]

    new_train_delta_time_stamps = train_delta_time_stamps[new_training_ids]

    wrapped_train_y = MyDataset(new_train_y, new_masks_train, new_origin_train_masks, new_random_train_masks, new_train_lens, new_train_time_stamps, new_train_delta_time_stamps)

    print('training data size::', new_train_y.shape[0])

    return wrapped_train_y, wrapped_valid_y


def generate_new_time_series(args):

    dataset_name = args.dataset
    climate_var = 0
    time_stamps_var = 0

    if dataset_name.startswith(climate_data_name):
        climate_var = 1
        name_dir = climate_data_dir
        name_dir_base = os.path.join(os.path.join(data_folder, name_dir), dataset_name)
        name_train = name_dir_base + '/training_samples'
        name_train_mask = name_dir_base + '/training_masks'
        name_test = name_dir_base + '/test_samples'
        name_test_mask = name_dir_base + '/test_masks'
        min_time_series_len_var = True
        data_train_len = climate_data_train_len

    elif dataset_name.startswith(kddcup_data_name):
        print('kdd')
        climate_var = 1
        name_dir = beijing_data_dir
        name_dir_base = os.path.join(data_folder, name_dir)
        name_train = name_dir_base + '/training_tensor'
        name_train_mask = name_dir_base + '/training_mask'
        name_test = name_dir_base + '/test_tensor'
        name_test_mask = name_dir_base + '/test_mask'
        min_time_series_len_var = True
        data_train_len = beijing_data_train_len

    elif dataset_name.startswith(mimic_data_name) :
        print('mimic')
        time_stamps_var = 1
        name_dir = mimic3_data_dir
        name_dir_base = os.path.join(os.path.join(data_folder, name_dir), dataset_name)
        name_train = name_dir_base + '/mimic3_train_tensor'
        name_train_mask = name_dir_base + '/mimic3_train_masks'
        name_test = name_dir_base + '/mimic3_test_tensor'
        name_test_mask = name_dir_base + '/mimic3_test_masks'
        min_time_series_len_var = False
        data_train_len = mimic3_data_train_len

    if climate_var == 1 :

        print('generate climate time series')

    train_dataset = torch.load(name_train).type(torch.FloatTensor)

    test_dataset = torch.load(name_test).type(torch.FloatTensor)

    if time_stamps_var == 1 :

        masks_train = torch.load(name_train_mask)[:, :, 1:]

        train_y = train_dataset[:, :, 1:]

        masks_test = torch.load(name_test_mask)[:, :, 1:]

        test_y = test_dataset[:, :, 1:]

        if dataset_name == 'mimic3_17_5':

            time_gap_in_hour = 1.0/12

            time_stamp_count = int(6/time_gap_in_hour)

            single_train_time_stamp = torch.tensor(list(range(time_stamp_count)))*time_gap_in_hour

            train_time_stamps = single_train_time_stamp.view(1, time_stamp_count)

            train_time_stamps = train_time_stamps.repeat(train_dataset.shape[0], 1)

            train_lens = torch.tensor(time_stamp_count)

            train_lens = train_lens.repeat(train_dataset.shape[0])

            single_test_time_stamp = torch.tensor(list(range(time_stamp_count)))*time_gap_in_hour

            test_time_stamps = single_test_time_stamp.view(1, time_stamp_count)

            test_time_stamps = test_time_stamps.repeat(test_dataset.shape[0], 1)

            test_lens = torch.tensor(time_stamp_count)

            test_lens = test_lens.repeat(test_dataset.shape[0])

        else:
            single_train_time_stamp = torch.tensor(list(range(mimic3_data_len)))

            train_time_stamps = single_train_time_stamp.view(1, mimic3_data_len)

            train_time_stamps = train_time_stamps.repeat(train_dataset.shape[0], 1)

            train_lens = torch.tensor(mimic3_data_len)

            train_lens = train_lens.repeat(train_dataset.shape[0])

            train_y = train_y[:, 0:mimic3_data_len]

            masks_train = masks_train[:, 0:mimic3_data_len]

            single_test_time_stamp = torch.tensor(list(range(mimic3_data_len)))

            test_time_stamps = single_test_time_stamp.view(1, mimic3_data_len)

            test_time_stamps = test_time_stamps.repeat(test_dataset.shape[0], 1)

            test_lens = torch.tensor(mimic3_data_len)

            test_lens = test_lens.repeat(test_dataset.shape[0])

            test_y = test_y[:, 0:mimic3_data_len]

            masks_test = masks_test[:, 0:mimic3_data_len]

        print('non missing ratio::', torch.mean(masks_train))

        assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))

    else :
        masks_train = torch.load(name_train_mask)

        train_y = train_dataset

        train_time_stamps = torch.tensor(list(range(train_y.shape[1])))

        train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])

        train_lens = torch.ones(train_y.shape[0], dtype=torch.long)*train_y.shape[1]

        test_y = test_dataset

        test_lens = torch.ones(test_y.shape[0], dtype=torch.long)*test_y.shape[1]

        test_time_stamps = torch.tensor(list(range(test_y.shape[1])))

        test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])

        masks_test = torch.load(name_test_mask)

        print(torch.norm(torch.sum(masks_train, 2) - torch.sum(1-np.isnan(train_y), 2)))
        print(torch.norm(torch.sum(masks_train, [1, 2]) - torch.sum(1-np.isnan(train_y), [1, 2])))

        assert torch.sum(masks_train, dtype=torch.double) == torch.sum(1-np.isnan(train_y), dtype=torch.double)

    train_delta_time_stamps = train_time_stamps.clone()

    test_delta_time_stamps = test_time_stamps.clone()

    all_features_not_all_missing_values = get_features_with_one_value(masks_train.clone(), masks_test.clone())

    train_y = train_y[:, :, all_features_not_all_missing_values]

    masks_train = masks_train[:, :, all_features_not_all_missing_values]

    if min_time_series_len_var :

        train_y = train_y[train_lens >= min_time_series_len]

        train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]

        masks_train = masks_train[train_lens >= min_time_series_len]

        train_lens = train_lens[train_lens >= min_time_series_len]

    test_y = test_y[:, :, all_features_not_all_missing_values]

    masks_test = masks_test[:, :, all_features_not_all_missing_values]

    test_y = test_y[test_lens > 1]

    test_time_stamps = test_time_stamps[test_lens >= 1]

    masks_test = masks_test[test_lens > 1]

    test_lens = test_lens[test_lens > 1]

    train_y[train_y != train_y] = -1000

    test_y[test_y != test_y] = -1000

    args.n = train_y.shape[0] + test_y.shape[0]

    train_y, masks_train = remove_outliers2(train_y, masks_train)

    test_y, masks_test = remove_outliers2(test_y, masks_test)

    origin_train_masks = masks_train.clone()

    origin_test_masks = masks_test.clone()

    random_train_masks = torch.ones_like(origin_train_masks)

    random_test_masks = torch.ones_like(origin_test_masks)

    train_y, test_y = standardize_dataset(train_y, test_y, masks_train)

    masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, data_train_len)

    masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, data_train_len)

    wrapped_train_y, wrapped_valid_y = partition_validation_set(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)

    wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if not os.path.exists(data_folder + name_dir):
        os.makedirs(data_folder + name_dir)

    torch.save(wrapped_train_y, name_dir_base + '/dataset_train_y')

    torch.save(wrapped_valid_y, name_dir_base + '/dataset_valid_y')

    torch.save(wrapped_test_y, name_dir_base + '/dataset_test_y')


def load_time_series_dataset(dataset_name):

    if dataset_name.startswith(climate_data_name):
        data_dir = os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name)

        training_data_file_name = 'dataset_train_y'

        test_data_file_name = 'dataset_test_y'

        inference_len = climate_data_train_len

    wrapped_train_y = torch.load(data_dir + '/' + training_data_file_name)

    wrapped_test_y = torch.load(data_dir + '/' + test_data_file_name)

    return wrapped_train_y, wrapped_test_y, inference_len


def load_time_series(args):

    dataset_name = args.dataset

    data_dir = ""

    training_data_file_name = None

    test_data_file_name = None

    inference_len = 0

    if dataset_name.startswith(mimic_data_name):

        data_dir = os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name)

        training_data_file_name = 'dataset_train_y'

        test_data_file_name = 'dataset_test_y'

        valid_data_file_name = 'dataset_valid_y'

        inference_len = mimic3_data_train_len

    if dataset_name.startswith(climate_data_name):
        data_dir = os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name)

        training_data_file_name = 'dataset_train_y'

        test_data_file_name = 'dataset_test_y'

        valid_data_file_name = 'dataset_valid_y'

        inference_len = climate_data_train_len

    if dataset_name.startswith(kddcup_data_name):
        data_dir = os.path.join(data_folder, beijing_data_dir)

        training_data_file_name = 'dataset_train_y'

        valid_data_file_name = 'dataset_valid_y'

        test_data_file_name = 'dataset_test_y'

        inference_len = beijing_data_train_len

    wrapped_train_y = torch.load(data_dir + '/' + training_data_file_name)

    wrapped_valid_y = torch.load(data_dir + '/' + valid_data_file_name)

    wrapped_test_y = torch.load(data_dir + '/' + test_data_file_name)

    is_missing = torch.sum(wrapped_train_y.mask) < \
        (wrapped_train_y.mask.shape[0]*wrapped_train_y.mask.shape[1]*wrapped_train_y.mask.shape[2])

    if args.model == cluster_ODE_method or args.model == l_ODE_method:
        new_train_time_stamps = wrapped_train_y.time_stamps.type(torch.float)/wrapped_train_y.time_stamps.shape[1]
        new_test_time_stamps = wrapped_test_y.time_stamps.type(torch.float)/wrapped_test_y.time_stamps.shape[1]
        new_valid_time_stamps = wrapped_valid_y.time_stamps.type(torch.float)/wrapped_valid_y.time_stamps.shape[1]

        wrapped_train_y.time_stamps = new_train_time_stamps
        wrapped_valid_y.time_stamps = new_valid_time_stamps
        wrapped_test_y.time_stamps = new_test_time_stamps

    input_dim = wrapped_train_y.data.shape[-1]

    args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]

    print('training data size::', wrapped_train_y.data.shape)

    print('validation data size::', wrapped_valid_y.data.shape)

    print('test data size::', wrapped_test_y.data.shape)

    print('inference missing ratio::', 1 - torch.mean(torch.cat([wrapped_train_y.mask[:, 0:inference_len], wrapped_valid_y.mask[:, 0:inference_len], wrapped_test_y.mask[:, 0:inference_len]], 0)))

    print('forecasting missing ratio::', 1 - torch.mean(torch.cat([wrapped_train_y.mask[:, inference_len:], wrapped_valid_y.mask[:, inference_len:], wrapped_test_y.mask[:, inference_len:]], 0)))

    batch_size = args.batch_size
    train_dataloader = DataLoader(wrapped_train_y, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda batch: basic_collate_fn(batch, args, data_type="train"))
    valid_dataloader = DataLoader(wrapped_valid_y, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda batch: basic_collate_fn(batch, args, data_type="train"))
    test_dataloader = DataLoader(wrapped_test_y, batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda batch: basic_collate_fn(batch, args, data_type="test"))

    data_objects = {
                "train_dataloader": train_dataloader,
                "valid_dataloader": valid_dataloader,
                "test_dataloader": test_dataloader,
                "input_dim": input_dim,
                "n_train_batches": len(train_dataloader),
                "n_test_batches": len(test_dataloader)
                }

    train_mean = get_train_mean(data_objects, inference_len)

    return data_objects, is_missing, train_mean


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--dataset', type=str, default='USHCN', help='name of dataset')
    parser.add_argument('-ms', '--missing_ratio', type=float, default=0.00)

    args = parser.parse_args()
    generate_new_time_series(args)
    print('generate time series done!!!')

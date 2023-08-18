import argparse
import os
import sys
import numpy as np
import torch
from torch.autograd import Variable

import configs
import models
from data.generate_time_series import *
from lib.utils import *

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/lib")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/imputation")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/initialize")


# setup, training, and evaluation



def evaluate_imputation_errors(data_obj, model, is_GPU, device):
    """
    This function computes and prints the imputation errors for training and testing data.
    It iterates through the data batches, calculates imputed MSE and MAE losses for each batch, 
    and accumulates these losses. It calculates the final imputation RMSE and MAE losses by 
    dividing the accumulated losses by the total number of imputed values and prints them.

    Arguments :
        - data_obj: A dictionary containing data and dataloaders.
        - model: The model used for imputation.
        - is_GPU: Boolean indicating whether to use GPU.
        - device: The device used for memory.
    """
    with torch.no_grad():

        training_imputed_mse_loss = 0

        training_imputed_mse_loss2 = 0

        training_imputed_mae_loss = 0

        training_imputed_mae_loss2 = 0

        testing_imputed_mse_loss = 0

        testing_imputed_mse_loss2 = 0

        testing_imputed_mae_loss = 0

        testing_imputed_mae_loss2 = 0

        training_count = 0

        testing_count = 0

        model.evaluate = True

        for _, batch_dict in enumerate(data_obj["train_dataloader"]):
            _, (
                imputed_mse_loss,
                imputed_mse_loss2,
                imputed_loss,
                imputed_loss2,
            ) = model.infer(
                batch_dict["observed_data"],
                batch_dict["origin_observed_data"],
                batch_dict["observed_mask"],
                batch_dict["observed_origin_mask"],
                batch_dict["observed_new_mask"],
                batch_dict["observed_lens"],
                batch_dict["data_to_predict"],
                batch_dict["origin_data_to_predict"],
                batch_dict["mask_predicted_data"],
                batch_dict["origin_mask_predicted_data"],
                batch_dict["new_mask_predicted_data"],
                batch_dict["lens_to_predict"],
                is_GPU,
                device,
            )

            new_x_mask_count = torch.sum(1 - batch_dict["observed_new_mask"])

            training_count += new_x_mask_count

            training_imputed_mse_loss += imputed_mse_loss**2 * new_x_mask_count

            training_imputed_mse_loss2 += imputed_mse_loss2**2 * new_x_mask_count

            training_imputed_mae_loss += imputed_loss * new_x_mask_count

            training_imputed_mae_loss2 += imputed_loss2 * new_x_mask_count

        for _, batch_dict in enumerate(data_obj["test_dataloader"]):
            _, (
                imputed_mse_loss,
                imputed_mse_loss2,
                imputed_loss,
                imputed_loss2,
            ) = model.infer(
                batch_dict["observed_data"],
                batch_dict["origin_observed_data"],
                batch_dict["observed_mask"],
                batch_dict["observed_origin_mask"],
                batch_dict["observed_new_mask"],
                batch_dict["observed_lens"],
                batch_dict["data_to_predict"],
                batch_dict["origin_data_to_predict"],
                batch_dict["mask_predicted_data"],
                batch_dict["origin_mask_predicted_data"],
                batch_dict["new_mask_predicted_data"],
                batch_dict["lens_to_predict"],
                is_GPU,
                device,
            )

            new_x_mask_count = torch.sum(1 - batch_dict["observed_new_mask"])

            testing_count += new_x_mask_count

            testing_imputed_mse_loss += imputed_mse_loss**2 * new_x_mask_count

            testing_imputed_mse_loss2 += imputed_mse_loss2**2 * new_x_mask_count

            testing_imputed_mae_loss += imputed_loss * new_x_mask_count

            testing_imputed_mae_loss2 += imputed_loss2 * new_x_mask_count

        final_training_imputed_mse_loss = torch.sqrt(
            training_imputed_mse_loss / training_count
        )

        final_training_imputed_mse_loss2 = torch.sqrt(
            training_imputed_mse_loss2 / training_count
        )

        final_training_imputed_mae_loss = training_imputed_mae_loss / training_count

        final_training_imputed_mae_loss2 = training_imputed_mae_loss2 / training_count

        final_testing_imputed_mse_loss = torch.sqrt(
            testing_imputed_mse_loss / testing_count
        )

        final_testing_imputed_mse_loss2 = torch.sqrt(
            testing_imputed_mse_loss2 / testing_count
        )

        final_testing_imputed_mae_loss = testing_imputed_mae_loss / testing_count

        final_testing_imputed_mae_loss2 = testing_imputed_mae_loss2 / testing_count

        print("training imputation rmse loss::", final_training_imputed_mse_loss)

        print("training imputation rmse loss 2::", final_training_imputed_mse_loss2)

        print("training imputation mae loss::", final_training_imputed_mae_loss)

        print("training imputation mae loss 2::", final_training_imputed_mae_loss2)

        print("testing imputation rmse loss::", final_testing_imputed_mse_loss)

        print("testing imputation rmse loss 2::", final_testing_imputed_mse_loss2)

        print("testing imputation mae loss::", final_testing_imputed_mae_loss)

        print("testing imputation mae loss 2::", final_testing_imputed_mae_loss2)


def evaluate(data_obj, model, is_GPU, device, objective):
    """
    This function performs evaluation on forecasting and imputation for a 
    specific objective (test or validation). It iterates through the data batches, 
    calculates various losses including RMSE and MAE, and accumulates these losses.
    It also computes forecasting RMSE and MAE losses over time steps.
    Finally, it prints the evaluation results including forecasting and imputation losses.

    Arguments:
        - data_obj: A dictionary containing data and dataloaders.
        - model: The model used for evaluation.
        - is_GPU: Boolean indicating whether to use GPU.
        - device: The device used for memory.
        - objective: A string indicating the evaluation objective ("test" or "valid").
    """

    final_rmse_loss = 0

    final_rmse_loss2 = 0

    final_mae_losses = 0

    final_mae_losses2 = 0

    final_nll_loss = 0

    final_nll_loss2 = 0

    all_count1 = 0

    all_count2 = 0

    final_imputed_rmse_loss = 0

    final_imputed_mae_loss = 0

    final_imputed_rmse_loss2 = 0

    final_imputed_mae_loss2 = 0

    all_count3 = 0

    all_count4 = 0

    all_count5 = 0

    forecasting_rmse_list = 0

    forecasting_mae_list = 0

    forecasting_rmse_list2 = 0

    forecasting_mae_list2 = 0

    forecasting_count = 0

    with torch.no_grad():

        for _, data_dict in enumerate(data_obj[objective + "_dataloader"]):

            batch_dict = data_dict

            (
                rmse_loss,
                rmse_loss_count,
                mae_losses,
                mae_loss_count,
                nll_loss,
                nll_loss_count,
                list_res,
                imputed_res,
            ) = model.test_samples(
                Variable(batch_dict["observed_data"]),
                Variable(batch_dict["origin_observed_data"]),
                Variable(batch_dict["observed_mask"]),
                Variable(batch_dict["observed_origin_mask"]),
                Variable(batch_dict["observed_new_mask"]),
                Variable(batch_dict["observed_lens"]),
                Variable(batch_dict["data_to_predict"]),
                Variable(batch_dict["origin_data_to_predict"]),
                Variable(batch_dict["mask_predicted_data"]),
                Variable(batch_dict["origin_mask_predicted_data"]),
                Variable(batch_dict["new_mask_predicted_data"]),
                Variable(batch_dict["lens_to_predict"]),
                is_GPU,
                device,
                batch_dict["delta_time_stamps"],
                batch_dict["delta_time_stamps_to_predict"],
                batch_dict["time_stamps"],
                batch_dict["time_stamps_to_predict"],
            )

            all_count1 += rmse_loss_count

            all_count2 += mae_loss_count

            if type(rmse_loss) is tuple and len(list(rmse_loss)) == 2:

                rmse_loss_list = list(rmse_loss)

                mae_loss_list = list(mae_losses)

                final_rmse_loss += (rmse_loss_list[0] ** 2) * rmse_loss_count

                final_mae_losses += (mae_loss_list[0]) * mae_loss_count

                final_rmse_loss2 += (rmse_loss_list[1] ** 2) * rmse_loss_count

                final_mae_losses2 += (mae_loss_list[1]) * mae_loss_count

            else:
                final_rmse_loss += (rmse_loss**2) * rmse_loss_count

                final_mae_losses += (mae_losses) * mae_loss_count

            if nll_loss_count is not None:

                if type(nll_loss) is tuple and len(list(nll_loss)) == 2:
                    nll_loss_list = list(nll_loss)

                    final_nll_loss += (nll_loss_list[0]) * nll_loss_count

                    final_nll_loss2 += (nll_loss_list[1]) * nll_loss_count

                else:
                    final_nll_loss += (nll_loss) * nll_loss_count
                all_count3 += nll_loss_count
            else:
                final_nll_loss = None

            if imputed_res is not None:
                (
                    imputed_mae_res,
                    imputed_mae_count,
                    imputed_rmse_res,
                    imputed_rmse_count,
                ) = imputed_res

                if type(imputed_mae_res) is tuple:

                    imputed_rmse_loss, imputed_rmse_loss2 = imputed_rmse_res

                    imputed_mae_loss, imputed_mae_loss2 = imputed_mae_res

                    final_imputed_rmse_loss += (
                        imputed_rmse_loss**2
                    ) * imputed_rmse_count

                    final_imputed_mae_loss += (imputed_mae_loss) * imputed_mae_count

                    final_imputed_rmse_loss2 += (
                        imputed_rmse_loss2**2
                    ) * imputed_rmse_count

                    final_imputed_mae_loss2 += (imputed_mae_loss2) * imputed_mae_count
                else:

                    final_imputed_rmse_loss += (
                        imputed_rmse_res**2
                    ) * imputed_rmse_count

                    final_imputed_mae_loss += (imputed_mae_res) * imputed_mae_count

                all_count4 += imputed_rmse_count

                all_count5 += imputed_mae_count

            if type(list_res[0]) is tuple:

                curr_forecasting_rmse_list = list(list_res[0])[0]

                curr_forecasting_mae_list = list(list_res[1])[0]

                curr_forecasting_rmse_list2 = list(list_res[0])[1]

                curr_forecasting_mae_list2 = list(list_res[1])[1]

                curr_forecasting_count = list_res[2]

                forecasting_rmse_list += (
                    curr_forecasting_rmse_list**2
                ) * curr_forecasting_count

                forecasting_mae_list += (
                    curr_forecasting_mae_list * curr_forecasting_count
                )

                forecasting_rmse_list2 += (
                    curr_forecasting_rmse_list2**2
                ) * curr_forecasting_count

                forecasting_mae_list2 += (
                    curr_forecasting_mae_list2 * curr_forecasting_count
                )

                forecasting_count += curr_forecasting_count

            else:
                curr_forecasting_rmse_list = list_res[0]

                curr_forecasting_mae_list = list_res[1]

                curr_forecasting_count = list_res[2]

                forecasting_rmse_list += (
                    curr_forecasting_rmse_list**2
                ) * curr_forecasting_count

                forecasting_mae_list += (
                    curr_forecasting_mae_list * curr_forecasting_count
                )

                forecasting_count += curr_forecasting_count

    final_rmse_loss = torch.sqrt(final_rmse_loss / all_count1)

    final_mae_losses = final_mae_losses / all_count2

    final_rmse_loss2 = torch.sqrt(final_rmse_loss2 / all_count1)

    final_mae_losses2 = final_mae_losses2 / all_count2

    print(objective + " results::")

    print(objective + " forecasting rmse loss::", final_rmse_loss)

    print(objective + " forecasting mae loss::", final_mae_losses)

    print(objective + " forecasting rmse loss 2::", final_rmse_loss2)

    print(objective + " forecasting mae loss 2::", final_mae_losses2)

    if final_nll_loss is not None:

        final_nll_loss = final_nll_loss / all_count3

        final_nll_loss2 = final_nll_loss2 / all_count3

        print(objective + " forecasting neg likelihood::", final_nll_loss)

        print(objective + " forecasting neg likelihood 2::", final_nll_loss2)

    forecasting_rmse_list = torch.sqrt(forecasting_rmse_list / forecasting_count)

    forecasting_mae_list = forecasting_mae_list / forecasting_count

    forecasting_rmse_list2 = torch.sqrt(forecasting_rmse_list2 / forecasting_count)

    forecasting_mae_list2 = forecasting_mae_list2 / forecasting_count

    print(objective + " forecasting rmse loss by time steps::")

    print(forecasting_rmse_list)

    print(forecasting_mae_list)

    print(objective + " forecasting rmse loss 2 by time steps::")

    print(forecasting_rmse_list2)

    print(forecasting_mae_list2)

    if imputed_res is not None:
        final_imputed_rmse_loss = torch.sqrt(final_imputed_rmse_loss / all_count4)
        final_imputed_rmse_loss2 = torch.sqrt(final_imputed_rmse_loss2 / all_count4)
        final_imputed_mae_loss = final_imputed_mae_loss / all_count5
        final_imputed_mae_loss2 = final_imputed_mae_loss2 / all_count5

    print(objective + " imputation rmse loss::", final_imputed_rmse_loss)

    print(objective + " imputation mae loss::", final_imputed_mae_loss)

    print(objective + " imputation rmse loss 2::", final_imputed_rmse_loss2)

    print(objective + " imputation mae loss 2::", final_imputed_mae_loss2)

    if not os.path.exists(data_dir + output_dir):
        os.makedirs(data_dir + output_dir)
    torch.save(model, data_dir + output_dir + "model")

    return (
        final_rmse_loss,
        final_mae_losses,
        final_rmse_loss2,
        final_mae_losses2,
        final_imputed_rmse_loss,
        final_imputed_mae_loss,
        final_imputed_rmse_loss2,
        final_imputed_mae_loss2,
    )


def test(data_obj, model, is_GPU, device):
    """
    This function calls the evaluate function with the "test" 
    objective and returns the results.

    Arguments:
        - data_obj: A dictionary containing data and dataloaders.
        - model: The model used for evaluation.
        - is_GPU: Boolean indicating whether to use GPU.
        - device: The device used for memory.
    """
    return evaluate(data_obj, model, is_GPU, device, objective='test')


def validate(data_obj, model, is_GPU, device):
    """
    This function calls the evaluate function with the "valid" objective 
    and returns the RMSE loss.

    Arguments:
        - data_obj: A dictionary containing data and dataloaders.
        - model: The model used for evaluation.
        - is_GPU: Boolean indicating whether to use GPU.
        - device: The device used for memory.
    """
    return_evaluate_list = evaluate(data_obj, model, is_GPU, device, objective='valid')
    return return_evaluate_list[0]


def print_test_res(all_valid_rmse_list, all_test_res, args):
    """
    This function prints the test results, including forecasting and 
    imputation RMSE and MAE losses.

    Arguments:
        - all_valid_rmse_list: A list containing all validation RMSE losses.
        - all_test_res: A list containing all test results (output from the test function).
        - args: The command-line arguments used for running the script.
    """

    all_valid_rmse_array = np.array(all_valid_rmse_list)

    selected_id = np.argmin(all_valid_rmse_array)

    test_res = all_test_res[selected_id]

    (
        final_rmse_loss,
        final_mae_losses,
        final_rmse_loss2,
        final_mae_losses2,
        final_imputed_rmse_loss,
        final_imputed_mae_loss,
        final_imputed_rmse_loss2,
        final_imputed_mae_loss2,
    ) = test_res

    print("test results::")

    if args.model.startswith(cluster_ODE_method):

        final_rmse_loss = min(final_rmse_loss, final_rmse_loss2)

        final_mae_losses = min(final_mae_losses, final_mae_losses2)

        final_imputed_rmse_loss = min(final_imputed_rmse_loss, final_imputed_rmse_loss2)

        final_imputed_mae_loss = min(final_imputed_mae_loss, final_imputed_mae_loss2)
    
    print("test forecasting rmse loss::", final_rmse_loss)

    print("test forecasting mae loss::", final_mae_losses)

    print("test imputation rmse loss::", final_imputed_rmse_loss)

    print("test imputation mae loss::", final_imputed_mae_loss)


def main(args):
    # setup logging

    args.GRUD = False
    if args.model == GRUD_method:
        args.GRUD = True

    config = getattr(configs, "config_" + args.model)()

    data_obj, is_missing, train_mean = load_time_series(args)

    config["cluster_num"] = args.cluster_num

    config["input_dim"] = data_obj["input_dim"]

    config["phi_std"] = args.std

    config["epochs"] = args.epochs

    config["is_missing"] = is_missing

    if args.use_gate:
        config["use_gate"] = True
    else:
        config["use_gate"] = False

    config["train_mean"] = train_mean

    config["gaussian"] = args.gaussian

    max_kl = args.max_kl

    is_GPU = args.GPU

    if not is_GPU:
        device = torch.device("cpu")
    else:
        GPU_ID = int(args.GPUID)
        device = torch.device(
            "cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu"
        )
    config["device"] = device
    model = getattr(models, args.model)(config)

    model.init_params()
    model = model.to(device)

    #################
    # TRAINING LOOP #
    #################

    wait_until_kl_inc = args.wait_epoch

    itr = 0

    test_period = 1

    all_valid_rmse_list = []

    all_test_res = []

    for epoch in range(config["epochs"]):

        # accumulator for our estimate of the negative log
        # likelihood (or rather -elbo) for this epoch
        epoch_nll = 0.0

        i_batch = 1

        if epoch >= 25:
            print("here")

        if epoch < wait_until_kl_inc:
            kl_anneal = 0.0

        else:

            print("max kl coefficient::", max_kl)

            kl_anneal = min(
                (20 - 20 * 0.9 ** (((epoch - wait_until_kl_inc) * 1.0)), max_kl)
            )

        print("epoch::", epoch, kl_anneal)

        for id, data_dict in enumerate(data_obj["train_dataloader"]):

            batch_dict = data_dict

            print(id)

            if id >= 1:
                print("here")

            loss_AE = model.train_AE(
                batch_dict["observed_data"],
                batch_dict["origin_observed_data"],
                batch_dict["observed_mask"],
                batch_dict["observed_origin_mask"],
                batch_dict["observed_new_mask"],
                batch_dict["observed_lens"],
                kl_anneal,
                batch_dict["data_to_predict"],
                batch_dict["origin_data_to_predict"],
                batch_dict["mask_predicted_data"],
                batch_dict["origin_mask_predicted_data"],
                batch_dict["new_mask_predicted_data"],
                batch_dict["lens_to_predict"],
                is_GPU,
                device,
                batch_dict["time_stamps"],
                batch_dict["time_stamps_to_predict"],
            )

            epoch_nll += loss_AE["train_loss_AE"]
            i_batch = i_batch + 1

            itr += 1

        if epoch % test_period == 0:
            print("test loss::")

            valid_rmse = validate(data_obj, model, is_GPU, device)

            all_valid_rmse_list.append(valid_rmse)

            test_res = test(data_obj, model, is_GPU, device)

            all_test_res.append(test_res)

    print("final test loss::")

    print_test_res(all_valid_rmse_list, all_test_res, args)

    if not os.path.exists(data_dir + output_dir):
        os.makedirs(data_dir + output_dir)
    torch.save(model, data_dir + output_dir + "model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--model", type=str, default="DGM2_L", help="model name")
    parser.add_argument(
        "--dataset", type=str, default="climate_NY", help="name of dataset"
    )
    parser.add_argument(
        "-std", type=float, default=0.5, help="std of the initial phi table"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=40)
    parser.add_argument("--wait_epoch", type=int, default=0)
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("--use_gate", action="store_true", help="use gate in the model")
    parser.add_argument("--GPU", action="store_true", help="GPU flag")
    parser.add_argument("-G", "--GPUID", type=int, help="GPU ID")
    parser.add_argument(
        "--cluster_num", type=int, default=20, help="number of clusters"
    )
    parser.add_argument("--max_kl", type=float, default=1.0, help="max kl coefficient")
    parser.add_argument(
        "--gaussian", type=float, default=0.000001, help="gaussian coefficient"
    )

    args = parser.parse_args()

    os.makedirs(
        "./output/{args.model}/{args.expname}/{args.dataset}/models", exist_ok=True
    )
    main(args)

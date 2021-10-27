import pandas as pd
import pickle

from collections import Counter


def get_ovo_classication(
    y_hat_history, y_test, config, store_results=True, prefix=None
):

    y_hat_history = pd.DataFrame(y_hat_history)
    y_class_multi = pd.Series(dtype=str)
    row_prediction_history = {}
    # Calculate overall performance on test said OneVsOne style
    for test_idx, test_row in y_test.iteritems():
        # Which models predicted this row
        model_ind = y_hat_history["X_test_ind"].isin(
            [item for item in y_hat_history["X_test_ind"] if test_idx in item]
        )
        if model_ind.any():
            # Which of the models predictions corresponds to the specific rows
            row_predictions = {"label": [], "y_hat": []}
            for model_index, model_row in y_hat_history[model_ind].iterrows():
                prediction_idx = list(model_row["X_test_ind"]).index(test_idx)
                y_hat_tmp = model_row["y_hat"][prediction_idx]
                mx_idx = list(y_hat_tmp).index(max(y_hat_tmp))
                label = model_row["target_pair"][mx_idx]
                # .numpy to convert from tensor to value
                mx_yhat = max(y_hat_tmp).numpy()
                row_predictions["label"].append(label)
                row_predictions["y_hat"].append(mx_yhat)
            # Index of row predictions having highest prediction

            value_counts = Counter(row_predictions["label"])
            # Get most common label from comparisons
            final_label = value_counts.most_common()[0][0]
            # if value_counts.most_common()[0][1] == 1:
            #     # If all occur the same amount of times, choose most confident occurance (still not really a good way to do it though)
            #     best_idx = list(row_predictions["y_hat"]).index(
            #         max(row_predictions["y_hat"])
            #     )
            #     final_label = row_predictions["label"][best_idx]

            y_class_multi.loc[test_idx] = final_label
            row_prediction_history[test_idx] = row_predictions
    if store_results is True:
        result_path = f"{config.get('path')}/{config.get('ID')}"
        with open(f"{result_path}/{prefix}-row-prediction-history.pkl", "wb+") as f:
            pickle.dump(row_prediction_history, f, pickle.HIGHEST_PROTOCOL)
        # how to load again
        # with open(f"{prefix}-row-prediction-history.pkl", 'rb') as f:
        #     pickle.load(f)
        y_class_multi.to_csv(f"{result_path}/{prefix}-yclass-multi.csv")
    return y_class_multi, row_prediction_history

def get_ova_classication(
    y_hat_history, y_test, config, store_results=True, prefix=None
):

    y_hat_history = pd.DataFrame(y_hat_history)
    y_class_multi = pd.Series(dtype=str)
    row_prediction_history = {}
    # Calculate overall performance on test said OneVsOne style
    for test_idx, test_row in y_test.iteritems():
        # Which models predicted this row
        model_ind = y_hat_history["X_test_ind"].isin(
            [item for item in y_hat_history["X_test_ind"] if test_idx in item]
        )
        if model_ind.any():
            # Which of the models predictions corresponds to the specific rows
            row_predictions = {"label": [], "y_hat": []}
            for model_index, model_row in y_hat_history[model_ind].iterrows():
                prediction_idx = list(model_row["X_test_ind"]).index(test_idx)
                y_hat_tmp = model_row["y_hat"][prediction_idx]
                yhat = y_hat_tmp[1].numpy()
                label = model_row["target_pair"][1]
                row_predictions["label"].append(label)
                row_predictions["y_hat"].append(yhat)
            # Index of row predictions having highest prediction
            import numpy as np
            # value_counts = Counter(row_predictions["label"])
            # Get most common label from comparisons
            final_label = np.array(row_predictions["label"])[np.array(row_predictions["y_hat"])==max(row_predictions["y_hat"])]
            # if value_counts.most_common()[0][1] == 1:
            #     # If all occur the same amount of times, choose most confident occurance (still not really a good way to do it though)
            #     best_idx = list(row_predictions["y_hat"]).index(
            #         max(row_predictions["y_hat"])
            #     )
            #     final_label = row_predictions["label"][best_idx]

            y_class_multi.loc[test_idx] = final_label
            row_prediction_history[test_idx] = row_predictions
    if store_results is True:
        result_path = f"{config.get('path')}/{config.get('ID')}"
        with open(f"{result_path}/{prefix}-row-prediction-history.pkl", "wb+") as f:
            pickle.dump(row_prediction_history, f, pickle.HIGHEST_PROTOCOL)
        # how to load again
        # with open(f"{prefix}-row-prediction-history.pkl", 'rb') as f:
        #     pickle.load(f)
        y_class_multi.to_csv(f"{result_path}/{prefix}-yclass-multi.csv")
    return y_class_multi, row_prediction_history
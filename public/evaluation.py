import csv

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


class Evaluation:
    def __init__(self, save_dir, save_alldata):
        self.save_dir = save_dir
        self.save_alldata = save_alldata

    def measure_accessibility_prediction(self, y_label, ground_label):
        mae_avg = []
        mse_avg = []
        cosine_sim_avg = []
        pearson_corr_avg = []

        for i in range(len(y_label)):
            single_y_label = y_label[i]
            single_ground_label = ground_label[i]
            try:
                mae = mean_absolute_error(y_true=single_ground_label, y_pred=single_y_label)
                mae_avg.append(mae)

                mse = mean_squared_error(y_true=single_ground_label, y_pred=single_y_label)
                mse_avg.append(mse)

                cosine_sim = cosine_similarity(single_ground_label, single_y_label)
                cosine_sim_avg.append(np.mean(cosine_sim))

                pearson_corr, _ = pearsonr(single_ground_label.flatten(), single_y_label.flatten())
                pearson_corr_avg.append(pearson_corr)
            except ValueError:
                pass

        final_mae = np.mean(np.array(mae_avg))
        final_mse = np.mean(np.array(mse_avg))
        final_cosine_sim = np.mean(np.array(cosine_sim_avg))
        final_pearson_corr = np.mean(np.array(pearson_corr_avg))

        res_save = pd.DataFrame({'mae': mae_avg, 'mse': mse_avg, 'cosine_sim': cosine_sim_avg, 'pearson_corr': pearson_corr_avg})
        res_save.to_csv(self.save_alldata, index=False)

        with open(self.save_dir, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows([['mae'], [final_mae], ['mse'], [final_mse], ['cosine_sim'], [final_cosine_sim], ['pearson_corr'], [final_pearson_corr]])


if __name__ == "__main__":
    # Create fake data
    y_label = [np.random.rand(2, 19), np.random.rand(2, 19), np.random.rand(2, 19)]
    ground_label = [np.random.rand(2, 19), np.random.rand(2, 19), np.random.rand(2, 19)]

    # Instantiate the Evaluation class
    evaluation = Evaluation(save_dir="../performance/index/infer_performance.index",
                            save_alldata="../performance/index_batch/infer_performance.index")

    # Call the measure_accessibility_prediction method
    evaluation.measure_accessibility_prediction(y_label, ground_label)


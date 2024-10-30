from metrics.metrics import *
from utils.general_utils import separate_attributes_for_scoring, separate_and_rescale_attributes_for_scoring

'''import os
cur_dir = os.getcwd()
ckpt_dir = 'checkpoints'
dir = os.path.join(cur_dir, ckpt_dir)
os.makedirs(dir, exist_ok=True)
'''

# This Evaluator class provides a structured way to evaluate model performance on multiple attributes of test prompts. 
# Here are its key functionalities:

# Performance Metrics: It computes and tracks multiple performance metrics (Pearson, Spearman, Kappa, RMSE) that 
# are essential in evaluating the model's predictions against the true scores.

# Checkpointing Best Models: Although the saving mechanism for the best model is commented out, it indicates 
# that the evaluator is designed to save the model state when the performance on the validation set improves.

# Detailed Reporting: The methods for printing information provide comprehensive insights into model performance, 
# making it easier to analyze and debug the evaluation process.

class Evaluator():

    def __init__(self, test_prompt_id, X_dev_prompt_ids, X_test_prompt_ids, dev_features_list, test_features_list,
                 Y_dev, Y_test, seed):
        self.test_prompt_id = test_prompt_id
        self.dev_features_list = dev_features_list
        self.test_features_list = test_features_list
        self.X_dev_prompt_ids, self.X_test_prompt_ids = X_dev_prompt_ids, X_test_prompt_ids
        self.Y_dev, self.Y_test = Y_dev, Y_test
        self.Y_dev_upscale = Y_dev * 100
        self.Y_dev_org = separate_attributes_for_scoring(self.Y_dev_upscale, self.X_dev_prompt_ids)
        self.Y_test_org = separate_and_rescale_attributes_for_scoring(Y_test, self.X_test_prompt_ids)
        self.best_dev_kappa_mean = -1
        self.best_test_kappa_mean = -1
        self.best_dev_kappa_set = {}
        self.best_test_kappa_set = {}
        self.seed = seed
        
    # calc_pearson: Calculates the Pearson correlation coefficient between predictions and original values.
    # calc_spearman: Calculates the Spearman rank correlation coefficient.
    # calc_kappa: Calculates the Cohen's kappa score, a measure of agreement between two raters.
    # calc_rmse: Computes the root mean square error (RMSE) between predictions and original values.

    @staticmethod
    def calc_pearson(pred, original):
        pr = pearson(pred, original)
        return pr

    @staticmethod
    def calc_spearman(pred, original):
        spr = spearman(pred, original)
        return spr

    @staticmethod
    def calc_kappa(pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    @staticmethod
    def calc_rmse(pred, original):
        rmse = root_mean_square_error(original, pred)
        return rmse

    # Makes predictions using the provided model for both development and test datasets.
    # Rescales predictions and organizes them for scoring.
    # Calculates various metrics (Pearson, Spearman, Kappa) for both development and test datasets.
    # Compares current kappa scores with the best recorded values, updating if current scores are better.
    # Optionally prints evaluation results.
    def evaluate(self, model, epoch, print_info=True):
        self.current_epoch = epoch

        dev_pred = model.predict(self.dev_features_list, batch_size=32)
        test_pred = model.predict(self.test_features_list, batch_size=32)

        dev_pred_int = dev_pred * 100
        dev_pred_dict = separate_attributes_for_scoring(dev_pred_int, self.X_dev_prompt_ids)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred, self.X_test_prompt_ids)

        pearson_dev = {key: self.calc_pearson(dev_pred_dict[key], self.Y_dev_org[key]) for key in
                       dev_pred_dict.keys()}
        pearson_test = {key: self.calc_pearson(test_pred_dict[key], self.Y_test_org[key]) for key in
                        test_pred_dict.keys()}

        spearman_dev = {key: self.calc_spearman(dev_pred_dict[key], self.Y_dev_org[key]) for key in
                        dev_pred_dict.keys()}
        spearman_test = {key: self.calc_spearman(test_pred_dict[key], self.Y_test_org[key]) for key in
                        test_pred_dict.keys()}

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], self.Y_dev_org[key]) for key in
                        dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], self.Y_test_org[key]) for key in
                         test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            '''
            file_path = os.path.join(dir, "checkpoint_best"+str(self.test_prompt_id)+"_"+str(self.seed)+".ckpt")
            model.save_weights(file_path) # save the best model
            print("Save best model to ", file_path)
            '''
        if print_info:
            self.print_info()

    # Outputs the current epoch, average kappa scores for development and test datasets, and the best recorded scores
    def print_info(self):
        print('CURRENT EPOCH: {}'.format(self.current_epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(att, round(self.kappa_dev[att], 3)))
        print(
            '------------------------')
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(att, round(self.kappa_test[att], 3)))
        print(
            '------------------------')
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print(
            '--------------------------------------------------------------------------------------------------------------------------')

    # Outputs the best epoch, average kappa scores for development and test datasets, and the best recorded scores
    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print(
            '--------------------------------------------------------------------------------------------------------------------------')

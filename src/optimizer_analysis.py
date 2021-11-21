import numpy as np
import optuna
import joblib
import matplotlib.pyplot as plt

study = joblib.load("../analysis_results/Opt_test_OPTUNA/model/study.pkl")

bayes = np.loadtxt('../analysis_results/Opt_test_BAYES/opt_history_target.txt')

trials = study.trials
trials_array = np.array([t.values[0] for t in trials])
trials_array = np.extract(trials_array >= 0.993, trials_array)

plt.plot(trials_array, 'o', label = 'Optuna Trials', alpha = 0.2)
plt.plot(np.maximum.accumulate(trials_array), label = 'Optuna Best')
plt.plot(np.ones(len(trials_array)) * 0.994424, label = 'Default XGBOOST')
plt.plot(np.ones(len(trials_array)) * 0.99313, label = 'PbPb hyperparameters')
plt.plot(bayes, 'o', label = 'Bayes trials', alpha = 0.2)
plt.plot(np.maximum.accumulate(bayes), label = 'Bayes best')

plt.title('Comparison of hyperparameters optimizers')
plt.xlabel('Iteration')
plt.ylabel('ROC AUC')
plt.legend()

plt.savefig('../opt_comp.png', dpi = 100, facecolor = 'white')
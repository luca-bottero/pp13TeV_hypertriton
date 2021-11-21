import numpy as np
import optuna
import joblib
import matplotlib.pyplot as plt

study = joblib.load("../analysis_results/Opt_test_OPTUNA/model/study.pkl")
print("Best trial until now:")
print(" Value: ", study.best_trial.value)

trials = study.trials
#plt.plot(trials.)
trials_array = np.array([t.values[0] for t in trials])
trials_array = np.extract(trials_array >= 0.993, trials_array)

plt.plot(trials_array, 'o')
plt.plot(np.maximum.accumulate(trials_array))
plt.plot(np.ones(len(trials_array)) * 0.994424, label = 'Default XGBOOST')

plt.x_label('Iteration')
plt.legend()

plt.savefig('../opt_comp.png')
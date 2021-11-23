import numpy as np
import optuna
import joblib
import matplotlib.pyplot as plt
import pickle
from hipe4ml.model_handler import ModelHandler

##################################################################################

# PERFORMANCE AS A FUNCTION OF ITERATION 

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
plt.close()

##################################################################################

# BEST HYPERPARAMETERS FOR EACH METHOD

names = ['Opt_test_OPTUNA', 'Opt_test_BAYES', 'Opt_test_DEFAULT', 'Opt_test_PbPb']

if False:
    for name in names:
        model_hdl = ModelHandler()
        model_hdl.load_model_handler('../analysis_results/' + name + '/model/model_hdl')

        print(name)
        print(model_hdl.get_model_params())
        print('\n---------------\n')

##################################################################################

# PLOT SUPERIMPOSED ROC

plt.close()
objects = []

for n in names:
    with (open('../analysis_results/' + n + '/images/training/ROC_AUC_train_test.pickle', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

ax = []

for i,f in enumerate(objects):
    objects[i].gca().set_facecolor('none')
    objects[i].set_facecolor('none')
    objects[i].savefig(str(i) + '.png')
    ax.append(objects[i])
    objects[i].gca().plot()

plt.show()
for i in range(len(ax)):
    ax[i]
    plt.plot()

plt.savefig('roc.png')
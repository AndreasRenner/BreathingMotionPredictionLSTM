# Authors: Ing. Ingo Gulyas, MSc, MSc; DI Dr. Andreas Renner
# Date: 19.05.2022
# Medical University of Vienna / AKH Wien
# Department of Radiation Oncology

import logging
import os

from train_gridsearch import *
from train_crossval import *
from core_functions import *


# disable warning messages
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TODO:
# evaluate error regarding phi, r and baseline for optimal hyper parameters
# show potential improvements regarding the int/ext correlation model (setup time and redundant x-ray images)

# SYSTEM INFO:
#   Python 3.8.7
#   Tensorflow 2.4.0
#   Cuda 11.2

# KALMAN INFO: RMSE @ 500 ms prediction horizon -> 1.33 mm (22 % @ 6 mm)
#              RMSE without prediction -> 1.87 mm

########################################################################################################################
# global settings
########################################################################################################################

BASE_DIR = './VALIDATION-PHASE'

# Input here the respectiv IDs - or a function which inputs IDs - of your datasets for training/testing/validation
training_ids = []
testing_ids = []
validation_ids = []

settings = {	'training_ids': training_ids,
				'testing_ids': testing_ids,
				'crossval_splits': 0,
				'Ts': 0.05,
				'horizon_sec': 0.5,
				'n_horizon': 10,
				'n_prediction': 1,
				'use_data_shuffle': True,
				'use_randseed_reset': True,
				'epochs_max': 50,
				'save_best_test_model' : True}


########################################################################################################################
# main code
########################################################################################################################



if not (os.path.exists('./tmp')):
	os.makedirs('./tmp')
if not (os.path.exists(BASE_DIR)):
	os.makedirs(BASE_DIR)



# INFO: create_search_grid(UNITS, LRs, BS, N_TIMELAG, SIGNALS_IN, SIGNALS_OUT, FILEPATH, DYN_SCALER)
params_lst = create_search_grid([40],   											# untis
								[0.01],								# learning-rate
								[512],  							# batch-size
								[160],								# n_timelag
								[[0.0, 0.0, 0.0]],   						# dropout
								[['amplitude1']],						# input signals
								[['phi1', 'r1', 'baseline1']], 					# output signals
								['./Data/PhaseCombinedBreathingDataRegular_v13_v2.csv'],	# breathing data filepath
								['m1to1'])							# dynamic scaler type

model_training_manual_gridsearch(params_lst, settings, 'VALIDATION-PHASE')

exit(0)

y_true, y_pred, phi_r_bl_true, phi_r_bl_pred, model, config = model_predict_dynamic_scaler('Models/20220523-231257-LSTM-VAL-GS_BEST-0171_scaler-m1to1/HyperParamConfig0/BestTestModel-CV00', './Data/PhaseCombinedBreathingDataRegular_v13_v2.csv', validation_ids)
phi_r_bl_true = phi_r_bl_true.squeeze()
phi_r_bl_pred = phi_r_bl_pred.squeeze()

plot_breathing_signal_and_rmse(y_true, y_pred, root_mean_squared_error(y_true, y_pred), config['settings']['Ts'], '')
#phi_true = phi_r_bl_true[:, 0]
#phi_pred = phi_r_bl_pred[:, 0]
#plot_breathing_signal_and_rmse(phi_true, phi_pred, root_mean_squared_error(phi_true, phi_pred), config['settings']['Ts'], '')



d = {'y_true': y_true, 'phi_true': phi_r_bl_true[:, 0], 'r_true': phi_r_bl_true[:, 1], 'bl_true': phi_r_bl_true[:, 2],
	 'y_pred': y_pred, 'phi_pred': phi_r_bl_pred[:, 0], 'r_pred': phi_r_bl_pred[:, 1], 'bl_pred': phi_r_bl_pred[:, 2]}
df = pd.DataFrame(data=d)
df.to_csv('./tmp/patients-testing.csv', index=False)



########################################################################################################################
# Crossvalidation training
#model_training_crossval_gridsearch(params_lst, settings, BASE_DIR)

# Ensemble prediciton
#y_true, phi_true, r_true, bl_true, y_pred, phi_pred, r_pred, bl_pred = ensemble_predict('./Models/20220506-024831-LSTM-VAL-CV-regularV11-combined(alternating)/HyperParamConfig0', 'BestTestModel*', './Data/PhaseCombinedBreathingDataRegular_v11.csv', testing_ids)
y_true, phi_true, r_true, bl_true, y_pred, phi_pred, r_pred, bl_pred = ensemble_predict('./Models/20220507-000840-LSTM-VAL-CV-regularV11-combined(sameAmps)/HyperParamConfig0', 'BestTestModel*', './Data/PhaseCombinedBreathingDataRegular_v13_v2.csv', validation_ids)

y_pred_mean = np.mean(y_pred, 0)
y_pred_median = np.median(y_pred, 0)

rmse_mean = root_mean_squared_error(y_true, y_pred_mean)
rmse_median = root_mean_squared_error(y_true, y_pred_median)

print('rmse_mean: %.3f' % rmse_mean)
print('rmse_median: %.3f' % rmse_median)

#plt.plot(y_pred.T)
#plt.plot(y_true, c="black")
#plt.plot(y_pred_mean-y_true, c="red")

y_pred_sigma = np.std(y_pred, 0)

plt.scatter(y_pred_sigma, y_true-y_pred_mean, c="blue")
plt.ylabel('prediction error [mm]')
plt.xlabel('sigma')
plt.show()

plt.scatter(y_pred_sigma, y_true-y_pred_median, c="blue")
plt.ylabel('prediction error [mm]')
plt.xlabel('sigma')
plt.show()

plt.hist(y_true-y_pred_median, bins=40)
plt.show()


exit(0)


#######################################################################################################################
# gridsearch training
#model_training_manual_gridsearch(params_lst, settings, BASE_DIR)

# sigle prediction
y_true, y_pred, phi_r_bl_true, phi_r_bl_pred, model, config = \
	model_predict('Models/20220503-232033-LSTM-VAL-MANUAL-regularV10/HyperParamConfig1/BestTestModel',
				  './Data/PhaseBreathingDataRegular_v10.csv', testing_ids)

plot_breathing_signal_and_rmse(y_true, y_pred, root_mean_squared_error(y_true, y_pred), config['settings']['Ts'], '')
#plot_breathing_signal_and_rmse(phi_r_bl_true, phi_r_bl_pred, -999, config['settings']['Ts'], '')

exit(0)







########################################################################################################################
# deprecated code
########################################################################################################################
''' # TEST DATA-GENERATOR
data_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]).T

group_len = [10]
group_sel = [0]
n_time_lag = 2
n_horizon = 2
n_prediction = 1
n_stride = 3

x,y = data_generator(data_array, group_len, group_sel, n_time_lag, n_horizon, n_prediction, n_stride)

from pprint import pprint
pprint(x)
pprint(y)
'''

''' # TEST MATPLOTLIB
		breathingdata, breathingdata_group_len = import_breathing_data(BREATHINGDATA_FILEPATH, training_ids, selected_signals)
		train_data_x, train_data_y = data_generator(breathingdata, breathingdata_group_len, [1], 20, n_horizon,
													n_prediction)

		for i in range(0, 600):
			print(i)
			# plot prediction result and error
			# plt.rcParams.update({'font.size': 20})
			x_ticks = np.asarray(range(0, len(train_data_y))) * Ts
			fig, axes = plt.subplots(2, 1)
			fig.tight_layout(pad=3.0)

			axes[0].plot(x_ticks, train_data_y.flat, 'b', linewidth=0.5, label='orig')
			axes[0].plot(x_ticks, train_data_y.flat, 'g', linewidth=0.5, label='pred')
			axes[0].legend(loc='upper left')
			axes[0].set_xlabel('time [s]')
			axes[0].set_ylabel('surrogate position [mm]')
			axes[0].grid()
			axes[1].plot(x_ticks, train_data_y.flat, 'r', linewidth=0.5, label='error')
			# axes[0].set_title('data prediction')

			# axes[1].legend(loc='upper left')
			axes[1].set_xlabel('time [s]')
			axes[1].set_ylabel('error [mm]')
			axes[1].grid()
			axes[1].set_title('prediction error (RMSE=%.3f)' % 0)
			# plt.get_current_fig_manager().full_screen_toggle()

			# plt.show()
			#fig.savefig(os.path.join('./tmp/test', 'TestPred_CVstep%d-Epoch%d.svg' % (0, i)))
			fig.clear()
			plt.close(fig)

'''


'''
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))

exit(0)
'''

'''
# create filepath for saving model
def create_model_path(dim_in, params, rmse_train):
	dt_now = datetime.now()
	dt_strg = dt_now.strftime('%Y%m%d-%H%M%S')
	path = os.path.join(MODELS_SUBDIR, dt_strg + '_DIM%d_TL%d_BS%d_EP%d_RMSE_CV%dum' % (dim_in, params['time_lag'], params['batch_size'], params['epochs'], round(rmse_train*1000)))
	if (os.path.exists(path)):
		print_err('model directory \'%s\' already exists!' % (path))
		exit(-1)
	else:
		os.makedirs(path)
	return path
'''

'''
# INFO: selected ids will be created due to filter params (id/src/prep)
def import_breathing_data(filepath, selected_ids, selected_signals):
	# open data file and read header
	with open(filepath) as f:
		first_line = f.readline()

	# check for comment character
	if (first_line[0] != '#'):
		print_err('file header ''#'' not found in file %s!' % filepath)

	# convert json header to dataframe
	json_hdr = first_line[1:]
	hdr_data = json.loads(json_hdr)
	hdr_data = hdr_data['info_header']
	hdr_tab = pd.DataFrame(hdr_data)

	# filter breathing data ids by parameter(s)
	#hdr_tab.loc[(tab['id']%1000) == 0]
	#hdr_tab.loc[tab['prep'] == 'orig']
	#hdr_tab[hdr_tab['src'].str.contains('Elke')]

	# import data
	selected_cols = list(selected_signals)
	selected_cols.insert(0, 'id')
	data = pd.read_csv(filepath, usecols=selected_cols, skiprows=1, sep=',')
	data = data[data['id'].isin(selected_ids)]
	group_len = data.groupby('id').size().to_numpy()
	data = data.to_numpy()

	return data, group_len
'''

''' 
# INFO: create list with all permutations of given parameters 
def init_gridsearch_param_list():
	value_lst = list(itertools.product(*param_grid.values()))
	key_lst = param_grid.keys()

	param_lst = []
	for vals in value_lst:
		param_lst.append(dict(zip(key_lst, vals)))

	return param_lst'''

# Author: Ing. Ingo Gulyas, MSc, MSc
# Date: 04.05.2022
# Medical University of Vienna / AKH Wien
# Department of Radiation Oncology

from datetime import datetime
import os.path
import json
from shutil import copyfile
import itertools

from core_functions import *

def create_search_grid(units, l_rate, batch_size, time_lag, dropout, selected_input_signals, selected_output_signals, breathing_data_filepath, dyn_scaler):

	params_lst = []
	for r in itertools.product(units, time_lag, batch_size, l_rate, dropout, selected_input_signals, selected_output_signals, breathing_data_filepath, dyn_scaler):
		params_cfg = dict()
		params_cfg['units'] = r[0]
		params_cfg['time_lag'] = r[1]
		params_cfg['batch_size'] = r[2]
		params_cfg['l_rate'] = r[3]
		params_cfg['dropout'] = r[4]
		params_cfg['selected_input_signals'] = r[5]
		params_cfg['selected_output_signals'] = r[6]
		params_cfg['breathing_data_filepath'] = r[7]
		params_cfg['dyn_scaler'] = r[8]
		params_lst.append(dict(params_cfg))

	return params_lst

def model_training_manual_gridsearch(params_lst, settings, base_directory):
	dt_start = datetime.now()
	dt_start_strg = dt_start.strftime('%Y%m%d-%H%M%S')
	working_dir = os.path.join(base_directory, '%s-LSTM-VAL-GS' % dt_start_strg)
	os.makedirs(working_dir)

	# save settings, breahting data and main script
	#fp = os.path.join(working_dir, 'TrainingSettings.json')
	#with open(fp, 'w') as json_file:
	#	json.dump(settings, json_file)

	copyfile('main.py', os.path.join(working_dir, 'main.py'))
	copyfile('train_gridsearch.py', os.path.join(working_dir, 'train_gridsearch.py'))
	copyfile('core_functions.py', os.path.join(working_dir, 'core_functions.py'))

	# for all hyper-parameter settings
	for i in range(0, len(params_lst)):
		dt_trainig_start = datetime.now()

		params = params_lst[i]

		breathing_data_filepath = params['breathing_data_filepath']
		selected_input_signals = params['selected_input_signals']
		selected_output_signals = params['selected_output_signals']

		breathingdata_train_in, breathingdata_train_out, breathingdata_train_group_len = import_data(breathing_data_filepath, settings['training_ids'], selected_input_signals, selected_output_signals)
		breathingdata_test_in, breathingdata_test_out, breathingdata_val_group_len = import_data(breathing_data_filepath, settings['testing_ids'], selected_input_signals, selected_output_signals)

		params.update({'breathing_data_filename': get_filename(breathing_data_filepath)})

		params.update({'scaler_input_min': 0})
		params.update({'scaler_input_max': 0})
		params.update({'scaler_output_min': 0})
		params.update({'scaler_output_max': 0})

		print('TRAINING STEP %d/%d -> %s' % (i + 1, len(params_lst), params))

		working_dir_hyperparams = os.path.join(working_dir, 'HyperParamConfig%d' % i)
		os.makedirs(working_dir_hyperparams)

		breathingdata_filepath_local = os.path.join(working_dir, get_filename(breathing_data_filepath))
		if not os.path.isfile(breathingdata_filepath_local):
			copyfile(breathing_data_filepath, breathingdata_filepath_local)

		logfilepath = logfile_create(working_dir_hyperparams, params)

		# training
		train_ix = list(range(0, len(settings['training_ids'])))
		test_ix = list(range(0, len(settings['testing_ids'])))

		# prepare breathing data (leave one out)
		# INFO:
		#  1 input: train_x.shape = (n,40,1), train_y.shape = (n,1,1)
		#  2 inputs: train_x.shape = (n,40,2), train_y.shape = (n,1,1)
		train_data_x, train_data_y = data_generator(breathingdata_train_in, breathingdata_train_out, breathingdata_train_group_len, train_ix,
													params['time_lag'], settings['n_horizon'], settings['n_prediction'], 1)
		test_data_x, test_data_y = data_generator(breathingdata_test_in, breathingdata_test_out, breathingdata_val_group_len, test_ix,
													params['time_lag'], settings['n_horizon'], settings['n_prediction'], 1)

		# normalize training and test data
		if params['dyn_scaler'] == '0to1':
			train_data_x_norm, train_data_y_norm, train_data_y_min, train_data_y_max = scaler_transform_0to1(train_data_x, train_data_y)
			test_data_x_norm, test_data_y_norm, test_data_y_min, test_data_y_max = scaler_transform_0to1(test_data_x, test_data_y)
		elif params['dyn_scaler'] == 'm1to1':
			train_data_x_norm, train_data_y_norm, train_data_y_min, train_data_y_max = scaler_transform_m1to1(train_data_x, train_data_y)
			test_data_x_norm, test_data_y_norm, test_data_y_min, test_data_y_max = scaler_transform_m1to1(test_data_x, test_data_y)

		# shuffle training data if required
		if settings['use_data_shuffle'] != 0:
			train_data_x_norm_mixed, train_data_y_norm_mixed = data_shuffle(train_data_x_norm, train_data_y_norm, 123456)
		else:
			train_data_x_norm_mixed = train_data_x_norm
			train_data_y_norm_mixed = train_data_y_norm

		# ensure size of training data matches an integer multiple of batch size
		if train_data_x.shape[0] % params['batch_size'] != 0:
			train_data_x_norm_mixed_bs = train_data_x_norm_mixed[:-(train_data_x_norm_mixed.shape[0] % params['batch_size'])]
			train_data_y_norm_mixed_bs = train_data_y_norm_mixed[:-(train_data_y_norm_mixed.shape[0] % params['batch_size'])]

		# init tensorflow random seed for reproducibility
		if settings['use_randseed_reset'] != 0:
			np.random.seed(42)
			tf.random.set_seed(17)

		model = create_model(train_data_x_norm, train_data_y_norm,  units=params['units'], batch_size=params['batch_size'],
							 l_rate=params['l_rate'], dropout=params['dropout'], epochs=settings['epochs_max'])

		my_callback = CustomCallback(train_data_x_norm, train_data_y, test_data_x_norm, test_data_y_norm, params['scaler_input_min'],
									params['scaler_input_max'], np.asarray(params['scaler_output_min']),
									np.asarray(params['scaler_output_max']), settings['Ts'], working_dir_hyperparams,
									logfilepath, 0, False, False, test_data_y_min, test_data_y_max, train_data_y_min,
									train_data_y_max, params['dyn_scaler'], settings['save_best_test_model'])

		if settings['use_randseed_reset'] != 0:
			np.random.seed(42)
			tf.random.set_seed(17)

		model.fit(train_data_x_norm_mixed_bs, train_data_y_norm_mixed_bs, shuffle=bool(settings['use_data_shuffle']), epochs=settings['epochs_max'],
				  batch_size=params['batch_size'], verbose=2, use_multiprocessing=True, callbacks=[my_callback])

		dt_training_end = datetime.now()

		params.update({'training-time': (dt_training_end-dt_trainig_start).total_seconds()})

		#  collect statistics: settings, params and losses
		all_stats = {}
		all_stats['loss_train'] = my_callback.loss_training
		all_stats['loss_test'] = my_callback.loss_testing
		all_stats['params'] = params
		all_stats['settings'] = settings

		my_callback.loss_training = []
		my_callback.loss_testing = []

		# store model results in model directory (cv-step)
		if os.path.isfile(os.path.join(working_dir_hyperparams, 'BestTestModel-CV00')):
			fd = open(os.path.join(working_dir_hyperparams, 'BestTestModel-CV00', 'stats.txt'), 'w')
			fd.write(json.dumps(all_stats))
			fd.write('\n')
			fd.close()

		# store collected results in working directory
		fd = open(os.path.join(working_dir, 'stats_tmp.txt'), 'a')
		fd.write(json.dumps(all_stats))
		fd.write('\n')
		fd.close()

		print('TRAINING step done - TRAIN:', settings['training_ids'], "-> VAL:", settings['testing_ids'])

		# loss = fit_stats.history['loss']
		# test_loss = fit_stats.history['val_loss']
		#x_ticks = np.asarray(range(1, len(loss)+1))
		#fig, axes = plt.subplots(1, 1)
		#fig.tight_layout(pad=3.0)
		#axes.plot(x_ticks, loss, 'b', linewidth=0.5, label='training')
		#axes.plot(x_ticks, test_loss, 'g', linewidth=0.5, label='test')
		#axes.legend(loc='upper right')
		#axes.set_xlabel('epoch')
		#axes.set_ylabel('loss')
		#axes.grid()
		#plt.show()
		#fig.savefig(os.path.join(working_dir_hyperparams, 'LossResult.svg'))


	# finalize overall stats file
	fd = open(os.path.join(working_dir, 'stats_tmp.txt'), 'r')
	lines = fd.read().splitlines()
	fd.close()

	fd = open(os.path.join(working_dir, '%s_stats.txt' % (os.path.basename(os.path.normpath(working_dir)))), 'w')
	fd.write('[')
	for i in range (0, len(lines)-1):
		fd.write(lines[i])
		fd.write(',\n')
	fd.write(lines[len(lines)-1])
	fd.write(']')
	fd.close()

	os.remove(os.path.join(working_dir, 'stats_tmp.txt'))

	return
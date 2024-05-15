# Author: Ing. Ingo Gulyas, MSc, MSc
# Date: 04.05.2022
# Medical University of Vienna / AKH Wien
# Department of Radiation Oncology

import numpy as np
import pandas as pd
import os.path
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import keras.backend as K
from tensorflow import keras
import time
import sys
import matplotlib.pyplot as plt
import json
import glob

def print_err(text):
	text = 'ERROR: ' + text
	text = '\033[38;2;{};{};{}m{} \033[38;2;255;255;255m'.format(255, 0, 0, text)
	print(text)


def get_filename(filepath):
	head, tail = os.path.split(filepath)
	return tail


# INFO: select and import data from csv file
def import_data(filepath, selected_ids, selected_input_signals, selected_output_signals):
	selected_cols = list(selected_input_signals)+list(selected_output_signals)
	#selected_cols = ['id', 'amplitude1', 'r1', 'phi1', 'baseline1']
	selected_cols.insert(0, 'id')
	df = pd.read_csv(filepath, usecols=selected_cols, sep=',')
	if(not set(selected_ids).issubset(np.unique(df['id']).tolist())):
		print_err('one ore more ids missing in csv file!')
		exit(-1)
	# filter ids
	df = df[df['id'].isin(selected_ids)]
	# group ids
	df_grouped = df.groupby(['id'])

	group_len = []
	df = df_grouped.get_group(selected_ids[0])
	group_len.append(len(df))
	for ix in range(1, len(selected_ids)):
		group = df_grouped.get_group(selected_ids[ix])
		df = df.append(group, ignore_index=True)
		group_len.append(len(group))

	data_in = df[selected_input_signals].to_numpy()
	data_out = df[selected_output_signals].to_numpy()
	#print(data_in)
	#print(data_out)
	#print(group_len)
	#data_in = df['amplitude1'].to_numpy()
	#data_out = df['phi1', 'r1', 'baseline1'].to_numpy()

	return data_in, data_out, group_len


# INFO: crop segments from data_array as input/output for the lstm network
def data_generator(data_input_array, data_output_array, group_len, group_sel, n_time_lag, n_horizon, n_prediction, n_stride=1):
	X, Y = [], []

	idx = 0
	group_idx = 0
	for len in group_len:
		if(group_idx in group_sel):
			n_samples = len-n_time_lag-n_horizon-n_prediction+2
			for i in range(0, n_samples, n_stride):
				tmp = data_input_array[(range(idx+i, idx+i+n_time_lag)), :]
				if tmp.ndim < 2:
					tmp = np.transpose([tmp])
				X.append(tmp)
				tmp2 = data_output_array[(range(idx+i+n_time_lag+n_horizon-1, idx+i+n_time_lag+n_horizon+n_prediction-1)), :]
				if tmp2.ndim < 2:
					tmp2 = np.transpose([tmp2])
				Y.append(tmp2)
		group_idx = group_idx+1
		idx = idx+len

	X = np.array(X)
	Y = np.array(Y)

	if(np.isnan(np.sum(X))):
		print_err('NaN found in data array X!')
		#exit(-1)

	if(np.isnan(np.sum(Y))):
		print_err('NaN found in data array X!')
		#exit(-1)

	return X, Y


def data_shuffle(X, Y, seed):
	np.random.seed(seed)
	ix_rand = np.random.permutation(len(X))
	Xshuffled = X[ix_rand, :, :]
	Yshuffled = Y[ix_rand, :, :]
	return Xshuffled, Yshuffled


def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_absolute_error(y_true, y_pred):
	return K.mean(K.abs(y_true - y_pred))


def custom_loss_function(y_true, y_pred):
	if len(y_true.shape) == 3 and y_true.shape[2] == 3:
		phi_true = y_true[:, 0, 0]
		r_true = y_true[:, 0, 1]
		bl_true = y_true[:, 0, 2]

		phi_pred = y_pred[:, 0]
		r_pred = y_pred[:, 1]
		bl_pred = y_pred[:, 2]

		#y_true = r_true+r_true*tf.cos(phi_true*np.pi)+bl_true
		#y_pred = r_pred+r_pred*tf.cos(phi_pred*np.pi)+bl_pred

		res = [root_mean_squared_error(phi_true, phi_pred), root_mean_squared_error(r_true, r_pred), root_mean_squared_error(bl_true, bl_pred)]
	else:
		res = root_mean_squared_error(y_true, y_pred)

	#tf.print(res)

	return res


def create_model(data_x, data_y, units=15, batch_size=100, l_rate=0.001, dropout=[0.0, 0.0, 0,0], epochs=1):
	# create and train the LSTM network
	model = Sequential()
	model.add(LSTM(units, return_sequences=True, input_shape=(data_x.shape[1], data_x.shape[2])))
	model.add(Dropout(dropout[0]))
	model.add(LSTM(units, return_sequences=True))
	model.add(Dropout(dropout[1]))
	model.add(LSTM(units))
	model.add(Dropout(dropout[2]))
	#model.add(Dense(data_y.shape[2], activation='tanh'))
	model.add(Dense(data_y.shape[2]))
	opt = keras.optimizers.Adam(learning_rate=l_rate)
	model.compile(loss=custom_loss_function, optimizer=opt)
	return model


def logfile_create(directory, params):
	filepath = os.path.join(directory, 'logging.csv')
	file_object = open(filepath, 'w')
	file_object.write('#%s\n' % params)
	file_object.write('Epoch;MAE_train;MAE_test;RMSE_train;RMSE_test\n')
	file_object.close()

	return filepath


def logfile_append(filepath, epoch, mae_train, mae_test, rmse_train, rmse_test):
	file_object = open(filepath, 'a')
	file_object.write('%d;%.6f;%.6f;%.6f;%.6f\n' % (epoch, mae_train, mae_test, rmse_train, rmse_test))
	file_object.close()

	return

class CustomCallback(keras.callbacks.Callback):
	def __init__(self, train_x_norm, train_y, test_x_norm, test_y_norm, scaler_min_x, scaler_max_x, scaler_min_y, scaler_max_y, Ts, working_dir, logfilepath, cv_step, save_plots_train, save_plots_test, test_data_y_min, test_data_y_max, train_data_y_min, train_data_y_max, dyn_scaler, save_best_test_model):
		super().__init__()
		if train_y.shape[2] == 3:
			self.mode_phi_r_bl = True
		else:
			self.mode_phi_r_bl = False
		self.scaler_min_x = scaler_min_x
		self.scaler_max_x = scaler_max_x
		self.scaler_min_y = scaler_min_y
		self.scaler_max_y = scaler_max_y
		self.train_x_norm = train_x_norm
		self.train_y = train_y.squeeze()
		# self.train_y = (train_y * (self.scaler_max_y - self.scaler_min_y) + self.scaler_min_y).squeeze()
		if self.mode_phi_r_bl:
			self.train_y = convert_phi_r_bl_to_breathing_signal(self.train_y)
		self.test_x_norm = test_x_norm
		self.test_y_norm = test_y_norm.squeeze()
		#self.test_y = (test_y * (self.scaler_max_y - self.scaler_min_y) + self.scaler_min_y).squeeze()
		if self.mode_phi_r_bl:
			self.test_y = convert_phi_r_bl_to_breathing_signal(self.test_y_norm)
		self.working_dir = working_dir
		self.logfilepath = logfilepath
		self.cv_step = cv_step
		self.t1 = time.time()
		self.best_rmse_test = sys.float_info.max
		self.best_epoch = 0
		self.loss_training = []
		self.loss_testing = []
		self.save_plots_train = save_plots_train
		self.save_plots_test = save_plots_test
		self.best_model = []

		self.test_y_min = test_data_y_min.squeeze()
		self.test_y_max = test_data_y_max.squeeze()
		self.train_y_min = train_data_y_min.squeeze()
		self.train_y_max = train_data_y_max.squeeze()
		self.Ts = Ts
		self.dyn_scaler = dyn_scaler
		self.save_best_test_model = save_best_test_model

	def on_epoch_end(self, epoch, logs=None):

		# predict training data
		#predict_train_y_sc = self.model.predict(self.train_x_norm).squeeze()

		# inverse normalization of training data
		#if self.dyn_scaler == '0to1':
		#	predict_train_y = scaler_inverse_transform_0to1(predict_train_y_sc, self.train_y_min, self.train_y_max)
		#elif self.dyn_scaler == 'm1to1':
		#	predict_train_y = scaler_inverse_transform_m1to1(predict_train_y_sc, self.train_y_min, self.train_y_max)

		#predict_train_y = convert_phi_r_bl_to_breathing_signal(predict_train_y)
		#mae_train = mean_absolute_error(self.train_y, predict_train_y)
		#rmse_train = (root_mean_squared_error(self.train_y, predict_train_y)).numpy()
		#self.loss_training.append(rmse_train)
		mae_train = 0
		rmse_train = 0


		# predict test data
		predict_test_y_sc = self.model.predict(self.test_x_norm).squeeze()

		rmse_test = custom_loss_function(self.test_y_norm, predict_test_y_sc)
		mae_test = 0

		# inverse normalization of test data
		if self.dyn_scaler == '0to1':
			predict_test_y = scaler_inverse_transform_0to1(predict_test_y_sc, self.test_y_min, self.test_y_max)
		elif self.dyn_scaler == 'm1to1':
			predict_test_y = scaler_inverse_transform_m1to1(predict_test_y_sc, self.test_y_min, self.test_y_max)

		predict_test_y = convert_phi_r_bl_to_breathing_signal(predict_test_y)
		#mae_test = mean_absolute_error(self.test_y, predict_test_y)
		#rmse_test = (root_mean_squared_error(self.test_y, predict_test_y)).numpy()
		self.loss_testing.append(rmse_test)

		# get elapsed time and print error stats
		t_now = time.time()
		t_elapsed = t_now-self.t1
		self.t1 = t_now
		print('Epoch: %d   Training MAE: %.6f   Test MAE: %.6f   Training RMSE: %.6f   Test RMSE: %.6f   time: %.3fs' % (epoch+1, mae_train, mae_test, rmse_train, rmse_test, t_elapsed))

		# save model if test rmse is lower than before
		if rmse_test < self.best_rmse_test:
			self.best_rmse_test = rmse_test
			self.best_epoch = epoch + 1
			if self.save_best_test_model:
				print('updating best test-model (RMSE=%.3f)' % self.best_rmse_test)
				self.model.save(os.path.join(self.working_dir, 'BestTestModel-CV%02d' % self.cv_step))

		logfile_append(self.logfilepath, epoch, mae_train, mae_test, rmse_train, rmse_test)

		if self.save_plots_test == True:
			# plot test prediction result and error
			#plt.rcParams.update({'font.size': 20})
			x_ticks = np.asarray(range(0, len(self.test_y))) * self.Ts
			fig, axes = plt.subplots(2, 1)
			fig.tight_layout(pad=3.0)

			axes[0].plot(x_ticks, self.test_y, 'b', linewidth=0.1, label='orig')
			axes[0].plot(x_ticks, predict_test_y, 'g', linewidth=0.1, label='pred')
			axes[0].legend(loc='upper left')
			axes[0].set_xlabel('time [s]')
			axes[0].set_ylabel('input')
			axes[0].grid()

			axes[1].plot(x_ticks, self.test_y - predict_test_y, 'r', linewidth=0.1, label='error')
			# axes[0].set_title('data prediction')
			# axes[1].legend(loc='upper left')
			axes[1].set_xlabel('time [s]')
			axes[1].set_ylabel('error')
			axes[1].grid()
			axes[1].set_title('test prediction error (RMSE=%.3f)' % rmse_test)
			#plt.get_current_fig_manager().full_screen_toggle()

			#plt.show()
			fig.savefig(os.path.join(self.working_dir, 'PredTest_CV%02d-EP%02d.svg' % (self.cv_step, epoch)))
			fig.clear()
			plt.close(fig)

		if self.save_plots_train == True:
			# plot training prediction result and error
			# plt.rcParams.update({'font.size': 20})
			x_ticks = np.asarray(range(0, len(self.train_y))) * self.Ts
			fig, axes = plt.subplots(2, 1)
			fig.tight_layout(pad=3.0)

			axes[0].plot(x_ticks, self.train_y, 'b', linewidth=0.5, label='orig')
			axes[0].plot(x_ticks, predict_train_y, 'g', linewidth=0.5, label='pred')
			axes[0].legend(loc='upper left')
			axes[0].set_xlabel('time [s]')
			axes[0].set_ylabel('input(s) & prediction(s)')
			axes[0].grid()

			axes[1].plot(x_ticks, self.train_y - predict_train_y, 'r', linewidth=0.5, label='error')
			# axes[0].set_title('data prediction')
			# axes[1].legend(loc='upper left')
			axes[1].set_xlabel('time [s]')
			axes[1].set_ylabel('error')
			axes[1].grid()
			axes[1].set_title('training prediction error (RMSE=%.3f)' % rmse_train)
			# plt.get_current_fig_manager().full_screen_toggle()

			#plt.show()
			fig.savefig(os.path.join(self.working_dir, 'PredTraining_CV%02d-EP%02d.svg' % (self.cv_step, epoch)))
			fig.clear()
			plt.close(fig)


def plot_breathing_signal_and_rmse(y_true, y_pred, rmse, lowerlimit, upperlimit, Ts, title, lowerxlimit, upperxlimit, errortitle, lowerelimit, upperelimit):
	# plt.rcParams.update({'font.size': 20})
	x_ticks = np.asarray(range(0, len(y_true))) * Ts
	fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
	fig.tight_layout(pad=2.0)
	#fig.suptitle(title)

	axes[0].plot(x_ticks, y_pred, color='#ff7f0e', linewidth=1, label='prediction', ls='--', marker='2', markersize=4)
	axes[0].plot(x_ticks, y_true, color='#1f77b4', linewidth=1.5, label='original')
	axes[0].legend(loc='upper right')
	#axes[0].set_xlabel('time [s]')
	axes[0].set_ylabel('breathing signal [mm]')
	axes[0].set_title(title)
	if(upperxlimit-lowerxlimit):
		axes[0].set_xlim(lowerxlimit,upperxlimit)
	if(upperlimit-lowerlimit):
		axes[0].set_ylim(lowerlimit,upperlimit)
	axes[0].grid(False)

	axes[1].plot(x_ticks, y_true - y_pred, 'r', linewidth=0.8, label='error')
	axes[1].legend(loc='lower right')
	axes[1].set_xlabel('time [s]')
	axes[1].set_ylabel('error [mm]')
	if(upperxlimit-lowerxlimit):
		axes[1].set_xlim(lowerxlimit,upperxlimit)
	if(upperelimit-lowerelimit):
		axes[1].set_ylim(lowerelimit,upperelimit)
	axes[1].grid(False)	
	if(errortitle):
		axes[1].set_title('prediction error (abs. RMSE=%.2f mm)' % rmse)
	else:
		axes[1].set_title('prediction error')
	plt.show()

	return


def model_load(model_path):
	# import settings
	with open(os.path.join(model_path, 'stats.txt')) as json_file:
		config = json.load(json_file)

	# load model
	model = keras.models.load_model(model_path, custom_objects={'custom_loss_function': custom_loss_function})
	# model.summary()

	return model, config


def model_predict(model_path, breathing_data_filepath, selected_ids):

	model, config = model_load(model_path)

	scaler_input_min = config['params']['scaler_input_min']
	scaler_input_max = config['params']['scaler_input_max']
	scaler_output_min = np.asarray(config['params']['scaler_output_min'])
	scaler_output_max = np.asarray(config['params']['scaler_output_max'])
	selected_input_signals = config['params']['selected_input_signals']
	selected_output_signals = config['params']['selected_output_signals']
	time_lag = config['params']['time_lag']

	Ts = config['settings']['Ts']
	n_horizon = config['settings']['n_horizon']
	n_prediction = config['settings']['n_prediction']

	# import breathing data
	breathingdata_in, breathingdata_out, breathingdata_group_len = import_data(breathing_data_filepath, selected_ids, selected_input_signals, selected_output_signals)

	# scale breathing data
	print('Scaler Input min and max: ')
	print(scaler_input_min, scaler_input_max)
	breathingdata_in_sc = (breathingdata_in - scaler_input_min) / (scaler_input_max - scaler_input_min)
	breathingdata_out_sc = (breathingdata_out - scaler_output_min) / (scaler_output_max - scaler_output_min)

	# prepare data
	data_x_sc, data_y_sc = data_generator(breathingdata_in_sc, breathingdata_out_sc, breathingdata_group_len, range(0, len(selected_ids)), time_lag, n_horizon, n_prediction)

	# predict training data and rescale result
	t_start = time.time()
	phi_r_bl_pred = model.predict(data_x_sc).squeeze() * (scaler_output_max - scaler_output_min) + scaler_output_min
	phi_r_bl_true = data_y_sc.squeeze() * (scaler_output_max - scaler_output_min) + scaler_output_min
	y_pred = convert_phi_r_bl_to_breathing_signal(phi_r_bl_pred)
	y_true = convert_phi_r_bl_to_breathing_signal(phi_r_bl_true)
	t_end = time.time()
	print(f'Runtime of breathing data prediction: {(t_end - t_start) * 1000} ms')

	return y_true, y_pred, phi_r_bl_true, phi_r_bl_pred, model, config



def model_predict_dynamic_scaler(model_path, breathing_data_filepath, selected_ids):

	model, config = model_load(model_path)

	selected_input_signals = config['params']['selected_input_signals']
	selected_output_signals = config['params']['selected_output_signals']
	time_lag = config['params']['time_lag']

	Ts = config['settings']['Ts']
	n_horizon = config['settings']['n_horizon']
	n_prediction = config['settings']['n_prediction']
	#print('Time lag: ', time_lag)
	#print('Ts: ', Ts)
	#print('Horizon: ', n_horizon)
	#print('Prediction: ', n_prediction)

	# import breathing data
	breathingdata_in, breathingdata_out, breathingdata_group_len = import_data(breathing_data_filepath, selected_ids, selected_input_signals, selected_output_signals)

	# prepare data
	data_x, data_y = data_generator(breathingdata_in, breathingdata_out, breathingdata_group_len, range(0, len(selected_ids)), time_lag, n_horizon, n_prediction)

	if config['params']['dyn_scaler'] == '0to1':
		data_x_sc, data_y_sc, y_min, y_max = scaler_transform_0to1(data_x, data_y)
	elif config['params']['dyn_scaler'] == 'm1to1':
		data_x_sc, data_y_sc, y_min, y_max = scaler_transform_m1to1(data_x, data_y)

	# predict training data and rescale result
	t_start = time.time()
	phi_r_bl_sc = model.predict(data_x_sc).squeeze()

	if config['params']['dyn_scaler'] == '0to1':
		phi_r_bl_pred = scaler_inverse_transform_0to1(phi_r_bl_sc, y_min.squeeze(), y_max.squeeze())
	elif config['params']['dyn_scaler'] == 'm1to1':
		phi_r_bl_pred = scaler_inverse_transform_m1to1(phi_r_bl_sc, y_min.squeeze(), y_max.squeeze())

	y_pred = convert_phi_r_bl_to_breathing_signal(phi_r_bl_pred)
	y_true = convert_phi_r_bl_to_breathing_signal(data_y.squeeze())
	t_end = time.time()
	print(f'Runtime of breathing data prediction: {(t_end - t_start) * 1000} ms')

	return y_true, y_pred, data_y, phi_r_bl_pred, model, config



def ensemble_predict(models_dir, model_path_prefix, breathing_data_filepath, selected_ids):

	model_path_list = glob.glob(os.path.join(models_dir, model_path_prefix))

	y_pred_all = []
	phi_pred_all = []
	r_pred_all = []
	bl_pred_all = []

	for path in model_path_list:
		y_true, y_pred, phi_r_bl_true, phi_r_bl_pred, _, _ = model_predict(path, breathing_data_filepath, selected_ids)

		y_pred_all.append(y_pred)
		phi_pred_all.append(phi_r_bl_pred[:, 0])
		r_pred_all.append(phi_r_bl_pred[:, 1])
		bl_pred_all.append(phi_r_bl_pred[:, 2])

	return np.array(y_true), np.array(phi_r_bl_true[:, 0]), np.array(phi_r_bl_true[:, 1]), np.array(phi_r_bl_true[:, 2]), np.array(y_pred_all), np.array(phi_pred_all), np.array(r_pred_all), np.array(bl_pred_all)

def import_and_scale_breathingdata(data_filepath, selected_ids, settings):

	data_in, data_out, group_len = import_data(data_filepath, selected_ids, settings['selected_input_signals'], settings['selected_output_signals'])
	prediction_ix = list(range(0, len(selected_ids)))
	data_in_sc = (data_in - settings['scaler_input_min']) / (np.asarray(settings['scaler_input_max']) - np.asarray(settings['scaler_input_min']))
	data_out_sc = (data_out - settings['scaler_output_min']) / (np.asarray(settings['scaler_output_max']) - np.asarray(settings['scaler_output_min']))
	data_x_sc, data_y_sc = data_generator(data_in_sc, data_out_sc, group_len, prediction_ix, settings['time_lag'], settings['n_horizon'], 1, 1)

	return data_x_sc, data_y_sc


def scaler_transform_0to1(x, y):
	x_min = np.min(x, 1)
	x_max = np.max(x, 1)
	x_min_arr = np.expand_dims(np.tile(x_min, x.shape[1]), 2)
	x_max_arr = np.expand_dims(np.tile(x_max, x.shape[1]), 2)
	x_sc = (x - x_min_arr) / (x_max_arr - x_min_arr)

	y_min = np.expand_dims(np.vstack((np.zeros(y.shape[0]).T, x_min.T, x_min.T)).T, 1)
	y_max = np.expand_dims(np.vstack(((np.ones(y.shape[0]) * 90).T, x_max.T, x_max.T)).T, 1)
	phi_sc = y[:, :, 0] / 90
	r_sc = y[:, :, 1] / (x_max - x_min)
	bl_sc = (y[:, :, 2] - x_min) / (x_max - x_min)
	y_sc = np.expand_dims(np.vstack((phi_sc.T, r_sc.T, bl_sc.T)).T, 1)

	return x_sc, y_sc, y_min, y_max

def scaler_inverse_transform_0to1(y_sc, y_min, y_max):
	phi = y_sc[:, 0] * 90
	r = y_sc[:, 1] * (y_max[:, 1] - y_min[:, 1])
	bl = y_sc[:, 2] * (y_max[:, 2] - y_min[:, 2]) + y_min[:, 2]
	y = np.vstack((phi.T, r.T, bl.T)).T

	return y


def scaler_transform_m1to1(x, y):
	x_min = np.min(x, 1)
	x_max = np.max(x, 1)
	x_min_arr = np.expand_dims(np.tile(x_min, x.shape[1]), 2)
	half_range_x = (x_max - x_min) / 2
	half_range_x_arr = np.expand_dims(np.tile(half_range_x, x.shape[1]), 2)
	x_sc = (x - half_range_x_arr - x_min_arr) / half_range_x_arr

	y_min = np.expand_dims(np.vstack(((np.ones(y.shape[0]) * -45).T, x_min.T, x_min.T)).T, 1)
	y_max = np.expand_dims(np.vstack(((np.ones(y.shape[0]) * 45).T, x_max.T, x_max.T)).T, 1)
	phi_norm = (y[:, :, 0] - 45) / 45
	r_norm = y[:, :, 1] / half_range_x
	bl_norm = (y[:, :, 2] - y_min[:, :, 2] - half_range_x) / half_range_x
	y_sc = np.expand_dims(np.vstack((phi_norm.T, r_norm.T, bl_norm.T)).T, 1)

	return x_sc, y_sc, y_min, y_max


def scaler_inverse_transform_m1to1(y_sc, y_min, y_max):
	phi = y_sc[:, 0] * 45 + 45
	r = y_sc[:, 1] * ((y_max[:, 1] - y_min[:, 1]) / 2)
	bl = (y_sc[:, 2] * ((y_max[:, 2] - y_min[:, 2]) / 2)) + ((y_max[:, 2] - y_min[:, 2]) / 2) + y_min[:, 2]
	y = np.vstack((phi.T, r.T, bl.T)).T

	return y

def convert_phi_r_bl_to_breathing_signal(y):
	phi = y[:, 0]
	radius = y[:, 1]
	baseline = y[:, 2]
	signal = radius + radius * np.cos(phi/180 * np.pi - np.pi) + baseline
	return signal
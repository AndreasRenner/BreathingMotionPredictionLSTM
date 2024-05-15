# Author: Ing. Ingo Gulyas, MSc, MSc
# Date: 04.05.2022
# Medical University of Vienna / AKH Wien
# Department of Radiation Oncology

from datetime import datetime
from sklearn.model_selection import KFold
from shutil import copyfile

from core_functions import *

# INFO: if n_splits == array count -> leave one out scenario
def init_cross_validation_ids(id_list, n_splits):
    if(n_splits > len(id_list)):
        print_err('cross-validation can not create more splits than elements in list!!!')
        exit(-1)

    if (len(id_list) % n_splits):
        print('WARNING: cross-validation will be done with unequal group sizes!!!')
        input('Press Enter to continue...')

    cv_train_ix, cv_val_ix = [], []
    kf = KFold(n_splits)
    for train_ix, val_ix in kf.split(id_list):
        cv_train_ix.append(train_ix)
        cv_val_ix.append(val_ix)

    print('cross validation allocation (train vs. validate):')
    for t_ix, v_ix in zip(cv_train_ix, cv_val_ix):
        print(t_ix, ' vs. ', v_ix)

    return cv_train_ix, cv_val_ix


def logfile_preprocessing_cv_data(filepath):
    logfile = open(filepath, 'r')
    params = logfile.readline()
    logfile.close()

    if params[0] != '#':
        print_err('logfile header invalid - \'#\' expected as first character but \'%c\' found' % (params[0]))
        return

    # read data section
    df = pd.read_csv(filepath, sep=';', skiprows=1)
    n_epochs = df['Epoch'].max()
    n_lines = df.shape[0]

    if (n_lines % (n_epochs+1)) != 0:
        print_err('logfile %s is incomplete' % filepath)
        return

    if (n_lines / (n_epochs+1)) <= 1:
        print_err('logfile %s is already collapsed' % filepath)

    n_cv = int(n_lines / (n_epochs+1))
    mae_train_cv = np.array(df['MAE_train']).reshape(n_cv, -1)
    mae_test_cv = np.array(df['MAE_test']).reshape(n_cv, -1)
    rmse_train_cv = np.array(df['RMSE_train']).reshape(n_cv, -1)
    rmse_test_cv = np.array(df['RMSE_test']).reshape(n_cv, -1)

    mae_train_avg = mae_train_cv.mean(axis=0)
    mae_test_avg = mae_test_cv.mean(axis=0)
    rmse_train_avg = rmse_train_cv.mean(axis=0)
    rmse_test_avg = rmse_test_cv.mean(axis=0)

    columns = []
    hdrs = ['MAE_train_cv', 'MAE_test_cv', 'RMSE_train_cv', 'RMSE_test_cv']

    for hdr in hdrs:
        for i in range(0, n_cv):
            columns.append('%s%d' % (hdr, i))

    columns.extend(('MAE_train_avg', 'MAE_test_avg', 'RMSE_train_avg', 'RMSE_test_avg'))

    data = np.hstack((np.array(mae_train_cv).T, np.array(mae_test_cv).T,
                      np.array(rmse_train_cv).T, np.array(rmse_test_cv).T,
                      np.array([mae_train_avg]).T, np.array([mae_test_avg]).T,
                      np.array([rmse_train_avg]).T, np.array([rmse_test_avg]).T))

    df2 = pd.DataFrame(data=data, columns=columns)

    fd = open(filepath, 'w', newline='\n')
    fd.write('%s' % params)
    df2.to_csv(fd, index_label='Epoch', sep=';', float_format='%.6f')
    fd.close()

    return



def model_training_crossval_gridsearch(params_lst, settings, base_directory):

    settings.update({'testing_ids': []})

    dt_start = datetime.now()
    dt_start_strg = dt_start.strftime('%Y%m%d-%H%M%S')
    working_dir = os.path.join(base_directory, '%s-LSTM-VAL-CV' % dt_start_strg)
    os.makedirs(working_dir)

    copyfile('main.py', os.path.join(working_dir, 'main.py'))
    copyfile('./train_crossval.py', os.path.join(working_dir, 'train_crossval.py'))
    copyfile('core_functions.py', os.path.join(working_dir, 'core_functions.py'))

    # for all hyper-parameter settings
    for i in range(0, len(params_lst)):
        dt_trainig_start = datetime.now()

        params = params_lst[i]

        breathing_data_filepath = params['breathing_data_filepath']
        selected_input_signals = params['selected_input_signals']
        selected_output_signals = params['selected_output_signals']

        breathingdata_in, breathingdata_out, breathingdata_group_len = import_data(breathing_data_filepath, settings['training_ids'], selected_input_signals, selected_output_signals)

        # separate scaler for each input signal
        if breathingdata_in.shape[1] == 1:
            scaler_input_min = breathingdata_in.min()
            scaler_input_max = breathingdata_in.max()
        else:
            scaler_input_min = np.min(breathingdata_in, axis=0)
            scaler_input_max = np.max(breathingdata_in, axis=0)

        breathingdata_in = (breathingdata_in - scaler_input_min) / (scaler_input_max - scaler_input_min)

        # separate scaler for each output signal
        if breathingdata_out.shape[1] == 1:
            scaler_output_min = np.min(breathingdata_out)
            scaler_output_max = np.max(breathingdata_out)
        else:
            scaler_output_min = np.min(breathingdata_out, axis=0)
            scaler_output_max = np.max(breathingdata_out, axis=0)

        breathingdata_out = (breathingdata_out - scaler_output_min) / (scaler_output_max - scaler_output_min)

        params.update({'breathing_data_filename': get_filename(breathing_data_filepath)})
        params.update({'scaler_input_min': scaler_input_min.tolist()})
        params.update({'scaler_input_max': scaler_input_max.tolist()})
        params.update({'scaler_output_min': scaler_output_min.tolist()})
        params.update({'scaler_output_max': scaler_output_max.tolist()})

        working_dir_hyperparams = os.path.join(working_dir, 'HyperParamConfig%d' % i)
        os.makedirs(working_dir_hyperparams)

        breathingdata_filepath_local = os.path.join(working_dir, get_filename(breathing_data_filepath))
        if not os.path.isfile(breathingdata_filepath_local):
            copyfile(breathing_data_filepath, breathingdata_filepath_local)

        logfilepath = logfile_create(working_dir_hyperparams, params)

        # init cross-validation
        cv_train_ids, cv_val_ids = init_cross_validation_ids(settings['training_ids'], settings['crossval_splits'])

        # cross validation training
        cv_step = 0
        for cv_train_ix, cv_val_ix in zip(cv_train_ids, cv_val_ids):
            dt_trainig_start = datetime.now()

            settings_val = dict(settings)
            settings_val['testing_ids'] = []
            settings_val['training_ids'] = (np.asarray(settings['training_ids'])[cv_train_ix]).tolist()
            settings_val['testing_ids'] = (np.asarray(settings['training_ids'])[cv_val_ix]).tolist()

            # prepare breathing data (leave one out)
            # INFO:
            #  1 input: train_x.shape = (n,40,1), train_y.shape = (n,1,1)
            #  2 inputs: train_x.shape = (n,40,2), train_y.shape = (n,1,1)
            train_data_x, train_data_y = data_generator(breathingdata_in, breathingdata_out, breathingdata_group_len, cv_train_ix,
                                                        params['time_lag'], settings['n_horizon'], settings['n_prediction'], 1)
            val_data_x, val_data_y = data_generator(breathingdata_in, breathingdata_out, breathingdata_group_len, cv_val_ix,
                                                    params['time_lag'], settings['n_horizon'], settings['n_prediction'], 1)

            # shuffle training and test data if required
            if settings['use_data_shuffle'] != 0:
                train_data_x, train_data_y = data_shuffle(train_data_x, train_data_y, 123456)

            # ensure size of training data matches an integer multiple of batch size
            if train_data_x.shape[0] % params['batch_size'] != 0:
                train_data_x = train_data_x[:-(train_data_x.shape[0] % params['batch_size'])]
                train_data_y = train_data_y[:-(train_data_y.shape[0] % params['batch_size'])]

            # init tensorflow random seed for reproducibility
            if settings['use_randseed_reset'] != 0:
                np.random.seed(42)
                tf.random.set_seed(17)

            model = create_model(train_data_x, train_data_y, units=params['units'], batch_size=params['batch_size'], l_rate=params['l_rate'], epochs=settings['epochs_max'])

            my_callback = CustomCallback(train_data_x, train_data_y, val_data_x, val_data_y,
                                         params['scaler_input_min'],
                                         params['scaler_input_max'], np.asarray(params['scaler_output_min']),
                                         np.asarray(params['scaler_output_max']), working_dir_hyperparams, logfilepath,
                                         cv_step, False, False)
            if settings['use_randseed_reset'] != 0:
                np.random.seed(42)
                tf.random.set_seed(17)

            model.fit(train_data_x, train_data_y, epochs=settings['epochs_max'], batch_size=params['batch_size'], verbose=2, callbacks=[my_callback])

            dt_training_end = datetime.now()
            params.update({'training-time': (dt_training_end - dt_trainig_start).total_seconds()})

            #  collect statistics: settings, params and losses
            all_stats = {}
            all_stats['loss_train'] = my_callback.loss_training
            all_stats['loss_test'] = my_callback.loss_testing
            all_stats['params'] = params
            all_stats['settings'] = settings_val

            my_callback.loss_training = []
            my_callback.loss_testing = []

            # store model results in model directory (cv-step)
            if os.path.isfile(os.path.join(working_dir_hyperparams, 'BestTestModel-CV%02d' % cv_step)):
                fd = open(os.path.join(working_dir_hyperparams, 'BestTestModel-CV%02d' % cv_step, 'stats.txt'), 'w')
                fd.write(json.dumps(all_stats))
                fd.write('\n')
                fd.close()

            # store collected results in working directory
            fd = open(os.path.join(working_dir_hyperparams, 'stats_tmp.txt'), 'a')
            fd.write(json.dumps(all_stats))
            fd.write('\n')
            fd.close()

            print('TRAINING step %d/%d done - TRAIN:' % (cv_step, len(cv_train_ids)-1), cv_train_ix, "-> VAL:", cv_val_ix)
            cv_step = cv_step + 1

        logfile_preprocessing_cv_data(logfilepath)

        # finalize overall stats file
        fd = open(os.path.join(working_dir_hyperparams, 'stats_tmp.txt'), 'r')
        lines = fd.read().splitlines()
        fd.close()

        fd = open(os.path.join(working_dir_hyperparams, '%s_stats.txt' % (os.path.basename(os.path.normpath(working_dir_hyperparams)))), 'w')
        fd.write('[')
        for i in range(0, len(lines) - 1):
            fd.write(lines[i])
            fd.write(',\n')
        fd.write(lines[len(lines) - 1])
        fd.write(']')
        fd.close()

        os.remove(os.path.join(working_dir_hyperparams, 'stats_tmp.txt'))

        dt_training_end = datetime.now()

        #params.update({'training-time': (dt_training_end-dt_trainig_start).total_seconds()})
        #fp = os.path.join(working_dir_hyperparams, 'Params.json')
        #with open(fp, 'w') as json_file:
        #    json.dump(params, json_file)

    return


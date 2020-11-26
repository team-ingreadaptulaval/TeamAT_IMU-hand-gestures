from Algo_dev.imusigndetect import *
from Algo_dev import utils
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from time import sleep
import json


class LiveSignatureDetection:

    def __init__(self, at, sequence_file, command_mapping_file, window_len, freq):
        """
        :type at: str
        :type sequence_file: str
        :type command_mapping_file: str
        :type freq: int, float
        """
        # Init classifier
        self.clf = ImuSignDetectClassifier()
        # self.default_fit()
        self.train_file = 'variable_len_plus.sd'
        self.sequence_file = sequence_file
        self.fit_from_trainfile(self.train_file)

        # Constants
        self.freq = freq
        self.idle_tol_accel = 20 / self.freq
        self.idle_tol_gyro = 1 / self.freq
        self.window_buff_len = self.freq
        self.window_len = window_len
        self.trigger_tresh = [15, 3]
        self.min_after_count = 5
        self.acccel_var_tresh = 2
        self.idle_pts_before = 5
        self.idle_time_max = 1.5
        self.missed_max = 1
        self.switch_released_time_offset = 2
        self.required_examples = 10
        self.hold_down_tresh = 2

        # Init counters
        self.inmotion_count = 0
        self.sign_count = 0
        self.window_wait = 0
        self.save_count = 0
        self.after_count = 0
        self.i_buff = 0
        self.i_rec = 0
        self.n_missed = 0
        self.example_count = 0

        # Init variables
        self.__mode = 0
        self.window_buff = np.zeros((6, window_len, self.window_buff_len))
        self.current_window = np.zeros((6, window_len))
        self.command_string = 'signdetect,0,0'
        self.new_target_name = ''
        self.signal_state = 0  # 0: waiting for motion, 1: waiting for motion completion
        self.accel_m1 = 0
        self.gyro_m1 = 0
        self.curr_seq = []
        self.last_command = 0
        self.last_real_command = 0
        self.tresh_sample = [[], []]
        self.fit_examples = []
        self.record_instruction = ''
        self.alternative_info = (None, 0, 0)


        # Timers
        self.idle_timer = float('inf')
        self.hold_down_timer = time()

        # Flags
        self.switch_switched = False
        self.new_fit_done = False
        self.unavailable_flag = False

        # Mappings
        self.known_seq = IMUSequences(self, self.target_id, seq_filename=sequence_file)
        self.command_map_file = command_mapping_file
        self.__pc_dev_map = {'copy': 2, 'paste': 3, 'lclick': 4, 'rclick': 5, 'up': 6, 'down': 7, 'right': 8, 'left': 9,
                           'tab': 10, 'enter': 11}
        self.set_at_map(at)
        self.command_map = self.load_mapping(self.command_map_file)
        print(self.__at_map)
        print('self.command_map: ', self.command_map)

    def get_at_map(self):
        """
        :rtype: dict
        """
        return self.__at_map  # {'copy': 2, 'paste': 3, 'lclick': 4, 'rclick': 5, 'up': 6, 'down': 7, 'right': 8, ...}

    def get_reversed_at_map(self):
        """
        :rtype: dict
        """
        return {v: k for k, v in self.__at_map.items()}   # {2: 'copy', 3: 'paste', 4: 'lclick'}

    def set_at_map(self, at):
        if at == 'at_pc_dev':
            self.__at_map = self.__pc_dev_map.copy()

    def sequence_detection(self, pred_info):
        """
        :rtype: int
        :type signature: list
        """
        # signature: prediction
        signature = pred_info[0]
        k = len(self.targets_names)
        command = self.last_command
        if signature == -1:
            self.idle_timer = time() + self.switch_released_time_offset
            self.last_command = self.last_real_command
            return self.last_real_command
        if signature < k:
            self.curr_seq.append(signature)
            self.idle_timer = time()
            self.alternative_info = (pred_info[2], pred_info[3], pred_info[1] - pred_info[3])
        elif signature == k and (self.n_missed < self.missed_max):
            self.idle_timer = time()
            self.n_missed += 1
        else:  # No significant motion
            # print(time() - self.idle_timer)
            if time() - self.idle_timer > self.idle_time_max:  # Timeout
                # print('timeout')
                action = self.check_sequence()
                self.curr_seq = []
                self.known_seq.possible_sequences = self.known_seq.enabled_sequences
                if action is not None:  # Action detected
                    command = self.__at_map[self.command_map[action]]
                    self.idle_timer = time() + self.switch_released_time_offset
                else:
                    self.last_real_command = self.last_command if self.last_command != 0 else self.last_real_command
                    command = 0
            else:
                action = self.forcheck_sequence(alternative=self.alternative_info)
                if action is not None:  # Action detected
                    self.curr_seq = []
                    self.known_seq.possible_sequences = self.known_seq.enabled_sequences.copy()
                    command = self.__at_map[self.command_map[action]]
                    self.idle_timer = time() + self.switch_released_time_offset
        self.last_command = command
        # print(self.curr_seq, command)
        return command

    def check_sequence(self):
        for keys, seqs in self.known_seq.possible_sequences.items():
            if self.curr_seq == seqs:
                return keys
        return None

    def forcheck_sequence(self, alternative=None):
        uptodate_seq = self.curr_seq
        if len(uptodate_seq) == 0:
            return None
        seqs = self.known_seq.possible_sequences
        cropt_seqs = {}
        for key, values in seqs.items():
            cropt_seqs[key] = values[0:len(uptodate_seq)]
        filtered_seqs = {}
        if uptodate_seq in cropt_seqs.values() and len(seqs) == 1:  # Correspondance
            return self.check_sequence()
        elif uptodate_seq in cropt_seqs.values():  # Debut correspondant
            for key, val in cropt_seqs.items():
                if val == uptodate_seq:
                    filtered_seqs[key] = seqs[key]
            self.known_seq.possible_sequences = filtered_seqs
            return None
        else:
            if alternative is not None:
                second_pred = alternative[0]
                second_prob = alternative[1]
                confidence = alternative[2]
                if confidence < 0.3:  # and second_prob > 0.15
                    print('RECURSIVE PREDICT',second_pred, second_prob, confidence)
                    self.curr_seq = self.curr_seq[0:-1]
                    print(self.curr_seq, '+', second_pred)
                    command = self.sequence_detection(pred_info=(second_pred, second_prob, len(self.targets_names) + 1, 0))
                    return self.get_reversed_at_map().get(command)
            self.idle_timer = 0
            print('here', alternative)
            return None

    def handle_new_data(self, data, switch):
        """
        :rtype: str
        :type switch: int, bool
        :type data: list
        """
        # print(self.signal_state)
        pred = (len(self.targets_names) + 1, None, None, None)
        mag_accel = linalg.norm(data[0:3])
        mag_gyro = linalg.norm(data[3::])
        if not switch:
            if self.switch_switched:
                # print(self.curr_seq, self.curr_seq == False)
                if not self.curr_seq:
                    print('offset')
                    self.idle_timer = time() + self.switch_released_time_offset
                else:
                    self.idle_timer = time()
                self.switch_switched = False
            if self.i_rec < self.window_len:
                self.current_window[:, self.i_rec] = np.array(data)
                self.i_rec += 1
            else:
                self.current_window = np.hstack([self.current_window[:, 1::], np.array(data).reshape(-1, 1)])
                if self.i_buff < self.window_buff_len:
                    self.window_buff[:, :, self.i_buff] = self.current_window
                    self.i_buff += 1
                else:
                    self.window_buff = np.dstack([self.window_buff[:, :, 1::], self.current_window])
                # print(linalg.norm(data[3::]) > self.trigger_tresh)
            if (mag_accel > self.trigger_tresh[0] or mag_gyro > self.trigger_tresh[
                1]) and self.signal_state == 0:  # There is motion // np.sum(np.abs(data[0:3])) > 50 or
                self.zero_up = self.__find_zero_motion(self.current_window[0:3, :])
                if self.zero_up is not None:
                    self.window_wait = self.zero_up
                    self.signal_state = 1
                    # print(self.signal_state)
            # IDLE METHOD
            if self.signal_state == 1:  # Signature live analysis
                self.inmotion_count += 1
                # print(self.after_count)
                if (-self.idle_tol_accel < mag_accel - self.accel_m1 < self.idle_tol_accel) or (
                        -self.idle_tol_gyro < mag_gyro - self.gyro_m1 < self.idle_tol_gyro):
                    self.after_count += 1
                    # print(self.after_count, mag_accel - self.accel_m1, mag_gyro - self.gyro_m1)
                else:
                    self.after_count = 0
                if self.after_count > self.min_after_count:
                    acc_var, gyro_var, _, _ = self.__signal_mag_var(
                        self.current_window[:, -self.inmotion_count - self.idle_pts_before::])
                    if self.__mode == 0:  # Live detect
                        if acc_var > self.acccel_var_tresh and self.inmotion_count < self.window_len:
                            pred = self.clf.predict([s.reshape(1, -1) for s in self.current_window[:,
                                                                               -self.inmotion_count - self.idle_pts_before::]], with_second_choice=True)
                            print('P R E D I C T: ', pred)
                            print(self.inmotion_count)
                            # if pred < len(self.targets_names):
                            # print(mag_accel, mag_gyro)
                            # plt.figure()
                            # plt.plot(linalg.norm(self.current_window[0:3, -self.inmotion_count - self.idle_pts_before::], axis=0))
                            # plt.plot(linalg.norm(self.current_window[3::, -self.inmotion_count - self.idle_pts_before::], axis=0))
                            # plt.show()
                    elif self.__mode == 1:  # Record signature
                        self.unavailable_flag = True
                        if acc_var > self.acccel_var_tresh/2 and self.inmotion_count < self.window_len:
                            if self.example_count < self.required_examples:
                                self. new_fit_done = False
                                self.fit_examples.append(self.current_window[:, -self.inmotion_count - self.idle_pts_before::])
                                self.example_count += 1
                                print(self.example_count)
                            if self.example_count >= self.required_examples:
                                print('compiling new data + train')
                                utils.append_train_file(self.train_file, self.fit_examples, self.new_target_name)
                                self.fit_from_trainfile(self.train_file)
                                self.example_count = 0
                                # self.__mode = 0
                                self.known_seq.target_id = self.target_id
                                self.new_fit_done = True
                                self.unavailable_flag = False
                            # np.savetxt(fname=f'train_{self.save_count}.csv',
                            #            X=self.current_window[:, -self.inmotion_count - self.idle_pts_before::],
                            #            delimiter=',')
                            # plt.plot(
                            #     linalg.norm(self.current_window[0:3, -self.inmotion_count - self.idle_pts_before::],
                            #                 axis=0).T)
                            # plt.plot(
                            #     linalg.norm(self.current_window[3::, -self.inmotion_count - self.idle_pts_before::],
                            #                 axis=0).T)
                            # plt.show()
                            # self.save_count += 1
                    elif self.__mode == 2:  # Teach tresholds
                        print('teach')
                        self.teach_tresh()
                    self.after_count = 0
                    self.signal_state = 0
                    self.inmotion_count = 0
        else:  # Switch on
            if self.last_command == 0:
                if not self.switch_switched:
                    self.hold_down_timer = time()
                    print('mode = 0 and ssw', self.last_real_command)
                elif time() - self.hold_down_timer > self.hold_down_tresh:
                    pred = (-1, None, None, None)
            self.idle_timer = float('inf')
            self.switch_switched = True
            # print('switch on')
        self.accel_m1 = mag_accel
        self.gyro_m1 = mag_gyro
        try:
            command = self.sequence_detection(pred)
        except KeyError:
            command = self.last_command # NOt sure...
            print('command not mapped')
        self.command_string = f'signdetect,{command},{switch}'
        # print(self.command_string)
        return self.command_string

    def delete_sign(self, sign):
        """
        :type sign: str
        """
        self.known_seq.clear_seq_on_sign_deleted(sign)
        utils.delete_from_train_file(self.train_file, sign)
        self.fit_from_trainfile(self.train_file)
        for key in self.command_map.copy().keys():
            if sign in key.split('-'):
                self.command_map.pop(key)
        with open(self.command_map_file, 'w') as f:
            f.write(json.dumps([self.at, self.command_map]))

    def teach_tresh(self):
        self.__mode = 2
        max_tap, max_swipe = 5, 5
        the_window = self.current_window[:, -self.inmotion_count - self.idle_pts_before::]
        # print(self.tresh_sample)
        if len(self.tresh_sample[0]) + len(self.tresh_sample[1]) == 0:
            self.store_tresh = (self.trigger_tresh, self.acccel_var_tresh)
            self.trigger_tresh = [0.75 * tt for tt in self.trigger_tresh]
            self.acccel_var_tresh *= 0.5
            print('tap')
        if len(self.tresh_sample[0]) < max_tap:
            self.tresh_sample[0].append(the_window)
            sleep(0.5)
            if len(self.tresh_sample[0]) < max_tap:
                print('tap')
            else:
                print('swipe')
        elif len(self.tresh_sample[1]) < max_swipe:
            self.tresh_sample[1].append(the_window)
            if len(self.tresh_sample[1]) < max_swipe:
                sleep(0.5)
                print('swipe')
        else:
            # TODO: trigger algo maybe (maybe a slider in GUI)
            var_mean = 0
            var_rec = [[], []]
            for tap, swipe in zip(*self.tresh_sample):
                tap_accel = linalg.norm(tap[0:3, :])
                tap_gyro = linalg.norm(tap[3::, :])
                swipe_accel = linalg.norm(swipe[0:3, :])
                swipe_gyro = linalg.norm(swipe[3::, :])
                tap_var, _, _, _ = self.__signal_mag_var(tap)
                swipe_var, _, _, _ = self.__signal_mag_var(swipe)
                var_rec[0].append(tap_var)
                var_rec[1].append(swipe_var)
            self.__mode = 0
            self.trigger_tresh = self.store_tresh[0]
            self.acccel_var_tresh = min(np.quantile(var_rec[0], 0.25), np.quantile(var_rec[1], 0.25))

    def __signal_mag_var(self, signal):
        """
        :type signal: np.array
        """
        mag_acc = linalg.norm(signal[0:3, :], axis=0)
        mag_gyr = linalg.norm(signal[3::, :], axis=0)
        dmag_acc = (mag_acc[1::] - mag_acc[0:-1]) / (1 / self.freq)
        dmag_gyr = (mag_gyr[1::] - mag_gyr[0:-1]) / (1 / self.freq)
        return np.var(mag_acc), np.var(mag_gyr), np.var(dmag_acc), np.var(dmag_gyr)

    def __find_zero_motion(self, sensor_signals):
        mag = linalg.norm(sensor_signals, axis=0)
        tol = self.idle_tol_accel
        diff = (mag[1::] - mag[0:-1]) * self.freq
        motion = (diff < tol * 100) & (diff > -tol * 100)  # True = no motion
        max_idle_pts = self.idle_pts_before
        for n_idle_pts in range(max_idle_pts, max_idle_pts - 3, -1):
            idle = motion[0:-(n_idle_pts - 1)] & motion[1:-(n_idle_pts - 2)]
            for i in range(2, n_idle_pts - 1):
                idle &= motion[i: -(n_idle_pts - (i + 1))]
            idle &= motion[n_idle_pts - 1::]
            zero_motion_point = np.where(idle)[0]
            if zero_motion_point.size != 0:
                zero_motion_point = np.where(idle)[0][-1]
                # print(zero_motion_point, n_idle_pts)
                # plt.plot(gyro_mag)
                # plt.show()
                return zero_motion_point
        return None

    def fit_from_trainfile(self, filename):
        self.unavailable_flag = True
        signals, targets, self.targets_names = utils.load_FS_IMU_data(
            file=filename)
        print(self.targets_names)
        file_targets = targets[np.sort(np.unique(targets, return_index=True)[1])]
        tfile_target2name = {ft: name for name, ft in zip(self.targets_names, file_targets)}
        self.target_id = {name: i for i, name in enumerate(self.targets_names)}  #  sign2#
        targets = np.array([self.target_id[tfile_target2name[t]] for t in targets])
        # self.target_id = {name: i for i, name in zip(targets[np.sort(np.unique(targets, return_index=True)[1])], self.targets_names)}
        print('target_id:', self.target_id)
        print('targets: ', targets)
        self.train_signals = signals.copy()
        self.train_targets = targets.copy()
        self.known_seq = IMUSequences(self, self.target_id, seq_filename=self.sequence_file)
        self.refit()
        # signals, targets = self.known_seq.transform(signals, targets)
        # try:
        #     self.clf.fit(signals, targets)
        # except AssertionError:
        #     # Only one class
        #     reject_signals = [sensor[self.train_targets != np.unique(targets)[0]] for sensor in self.train_signals]
        #     reject_targets = targets.max() * np.ones(reject_signals[0].shape[0])
        #     all_signal = [np.vstack([good, bad]) for good, bad in zip(signals, reject_signals)]
        #     all_targets = np.hstack([targets, reject_targets])
        #     self.clf.fit(all_signal, all_targets)
        #
        # print('train preds: ', self.clf.predict(signals))

    def refit(self):
        self.unavailable_flag = True
        print('REFIT', end='')
        signals, targets = self.known_seq.transform(self.train_signals, self.train_targets)
        try:
            self.clf.fit(signals, targets)
            print('train preds: ', self.clf.predict(signals))
        except AssertionError:
            # Only one class
            reject_signals = [sensor[self.train_targets != np.unique(targets)[0]] for sensor in self.train_signals]
            reject_targets = self.train_targets.max() * np.ones(reject_signals[0].shape[0]) + 1
            all_signal = [np.vstack([good, bad]) for good, bad in zip(signals, reject_signals)]
            all_targets = np.hstack([targets, reject_targets])
            self.clf.fit(all_signal, all_targets)
            print('train preds: ', self.clf.predict(all_signal))
        print(' --DONE')
        self.unavailable_flag = False

    def get_mode(self):
        return self.__mode

    def set_mode(self, mode):
        """
        :type mode: int
        """
        self.__mode = mode
        print(self.__mode)

    def load_mapping(self, user_filename):
        """
        :type user_filename: str
        :rtype: dict
        """
        with open(user_filename, 'r') as fich:
            string = fich.read()
        # return {'handtap-handtap': 11, 'handtap-swipe': 14, 'swipe-handtap': 66,
        #         'swipe-swipe': 99, 'swipe-swipe-handtap': 199, 'handtap-handtap-swipe': 111}
        # print(json.loads(string))
        info = json.loads(string)
        self.at = info[0]
        return info[1]


class IMUSequences:
    def __init__(self, parent, target_id, seq_filename):
        """
        :type seq_filename: str
        :type target_id: dict
        :type parent: LiveSignatureDetection
        """
        self.parent = parent
        self.target_id = target_id  # {'sign': #}
        self.num2sign = {v: k for k, v in self.target_id.items()}  # {#: 'sign'}
        self.seq_file = seq_filename
        with open(self.seq_file, 'r') as fich:
            string = fich.read()
            self.sequences_init = json.loads(string)  # {'sign_name1-sign_name2': bool}
            print('self.sequences_init: ', self.sequences_init)
        self.sequences = {key: [target_id[sign] for sign in key.split('-')] for key, val in self.sequences_init.items()} # {'sign_name1-sign_name2': [#1,#2]}
        self.enabled_sequences = {}
        for key, val in self.sequences_init.items():
            if val:
                self.enabled_sequences[key] = [target_id[sign] for sign in key.split('-')]  # {'sign_name1-sign_name2': [#1,#2]}
        # self.enabled_sequences = self.sequences.copy()
        self.possible_sequences = self.enabled_sequences.copy() # {'sign_name1-sign_name2': [#1,#2]}
        self.enabeled_sign = set([subelement for element in [sign.split('-') for sign in self.enabled_sequences.keys()] for subelement in element])
        self.available_sign = set(self.target_id.keys())
        self.used_signals = parent.train_signals.copy()
        self.used_targets = parent.train_targets.copy()

    def set_sequence(self, key_list):
        self.enabled_sequences = {}
        self.sequences_init = {key: 0 for key in self.sequences_init.keys()}
        for key in key_list:
            if self.parent.command_map.get(key) is not None:
                self.enabled_sequences[key] = self.sequences[key]
            self.sequences_init[key] = 1
        print('self.enabled_sequences: ', self.enabled_sequences)
        self.possible_sequences = self.enabled_sequences.copy()
        self.enabeled_sign = set(
            [subelement for element in [sign.split('-') for sign in self.enabled_sequences.keys()] for subelement in
             element])
        return self.evaluate_confusion()

    def evaluate_confusion(self):
        confusion = ()
        for key, values in self.enabled_sequences.items():
            cropt_seqs = {}
            for ckey, cvalues in self.enabled_sequences.items():
                cropt_seqs[ckey] = cvalues[0:len(values)]
            duplicate = utils.dictfind_duplicate(dict0=cropt_seqs, val_to_compare=values)
            if len(duplicate) > 1:
                confusion += (duplicate,)
        # print(confusion)
        return confusion

    def save_new_sequence(self, seq):
        # print(sequences)
        if self.sequences.get(seq) is not None:
            raise IndexError
        else:
            print('Seq finale', seq)
            self.sequences_init[seq] = 1
            self.sequences[seq] = [self.target_id[sign] for sign in seq.split('-')]
            self.enabled_sequences[seq] = self.sequences[seq].copy()
            self.possible_sequences = self.enabled_sequences.copy()
            self.save_init_modif()

    def delete_sequence(self, seqs):
        for seq in seqs:
            self.sequences_init.pop(seq)
            self.sequences.pop(seq)
            try:
                self.enabled_sequences.pop(seq)
                self.possible_sequences.pop(seq)
            except KeyError:
                pass
        self.save_init_modif()


    def save_init_modif(self):
        with open(self.seq_file, 'w') as fich:
            fich.write(json.dumps(self.sequences_init))

    def clear_seq_on_sign_deleted(self, sign):
        self.target_id.pop(sign)
        sign_names = list(self.target_id.keys())
        seq_to_pop = []
        for seq in self.sequences_init.keys():
            # if not any(ele in seq.split('-') for ele in sign_names):
            if sign in seq.split('-'):
                seq_to_pop.append(seq)
        for seq in seq_to_pop:
            for dicts in [self.sequences_init, self.sequences, self.enabled_sequences, self.possible_sequences]:
                try:
                    dicts.pop(seq)
                except KeyError:
                    pass
        self.save_init_modif()
        # for seq in self.sequences.keys():
        #     if not any(ele in seq for ele in sign_names):
        #         self.sequences.pop(seq)
        # for seq in self.enabled_sequences.keys():
        #     if not any(ele in seq for ele in sign_names):
        #         self.enabled_sequences.pop(seq)
        # for seq in self.possible_sequences.keys():
        #     if not any(ele in seq for ele in sign_names):
        #         self.possible_sequences.pop(seq)

    def transform(self, signals, targets):
        old_signals = self.used_signals.copy()
        old_targets = self.used_targets.copy()
        if self.enabeled_sign != self.available_sign:  # cut sign
            for unused_sign in self.available_sign - self.enabeled_sign:
                id = self.target_id[unused_sign]
                for s, sensor in enumerate(old_signals):
                    old_signals[s] = np.delete(sensor, np.where(old_targets == id)[0], axis=0)
                old_targets = old_targets[old_targets != id]
            for reused_sign in self.enabeled_sign - self.available_sign:  # add sign
                id = self.target_id[reused_sign]
                for s, sensor in enumerate(signals):
                    old_signals[s] = np.vstack([old_signals[s], sensor[targets == id, :]])
                old_targets = np.hstack([old_targets, id * np.ones(len(targets[targets == id]), dtype=int)])
        self.available_sign = {self.num2sign[t] for t in np.unique(old_targets)}
        self.used_signals = old_signals
        self.used_targets = old_targets
        return self.used_signals , self.used_targets


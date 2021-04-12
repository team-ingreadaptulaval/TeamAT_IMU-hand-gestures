from algo.imusigndetect import *
from algo import utils
import numpy as np
from numpy import linalg
from time import sleep
import json


class LiveSignatureDetection:

    def __init__(self, at_file, sequence_file, command_mapping_file, trainset_file, window_len, freq):
        """
        :type at: str
        :type sequence_file: str
        :type command_mapping_file: str
        :type freq: int, float
        """

        self.at_file = at_file
        self.at = at_file.split('.')[0]

        # Mappings
        # self.known_seq = IMUSequences(self, self.target_id, seq_filename=sequence_file)
        self.command_map_file = command_mapping_file
        self.__at_map = self.load_at_file()
        self.command_map = self.load_mapping(self.command_map_file, True)
        print(self.__at_map)
        print('self.command_map: ', self.command_map)

        # Init classifier
        self.clf = ImuSignDetectClassifier()
        self.train_file = trainset_file
        self.sequence_file = sequence_file
        self.fit_from_trainfile(self.train_file)

        # Constants
        self.FREQ = freq
        self.IDLE_TOL_ACCEL = 15 / self.FREQ
        self.IDLE_TOL_GYRO = 1 / self.FREQ
        self.WINDOW_BUF_LEN = self.FREQ
        self.WINDOW_LEN = window_len
        self.TRIGGER_TRESH = [15, 3]
        self.MIN_AFTER_COUNT = 5 # augmenter
        self.ACCEL_VARIANCE_TRESH = 2
        self.IDLE_PTS_BEFORE_WINDOW = 5
        self.IDLE_TIME_MAX = 1.5 # augmenter
        self.MISSED_MAX = 1
        self.SWITCH_RELEASE_TIME_OFFSET = 2
        self.REQUIRED_EXAMPLES = 10
        self.HOLD_DOWN_TRESH = 2
        self.N_SENSORS = 6

        # Init counters
        self.inmotion_count = 0
        self.sign_count = 0
        self.window_wait = 0
        self.save_count = 0
        self.after_count = 0
        self.i_window_buff = 0
        self.i_rec = 0
        self.n_missed = 0
        self.example_count = 0

        # Init variables
        self.__mode = 0
        self.window_buff = np.zeros((self.N_SENSORS, window_len, self.WINDOW_BUF_LEN))
        self.current_window = np.zeros((self.N_SENSORS, window_len))
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
        self.autoclick = False

    def fit_from_trainfile(self, filename):
        self.unavailable_flag = True
        signals, targets, self.targets_names = utils.load_imu_data(file=filename)
        print(targets, signals)
        if not self.targets_names:
            file_targets = np.array([])
        else:
            file_targets = targets[np.sort(np.unique(targets, return_index=True)[1])]
        tfile_target2name = {ft: name for name, ft in zip(self.targets_names, file_targets)}
        self.target_id = {name: i for i, name in enumerate(self.targets_names)}  #  sign to #id
        targets = np.array([self.target_id[tfile_target2name[t]] for t in targets])
        print('target_id:', self.target_id)
        print('targets: ', targets)
        self.train_signals = signals.copy()
        self.train_targets = targets.copy()
        self.sequence_context = IMUSequences(signals, targets, self.target_id, self.command_map, seq_filename=self.sequence_file)
        self.refit()

    def load_at_file(self):
        with open(self.at_file, 'r') as f:
            string = f.read()
        return json.loads(string)

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

    def handle_new_data(self, data, switch_pressed):
        """
        :rtype: str
        :type switch_pressed: int, bool
        :type data: list
        """
        pred = (len(self.targets_names) + 1, None, None, None)
        mag_accel = linalg.norm(data[0:3])
        mag_gyro = linalg.norm(data[3::])

        if not switch_pressed:
            if self.switch_switched:
                self.adjust_idle_timer()
            self.handle_sliding_window(data)
            if self.waiting_for_motion():
                if self.motion_detected(mag_gyro, mag_gyro):  # There is motion //
                    self.zero_up = self.__find_zero_motion(self.current_window[0:3, :])
                    if self.zero_up is not None:
                        self.window_wait = self.zero_up
                        self.signal_state = 1
            if self.waiting_for_motion_complete():  # Signature live analysis
                self.inmotion_count += 1
                if not self.is_significant_motion(mag_accel, mag_gyro):
                    self.after_count += 1
                else:
                    self.after_count = 0
                if self.motion_completed():
                    if self.__mode == 0:
                        pred = self.predict_current_window()
                    elif self.__mode == 1:
                        self.handle_new_signature_recording(mag_accel, mag_gyro)
                    elif self.__mode == 2:
                        print('teach thresholds')
                        self.teach_thresh()
                    self.after_count = 0
                    self.signal_state = 0
                    self.inmotion_count = 0
        else:  # Switch on
            if self.last_command == 0: # no command
                if not self.switch_switched:
                    self.hold_down_timer = time()
                elif time() - self.hold_down_timer > self.HOLD_DOWN_TRESH:
                    pred = [self.last_real_command]
            self.idle_timer = float('inf')
            self.switch_switched = True
        self.accel_m1 = mag_accel
        self.gyro_m1 = mag_gyro
        command = self.sequence_detection(pred)
        ################# ------------------#
        if self.autoclick and  command in [2, 3, 4, 5, 11]:
            switch_pressed = 1
            self.last_command = 0
        #------------------ #################
        self.command_string = f'signdetect,{command},{switch_pressed}'
        return self.command_string

    def sequence_detection(self, pred_info):
        """
        :rtype: int
        :type signature: list
        """
        # signature:= prediction
        signature = pred_info[0]
        k = len(self.targets_names)
        last_command = self.last_command
        if signature == -1: # No sign
            self.idle_timer = time() + self.SWITCH_RELEASE_TIME_OFFSET
            self.last_command = self.last_real_command
            return self.last_real_command
        if signature < k: # one of the available sign
            self.curr_seq.append(signature)
            self.idle_timer = time()
        elif signature == k and (self.n_missed < self.MISSED_MAX): # reject
            self.idle_timer = time()
            self.n_missed += 1
        else:  # No significant motion
            self.compute_command(last_command)
        return self.last_command

    def compute_command(self, last_command):
        # print(last_command)
        command = last_command
        if time() - self.idle_timer > self.IDLE_TIME_MAX:  # Timeout
            action = self.check_sequence()
            self.curr_seq = []
            self.sequence_context.possible_sequences = self.sequence_context.enabled_sequences
            if action is not None:  # Action detected
                command = self.__at_map[self.command_map[action]]
                self.idle_timer = time() + self.SWITCH_RELEASE_TIME_OFFSET
            else:
                self.last_real_command = self.last_command if self.last_command != 0 else self.last_real_command
                command = 0
        else:
            action = self.forcheck_sequence()
            if action is not None:  # Action detected
                self.curr_seq = []
                self.sequence_context.possible_sequences = self.sequence_context.enabled_sequences.copy()
                command = self.__at_map[self.command_map[action]]
                self.idle_timer = time() + self.SWITCH_RELEASE_TIME_OFFSET
        self.last_command = command

    def check_sequence(self):
        for keys, seqs in self.sequence_context.possible_sequences.items():
            if self.curr_seq == seqs:
                return keys
        return None

    def forcheck_sequence(self, alternative=None):
        uptodate_seq = self.curr_seq
        if len(uptodate_seq) == 0:
            return None
        seqs = self.sequence_context.possible_sequences
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
            self.sequence_context.possible_sequences = filtered_seqs
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

    def handle_sliding_window(self, data):
        if self.i_rec < self.WINDOW_LEN:
            self.current_window[:, self.i_rec] = np.array(data)
            self.i_rec += 1
        else:
            self.current_window = np.hstack([self.current_window[:, 1::], np.array(data).reshape(-1, 1)])
            if self.i_window_buff < self.WINDOW_BUF_LEN:
                self.window_buff[:, :, self.i_window_buff] = self.current_window
                self.i_window_buff += 1
            else:
                self.window_buff = np.dstack([self.window_buff[:, :, 1::], self.current_window])

    def adjust_idle_timer(self):
        if not self.curr_seq:
            print('offset')
            self.idle_timer = time() + self.SWITCH_RELEASE_TIME_OFFSET
        else:
            self.idle_timer = time()
        self.switch_switched = False

    def predict_current_window(self):
        pred = [self.last_command]
        acc_var, gyro_var, _, _ = self.__signal_mag_var(
            self.current_window[:, -self.inmotion_count - self.IDLE_PTS_BEFORE_WINDOW::])
        if acc_var > self.ACCEL_VARIANCE_TRESH and self.inmotion_count < self.WINDOW_LEN:
            pred = self.clf.predict([s.reshape(1, -1) for s in self.current_window[:,
                                                               -self.inmotion_count - self.IDLE_PTS_BEFORE_WINDOW::]], with_second_choice=True)
            print('P R E D I C T: ', pred)
        return pred

    def handle_new_signature_recording(self, acc, gyr):
        acc_var, gyro_var, _, _ = self.__signal_mag_var(
            self.current_window[:, -self.inmotion_count - self.IDLE_PTS_BEFORE_WINDOW::])
        self.unavailable_flag = True
        print(self.is_significant_motion(acc, gyr),)
        if acc_var > self.ACCEL_VARIANCE_TRESH/2 and self.inmotion_count < self.WINDOW_LEN:
            if self.example_count < self.REQUIRED_EXAMPLES:
                self.new_fit_done = False
                self.fit_examples.append(self.current_window[:, -self.inmotion_count - self.IDLE_PTS_BEFORE_WINDOW::])
                self.example_count += 1
                print(self.example_count)
            if self.example_count >= self.REQUIRED_EXAMPLES:
                print('compiling new data + train')
                utils.append_train_file(self.train_file, self.fit_examples, self.new_target_name)
                self.fit_examples = []
                print('file appended')
                self.fit_from_trainfile(self.train_file)
                self.example_count = 0
                # self.__mode = 0
                self.sequence_context.target_id = self.target_id
                self.new_fit_done = True
                self.unavailable_flag = False
                print('teching done')

    def delete_sign(self, sign):
        """
        :type sign: str
        """
        self.sequence_context.clear_seq_on_sign_deleted(sign)
        utils.delete_from_train_file(self.train_file, sign)
        self.fit_from_trainfile(self.train_file)
        for key in self.command_map.copy().keys():
            if sign in key.split('-'):
                self.command_map.pop(key)
        with open(self.command_map_file, 'w') as f:
            f.write(json.dumps([self.at, self.command_map]))

    def teach_thresh(self):
        self.__mode = 2
        max_tap, max_swipe = 5, 5
        the_window = self.current_window[:, -self.inmotion_count - self.IDLE_PTS_BEFORE_WINDOW::]
        # print(self.tresh_sample)
        if len(self.tresh_sample[0]) + len(self.tresh_sample[1]) == 0:
            self.store_tresh = (self.TRIGGER_TRESH, self.ACCEL_VARIANCE_TRESH)
            self.TRIGGER_TRESH = [0.75 * tt for tt in self.TRIGGER_TRESH]
            self.ACCEL_VARIANCE_TRESH *= 0.5
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
            self.TRIGGER_TRESH = self.store_tresh[0]
            self.ACCEL_VARIANCE_TRESH = min(np.quantile(var_rec[0], 0.25), np.quantile(var_rec[1], 0.25))

    def __signal_mag_var(self, signal):
        """
        :type signal: np.array
        """
        mag_acc = linalg.norm(signal[0:3, :], axis=0)
        mag_gyr = linalg.norm(signal[3::, :], axis=0)
        dmag_acc = (mag_acc[1::] - mag_acc[0:-1]) / (1 / self.FREQ)
        dmag_gyr = (mag_gyr[1::] - mag_gyr[0:-1]) / (1 / self.FREQ)
        return np.var(mag_acc), np.var(mag_gyr), np.var(dmag_acc), np.var(dmag_gyr)

    def __find_zero_motion(self, sensor_signals):
        mag = linalg.norm(sensor_signals, axis=0)
        tol = self.IDLE_TOL_ACCEL
        diff = (mag[1::] - mag[0:-1]) * self.FREQ
        motion = (diff < tol * 100) & (diff > -tol * 100)  # True = no motion
        max_idle_pts = self.IDLE_PTS_BEFORE_WINDOW
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

    def refit(self):
        self.unavailable_flag = True
        print('REFIT')
        print(self.train_targets)
        signals, targets = self.sequence_context.transform(self.train_signals, self.train_targets)
        try:
            self.clf.fit(signals, targets)
            print('train preds: ', self.clf.predict(signals))
        except AssertionError:
            print('DATASET ERROR')
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

    def load_mapping(self, user_filename=None, init=False):
        """
        :type user_filename: str
        :rtype: dict
        """
        print('user_filename', user_filename)
        if user_filename is None:
            user_filename = self.command_map_file
        with open(user_filename, 'r') as fich:
            string = fich.read()
        info = json.loads(string)
        self.at = info[0]
        if not init:
            self.sequence_context.command_map = info[1]
        return info[1]

    def is_significant_motion(self, current_accel, current_gyro):
        return not((-self.IDLE_TOL_ACCEL < current_accel - self.accel_m1 < self.IDLE_TOL_ACCEL) or
                   (-self.IDLE_TOL_GYRO < current_gyro - self.gyro_m1 < self.IDLE_TOL_GYRO))

    def waiting_for_motion_complete(self):
        return self.signal_state == 1

    def waiting_for_motion(self):
        return self.signal_state == 0

    def motion_completed(self):
        return self.after_count > self.MIN_AFTER_COUNT

    def motion_detected(self, accel, gyro):
        return (accel > self.TRIGGER_TRESH[0] or gyro > self.TRIGGER_TRESH[1])

    def update_enabeled_seqs(self, checklist):
        self.sequence_context.update_seq_file(checklist)

class IMUSequences:
    def __init__(self, x_train, y_train, target_id, cmd_map, seq_filename):
        """
        :type seq_filename: str
        :type target_id: dict
        :type parent: LiveSignatureDetection
        """
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
        self.possible_sequences = self.enabled_sequences.copy() # {'sign_name1-sign_name2': [#1,#2]}
        self.enabeled_sign = set([subelement for element in [sign.split('-') for sign in self.enabled_sequences.keys()] for subelement in element])
        self.available_sign = set(self.target_id.keys())
        self.enabeled_sign_m1 = self.enabeled_sign.copy()
        self.used_signals = x_train.copy()
        self.used_targets = y_train.copy()
        self.command_map = cmd_map

    def set_sequence(self, key_list):
        print('set sequence1', key_list, self.command_map)
        self.update_seq_file(key_list)
        self.enabled_sequences = {}
        self.sequences_init = {key: 0 for key in self.sequences_init.keys()}
        for key in key_list:
            if self.command_map.get(key) is not None:
                self.enabled_sequences[key] = self.sequences[key]
            self.sequences_init[key] = 1
        print('self.enabled_sequences: ', self.enabled_sequences)
        self.possible_sequences = self.enabled_sequences.copy()
        print('set sequence2', self.enabeled_sign)
        self.enabeled_sign = set(
            [subelement for element in [sign.split('-') for sign in self.enabled_sequences.keys()] for subelement in
             element])
        print('set sequence3', self.enabeled_sign)
        return self.evaluate_confusion()

    def evaluate_confusion(self):
        confusion = ()
        for key, values in self.enabled_sequences.items():
            cropt_seqs = {}
            for ckey, cvalues in self.enabled_sequences.items():
                cropt_seqs[ckey] = cvalues[0:len(values)]
            duplicate = utils.dictfind_duplicate(dict0=cropt_seqs, val_to_compare=values)
            if len(duplicate) > 1:
                confusion += (duplicate, )
        return confusion

    def save_new_sequence(self, seq):
        if self.sequences.get(seq) is not None:
            raise IndexError
        else:
            print('Seq finale', seq)
            self.sequences_init[seq] = 0
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
        print(self.sequences_init)
        with open(self.seq_file, 'w') as fich:
            fich.write(json.dumps(self.sequences_init))

    def clear_seq_on_sign_deleted(self, sign):
        self.target_id.pop(sign)
        # sign_names = list(self.target_id.keys())
        seq_to_pop = []
        for seq in self.sequences_init.keys():
            if sign in seq.split('-'):
                seq_to_pop.append(seq)
        for seq in seq_to_pop:
            for dicts in [self.sequences_init, self.sequences, self.enabled_sequences, self.possible_sequences]:
                try:
                    dicts.pop(seq)
                except KeyError:
                    pass
        self.save_init_modif()


    def transform(self, signals, targets):
        print('transform->enabeled_sign', self.enabeled_sign)
        if not self.enabeled_sign:
            return [], []
        new_signals = self.used_signals
        new_targets = self.used_targets
        if self.enabeled_sign != self.available_sign:  # cut sign
            for unused_sign in self.available_sign - self.enabeled_sign:
                id = self.target_id[unused_sign]
                for s, sensor in enumerate(new_signals):
                    new_signals[s] = np.delete(sensor, np.where(new_targets == id)[0], axis=0)
                new_targets = new_targets[new_targets != id]
            for reused_sign in self.enabeled_sign - self.available_sign:  # add sign
                id = self.target_id[reused_sign]
                for s, sensor in enumerate(signals):
                    new_signals[s] = np.vstack([new_signals[s], sensor[targets == id, :]])
                new_targets = np.hstack([new_targets, id * np.ones(len(targets[targets == id]), dtype=int)])
        self.available_sign = {self.num2sign[t] for t in np.unique(new_targets)}
        self.used_signals = new_signals
        self.used_targets = new_targets
        return self.used_signals , self.used_targets

    def update_seq_file(self, checked_seqs):
        with open(self.seq_file, 'r') as f:
            content = f.read()
        content = json.loads(content)
        for seq in content.keys():
            if seq in checked_seqs:
                content[seq] = 1
            else:
                content[seq] = 0
        with open(self.seq_file, 'w') as f:
            f.write(json.dumps(content))


    def sign_is_changed(self):
        changed = False
        if self.enabeled_sign != self.enabeled_sign_m1:
            changed = True
            self.enabeled_sign_m1 = self.enabeled_sign
        return changed
# https://kuntalchandra.wordpress.com/2017/08/23/python-socket-programming-server-client-application-using-threads/
# https://www.youtube.com/watch?v=T0rYSFPAR0A

import socket
import sys
import traceback
from threading import Thread, Lock
from queue import Queue
import os
from time import sleep

import win32api
from pid import PidFile


class MultiClientServer:

    def __init__(self, ip_adress='127.0.0.1', port=5005, max_bufsize=512):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.local_hostname = socket.gethostname()
        self.local_fqdn = socket.getfqdn()
        self.ip_adress = ip_adress
        self.port = port
        self.server_adress = (ip_adress, port)
        self.max_bufsize = max_bufsize
        self.connection = None
        self.display_init_info()
        self.lock = Lock()
        self.sensor_data = [0, 0, 0, 0, 0, 0]
        self.at_command = 9999
        self.count = 0
        # self.sensor_data_m1 = self.sensor_data.copy()
        # self.at_command_m1 = 0

    def display_init_info(self):
        print(f'Working on {self.local_hostname} ({self.local_fqdn}) with {self.ip_adress}')
        print(f'Starting up on {self.server_adress[0]} on port {self.server_adress[1]}')

    def start_server(self):
        try:
            self.s.bind(self.server_adress)
        except:
            print("Bind failed. Error : " + str(sys.exc_info()))
            sys.exit()
        self.s.listen(5)
        n_threads = 0
        while True:
            print('waiting for connection')
            connection, client_adress = self.s.accept()
            try:
                ip, port = str(client_adress[0]), str(client_adress[1])
                Thread(target=self.client_thread, args=(connection, ip, port, self.max_bufsize, n_threads)).start()
                n_threads += 1
            except:
                print("Thread did not start")
                traceback.print_exc()

    def client_thread(self, connection, ip, port, max_bufsize=512, number=0):
        try:
            print('connection from ', ip, ' ', port)
            data_count = 0
            while True:
                data = connection.recv(max_bufsize)
                if data:
                    # if data_count % 100 == 0:
                    sender, info = decodestr(data)
                    # print(f'({number}) {data_count} Sender: {sender} Data: {info}', flush=True)
                    data_count += 1
                    if sender == 'xsdata':
                        with self.lock:
                            self.sensor_data = info
                            connection.send(str('ack').encode('utf-8'))
                    elif sender == 'signdetect':
                        with self.lock:
                            # print(info, flush=True)
                            self.at_command = f'{info[0]},{info[1]}'
                            connection.send(','.join([str(s) for s in self.sensor_data]).encode('utf-8'))
                    elif sender == 'hid':
                        with self.lock:
                            connection.send(str(self.at_command).encode('utf-8'))
                else:
                    break
        except ConnectionResetError:
            print(f'lost {ip} on {port}')
        finally:
            connection.close()

def decodestr(reception):
    reception = reception.decode()
    reception = reception.split(',')
    return (reception[0], [float(num) for num in reception[1::]])

# def on_exit(sig, func=None):
#     print("exit handler")
#     os.remove(os.getcwd()+ '/' + 'signdetectsocket.pid')
# win32api.SetConsoleCtrlHandler(on_exit, True)

if __name__ == "__main__":
    with PidFile('signdetectsocket', os.getcwd()) as p:
        try:
            mcs = MultiClientServer()
            mcs.start_server()
        except KeyboardInterrupt:
            print('closing local server')
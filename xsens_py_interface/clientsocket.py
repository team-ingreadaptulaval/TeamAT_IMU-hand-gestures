import numpy as np
import socket


class ClientSocket:

    def __init__(self, port=5005, ip='127.0.0.1'):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.local_hostname = socket.gethostname()
        self.local_fqdn = socket.getfqdn()
        self.ip_adress = ip  # socket.gethostbyname(local_hostname)
        self.server_adress = (self.ip_adress, port)
        self.s.connect(self.server_adress)
        print(f'Connecting to {self.local_hostname}({self.local_fqdn}) with {self.ip_adress}')

    def exchange_test(self):
        data = [0]
        while True:
            if sum(data) > 20:
                self.s.sendall('signdetect,10'.encode("utf-8"))
            else:
                self.s.sendall('signdetect,-10'.encode("utf-8"))
            try:
                data = self.s.recv(512)
                if data:
                    data = [float(num) for num in data.decode().split(',')]
                    print(data)
            except:
                data = [0]
                print('-')

    def send(self, string_send):
        self.s.sendall(string_send.encode("utf-8"))

    def recieve(self, nbytes=512):
        try:
            data = self.s.recv(nbytes)
            if data:
                data = [float(num) for num in data.decode().split(',')]
                # print(data)
        except:
            data = [0]
            print('-')
        return data

    def __del__(self):
        self.s.close()


if __name__ == "__main__":
    cs = ClientSocket()
    cs.exchange_test()

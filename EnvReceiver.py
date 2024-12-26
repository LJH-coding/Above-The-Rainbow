import socket
import pickle

class EnvReceiver:
    def __init__(self):
        host = '0.0.0.0'
        port = 48763
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
        print("Waiting for a connection...")
        self.connection, client_address = sock.accept()
        print("Connection established with", client_address)

    def _send_data(self, data_to_send):
        data = pickle.dumps(data_to_send, 2)
        self.connection.sendall(data)

    def _recv_data(self):
        length = self.connection.recv(4)
        length = int.from_bytes(length, byteorder='big')
        data = b''
        while len(data) < length:
            chunk = self.connection.recv(4096)
            data += chunk
        return data
    
    def reset(self):
        self._send_data("reset")
        data = self._recv_data()
        return pickle.loads(data, encoding='bytes')

    def step(self, action):
        self._send_data(action)
        data = self._recv_data()
        return pickle.loads(data, encoding='bytes')


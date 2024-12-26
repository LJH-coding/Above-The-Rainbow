import socket
import pickle

def _send_data(conn, data_to_send):
    data = pickle.dumps(data_to_send, 2)
    conn.sendall(data)

def _recv_data(conn):
    length = conn.recv(4)
    length = int.from_bytes(length, byteorder='big')
    data = b''
    while len(data) < length:
        chunk = conn.recv(4096)
        data += chunk
    return data

def main():
    windows_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    windows_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    windows_socket.bind(('0.0.0.0', 9000))  # Bind to an appropriate port
    windows_socket.listen(1)
    print("Waiting for Windows connection...")
    windows_conn, windows_addr = windows_socket.accept()
    print(f'Connection from {windows_addr}')
    
    env_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    env_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    env_socket.bind(('0.0.0.0', 12345))
    env_socket.listen(1)
    print("Waiting for Env connection...")
    env_conn, env_addr = env_socket.accept()
    print(f'Connection from {env_addr}')
    
    while True:
        data = windows_conn.recv(4096)
        command = data.decode()
        print(command)
        print("----------------------")
        if command != 'reset':
            command = eval(command)
            _send_data(env_conn, command)
            data = _recv_data(env_conn)
            obs, reward, done, info = pickle.loads(data, encoding='bytes')
        else:
            _send_data(env_conn, command)
            data = _recv_data(env_conn)
            pickle.loads(data, encoding='bytes')
            done = False
        windows_conn.sendall(str(done).encode())

if __name__ == "__main__":
    main()
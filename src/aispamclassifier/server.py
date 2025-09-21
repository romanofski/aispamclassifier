import socket
import sys
import os
import pathlib

from aispamclassifier.inference import detect_spam_or_ham

SOCKET_PATH = pathlib.Path(f'/run/user/{os.getuid()}/aispamclassifier.sock')
MODEL_PATH = pathlib.Path(os.environ.get('AISPAMCLASSIFIER_MODEL', ''))

def main():
    if not MODEL_PATH.exists():
        print(f'Unable to find model under: {MODEL_PATH}. Exiting')
        sys.exit(1)

    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(SOCKET_PATH))
        SOCKET_PATH.chmod(0o600)
        server.listen()

        print(f'Listening on {SOCKET_PATH}')

        while True:
            connection, _ = server.accept()
            with connection:
                socket_stream = connection.makefile('rb')
                label = detect_spam_or_ham(socket_stream, MODEL_PATH)
                connection.sendall(label.encode('utf-8'))

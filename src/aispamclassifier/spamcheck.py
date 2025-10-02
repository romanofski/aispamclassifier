import sys
import argparse
import socket

from aispamclassifier.config import SOCKET_PATH

def main():
    parser = argparse.ArgumentParser(description='classify mail/spam non spam')
    parser.add_argument('emailfile',
                        type=argparse.FileType('rb'))
    args = parser.parse_args()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(str(SOCKET_PATH))
        client.sendall(args.emailfile.read())
        client.shutdown(socket.SHUT_WR)

        response = client.recv(1024).decode().strip()
        print(response.lower())


if __name__ == "__main__":
    main()

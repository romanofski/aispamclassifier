import sys
import argparse
import socket
import io
import email
from typing import Literal, BinaryIO
from email import policy
from email.generator import BytesGenerator

from aispamclassifier.config import SOCKET_PATH

DEFAULT_HEADER='X-AI-Spam'
ResultAction = Literal['print', 'tag']

def classify(rawbody: bytes) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(str(SOCKET_PATH))
        client.sendall(rawbody)
        client.shutdown(socket.SHUT_WR)

        response = client.recv(1024).decode().strip()
        return response.lower()

def handle_email(emailfile: BinaryIO, action: ResultAction, out: BinaryIO=sys.stdout.buffer):
    msg = email.message_from_binary_file(emailfile, policy=policy.default)
    with io.BytesIO() as buf:
        BytesGenerator(buf).flatten(msg)
        raw_bytes = buf.getvalue()

    result = classify(raw_bytes)

    if action == 'print':
        print(result)
        return

    msg.add_header(DEFAULT_HEADER, result)

    with io.BytesIO() as output:
        BytesGenerator(output).flatten(msg)
        out.write(output.getvalue())


def main():
    parser = argparse.ArgumentParser(description='classify mail/spam non spam')
    parser.add_argument('emailfile', nargs='?',
                        type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('--result-action', choices=('tag', 'print'),
                        default='tag', help='Print the classified result instead of changing the email to set a header')
    args = parser.parse_args()

    handle_email(args.emailfile, args.result_action)
    sys.exit(0)

if __name__ == "__main__":
    main()

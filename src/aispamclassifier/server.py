import socket
import sys
import os
import pathlib
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from aispamclassifier.inference import detect_spam_or_ham
from aispamclassifier.config import SOCKET_PATH

MODEL_PATH_ENV_VAR_NAME = 'AISPAMCLASSIFIER_MODEL'

def serve(modelpath: str):
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForSequenceClassification.from_pretrained(modelpath)

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
                label = detect_spam_or_ham(socket_stream,
                                           tokenizer=tokenizer,
                                           model=model)
                try:
                    connection.sendall(label.encode('utf-8'))
                except BrokenPipeError:
                    print('Client closed connection before we could answer')

def valid_directory(path_str: str):
    path = pathlib.Path(path_str)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"'{path}' not a directory")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"'{path}' does not exist")
    return path

def main():
    parser = argparse.ArgumentParser(description='classify mail/spam non spam')
    parser.add_argument('--modelpath', type=valid_directory)

    args = parser.parse_args()
    if args.modelpath is None:
        modelpath = os.getenv(MODEL_PATH_ENV_VAR_NAME)
        if modelpath is None:
            parser.error(
            f"Tried to load model from {MODEL_PATH_ENV_VAR_NAME} but found it not set.")
        args.modelpath = valid_directory(modelpath)

    serve(**args.__dict__)


if __name__ == '__main__':
    main()

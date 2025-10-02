import pathlib
import os

SOCKET_PATH = pathlib.Path(f'/run/user/{os.getuid()}/aispamclassifier.sock')

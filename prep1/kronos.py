#andres botello
import time

class krono(object):
    def __init__(self):
        self.s_time = time.time()
    def start(self):
        self.s_time = time.time()
    def elapsed(self):
        e_time = time.time() - self.s_time
        return e_time
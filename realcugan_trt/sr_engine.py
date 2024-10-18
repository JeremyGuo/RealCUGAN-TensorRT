import multiprocessing
import queue
import threading
from typing import Sequence
import numpy as np
from .upcunet2x import RealCUGANUpScaler2x

class SREngine:
    def __init__(self, models : Sequence[RealCUGANUpScaler2x], queue_size=256):
        self.models = models
        self.input_queue = queue.Queue(queue_size)
        self.output_queue = queue.Queue(queue_size)
        self.threads = []
        self.new_input_lock = threading.Lock()
        self.new_input_condition = threading.Condition(self.new_input_lock)
        self.stopped = True
    
    def start(self):
        self.stopped = False
        def proc(model):
            while not self.stopped:
                with self.new_input_condition:
                    while not self.stopped:
                        try:
                            index, input = self.input_queue.get(block=False)
                        except Exception as e:
                            input = None
                        if input is not None:
                            break
                        self.new_input_condition.wait()
                    if self.stopped: break
                output = model(input)
                self.output_queue.put((index, np.copy(output)))
        self.threads = [threading.Thread(target=proc, args=(model,)) for model in self.models]
        for p in self.threads:
            p.start()
    
    def stop(self):
        with self.new_input_condition:
            self.stopped = True
            self.new_input_condition.notify_all()
        for p in self.threads:
            p.join()
        self.threads = []
    
    def push(self, index, input, timeout=None):
        """
        Input must be (1, 3, H, W)
        """
        with self.new_input_condition:
            self.input_queue.put((index, input), block=(timeout is None), timeout=timeout)
            self.new_input_condition.notify()
    
    def get(self, timeout=None):
        return self.output_queue.get(block=(timeout is None), timeout=timeout)

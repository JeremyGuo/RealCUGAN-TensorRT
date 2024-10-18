from realcugan_trt.upcunet2x import RealCUGANUpScaler2x
import numpy as np
import time
import threading
import cv2
from realcugan_trt.sr_engine import SREngine

model1 = RealCUGANUpScaler2x('./weights/pro-conservative-up2x.pth', True, device='cuda:0', tile=5, alpha=1, w=1920, h=1080, export_engine=True)
model2 = RealCUGANUpScaler2x('./weights/pro-conservative-up2x.pth', True, device='cuda:0', tile=5, alpha=1, w=1920, h=1080, export_engine=True)

count_frames = 0
fram_lock = threading.Lock()

x = (np.random.rand(1, 1080, 1920, 3)*255).astype(np.uint8)
begin_time = time.time()
for i in range(64):
    xinput = np.copy(x)

not_stop = True
def run_thread(model):
    global count_frames
    global not_stop
    global fram_lock
    while not_stop:
        model(np.copy(x))
        with fram_lock:
            count_frames += 1

thread1 = threading.Thread(target=run_thread, args=(model1,))
thread2 = threading.Thread(target=run_thread, args=(model2,))
thread1.start()
thread2.start()

import signal
def signal_handler(sig, frame):
    global not_stop
    not_stop = False
signal.signal(signal.SIGINT, signal_handler)

while time.time() - begin_time < 20 and not_stop:
    print(count_frames / (time.time() - begin_time))
    time.sleep(1)

not_stop = False
thread1.join()
thread2.join()

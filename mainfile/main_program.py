import cv2
import numpy as np
import threading
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import cv2
import math
from torchvision import transforms
import numpy as np
import os
import time
from tqdm import tqdm

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model
pose_model = load_model()
def run_inference(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = img
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = pose_model(image)
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=pose_model.yaml['nc'], nkpt=pose_model.yaml['nkpt'],
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    return output, image
def visualize_output(output, image):
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    return nimg
def draw_class_on_image(label, img, bottomLeftCornerOfText = (10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img
def detect(model, lm_list):
    global label2
    global label3
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    xresults = tf.math.argmax(input = results[0])
    if xresults == 2:
        label3 ='Acc: '+ str("{:.4f}".format(results[0][xresults]))
        label2 = "Doi tuong co dau hieu trom cap"
    elif xresults == 1:
        label3 ='Acc: '+ str("{:.4f}".format(results[0][xresults]))
        label2 = "Doi tuong dang di bo"
    else:
        label3 ='Acc: '+ str("{:.4f}".format(results[0][xresults]))
        label2 = "Doi tuong dang dung im"
    return label2, label3
label = "Dang khoi dong..."
label1 = ''
label2 = ''
label3 = ''
labelacc = ''
n_time_steps = 5
lm_list = []

detect_model = tf.keras.models.load_model("detect_yolov7_model.h5")

cap = cv2.VideoCapture(0)

i = 0
j = 0
warmup_frames = 30
prev_frame_time = 0
new_frame_time = 0

while True:
    success, frame = cap.read()
    output, img = run_inference(frame)
    img = visualize_output(output, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    i = i + 1
    if i > warmup_frames:
        if output != []:
            label = 'Phat hien doi tuong xam nhap'
            j = j + 1
            if j > 50:
                label1 = 'Chua xac dinh hanh vi doi tuong'
            if len(output[0][7:]) == 51:
                lm_list.append(output[0][7:])
                if len(lm_list) == n_time_steps:
                    t1 = threading.Thread(target=detect, args=(detect_model, lm_list,))
                    t1.start()
                    lm_list = []
                    j = 0
                label1 = label2
                labelacc = label3
        else:
            label = 'Khong phat hien xam nhap'
            label1 = '...'
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    img = draw_class_on_image(label, img)
    img = draw_class_on_image(label1, img, bottomLeftCornerOfText = (10, 60))
    img = draw_class_on_image('FPS: '+ fps, img,bottomLeftCornerOfText = (10, 90))
    img = draw_class_on_image( labelacc, img,bottomLeftCornerOfText = (10, 120))
    cv2.imshow("Nhan dien xam nhap", img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
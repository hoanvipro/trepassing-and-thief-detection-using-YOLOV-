import matplotlib.pyplot as plt
import torch
import cv2
import math
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
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

model = load_model()
def run_inference(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = img
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = model(image)
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
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


cap = cv2.VideoCapture('stand.mp4')
lm_list = []
no_of_frames = 500
while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    output, img = run_inference(frame)
    img = visualize_output(output, img)
    if len(output[0][7:]) == 51:
        lm_list.append(output[0][7:])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord('q'):
        break
df  = pd.DataFrame(lm_list)
df.to_csv("stand.txt")
print("saved stand.txt")
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture('walk.mp4')
lm_list = []
no_of_frames = 500
while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    output, img = run_inference(frame)
    img = visualize_output(output, img)
    if len(output[0][7:]) == 51:
        lm_list.append(output[0][7:])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord('q'):
        break
df  = pd.DataFrame(lm_list)
df.to_csv("walk.txt")
print("saved walk.txt")
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture('thief.mp4')
lm_list = []
no_of_frames = 500
while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    output, img = run_inference(frame)
    img = visualize_output(output, img)
    if len(output[0][7:]) == 51:
        lm_list.append(output[0][7:])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord('q'):
        break
df  = pd.DataFrame(lm_list)
df.to_csv("thief.txt")
print("saved thief.txt")
cap.release()
cv2.destroyAllWindows()
# Đọc dữ liệu
stand_df = pd.read_csv("stand.txt")
walk_df = pd.read_csv("walk.txt")
thief_df = pd.read_csv("thief.txt")

X = []
y = []
no_of_timesteps = 5

dataset = stand_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

dataset = walk_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)

dataset = thief_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(2)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = tensorflow.keras.utils.to_categorical(y_train, 3)
y_test = tensorflow.keras.utils.to_categorical(y_test, 3)
model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 3, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")
early_stopping_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32,validation_data=(X_test, y_test),callbacks=[early_stopping_monitor])
model.save("detect_yolov7_model.h5")
print('model_save')

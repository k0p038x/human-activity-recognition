
# coding: utf-8

# In[2]:


import os
import random


# In[3]:


# Loading frames


path = os.getcwd()+'/Data'

file_names = []
target_names = []

assign = []

cnt = -1
for file  in os.listdir(path):
  cnt = cnt+1
  assign.append(file)
  print("Assigning "+str(cnt)+" to "+file)
  
  for ele  in os.listdir(path+'/'+file):
    file_names.append(ele)
    target_names.append(cnt)
  


print("Total Videos : "+str(len(file_names)))

c = list(zip(file_names,target_names))
random.shuffle(c)

file_names, target_names = zip(*c)

train_files = file_names[:623]
train_targets = target_names[:623]

test_files = file_names[623:]
test_targets = target_names[623:]

check = [0,0,0,0,0,0,0]
for i in train_targets:
  check[i] = check[i] + 1

print("Training set : ",end=" ")
print(check)

check = [0,0,0,0,0,0,0]
for i in test_targets:
  check[i] = check[i] + 1
print("Test set : ",end=" ")
print(check)

print("Size of training set : "+str(len(train_files)))
print("Size of test set : "+str(len(test_files)))

print("")

# Dividing train set into train set and valid set

valid_files = train_files[468:]
valid_targets = train_targets[468:]

train_files = train_files[:468]
train_targets = train_targets[:468]

print("Size of training set : "+str(len(train_files)))
print("Size of test set : "+str(len(test_files)))
print("Size of validation set : "+str(len(valid_files)))


# In[4]:


#Extracting frames and store them in a np array

import numpy as np
from skvideo.utils import rgb2gray
import cv2
from tqdm import tqdm


train_frames = []
tmp = 0
for i in tqdm(range(len(train_files))):
  
  file = train_files[i]
  vid_path = path+'/'+assign[train_targets[i]]+'/'+file
  ind_frames = []
  frames_count = 5
  while frames_count!=205:
    img = cv2.imread(vid_path+'/frame-'+str(frames_count)+'.png',1)
    img = cv2.resize(img, (160,120))
#     print(vid_path+'/frame-'+str(frames_count)+'.png')
    frames_count = frames_count + 5
    tmp = tmp + 1
    ind_frames.append(img)
  
  ind_frames = rgb2gray(ind_frames)
  train_frames.append(ind_frames)

train_frames = np.array(train_frames)
print(train_frames.shape)

    


# In[5]:


import numpy as np
from skvideo.utils import rgb2gray
import cv2
from tqdm import tqdm

test_frames = []
tmp = 0
for i in tqdm(range(len(test_files))):
  
  file = test_files[i]
  vid_path = path+'/'+assign[test_targets[i]]+'/'+file
  ind_frames = []
  frames_count = 5
  while frames_count!=205:
    img = cv2.imread(vid_path+'/frame-'+str(frames_count)+'.png',1)
    img = cv2.resize(img, (160,120))
    frames_count = frames_count + 5
    tmp = tmp + 1
    ind_frames.append(img)
  
  ind_frames = rgb2gray(ind_frames)
  test_frames.append(ind_frames)

test_frames = np.array(test_frames)
print(test_frames.shape)

valid_frames = []
tmp = 0
for i in tqdm(range(len(valid_files))):
  
  file = valid_files[i]
  vid_path = path+'/'+assign[valid_targets[i]]+'/'+file
  ind_frames = []
  frames_count = 5
  while frames_count!=205:
    img = cv2.imread(vid_path+'/frame-'+str(frames_count)+'.png',1)
    img = cv2.resize(img, (160,120))
    frames_count = frames_count + 5
    tmp = tmp + 1
    ind_frames.append(img)

  ind_frames = rgb2gray(ind_frames)
  valid_frames.append(ind_frames)

valid_frames = np.array(valid_frames)
print(valid_frames.shape)


# In[25]:


from keras.utils import to_categorical
import numpy as np



train_frames.astype('float32')
train_frames/=255
train_frames*=2
train_frames-=1

test_frames.astype('float32')
test_frames/=255
test_frames*=2
test_frames-=1

valid_frames.astype('float32')
valid_frames/=255
valid_frames*=2
valid_frames-=1


X_train = train_frames
Y_train = to_categorical(train_targets, num_classes=7)

X_test = test_frames
Y_test = to_categorical(test_targets, num_classes=7)

X_valid = valid_frames
Y_valid = to_categorical(valid_targets, num_classes=7)

print('Final Shape : ')
print("X_train : "+str(X_train.shape))
print("X_test : "+str(X_test.shape))
print("X_valid : "+str(X_valid.shape))
print()
print("Y_train : "+str(Y_train.shape))
print("Y_test : "+str(Y_test.shape))
print("Y_valid : "+str(Y_valid.shape))





# In[6]:


from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, BatchNormalization
from keras.layers.core import Dense, Dropout


model = Sequential()


model.add(Conv3D(filters=16, kernel_size=(5, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', 
                 input_shape=X_train.shape[1:]))
model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(Conv3D(filters=64, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu'))
model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(Conv3D(filters=256, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu'))
model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(Conv3D(filters=1024, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu'))
model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))


model.add(GlobalAveragePooling3D())


model.add(Dense(32, activation='relu'))

# Dropout Layer
model.add(Dropout(0.5))

# Output layer
model.add(Dense(7, activation='softmax'))

model.summary()


# In[ ]:



from keras.callbacks import ModelCheckpoint

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

# Saving the model that performed the best on the validation set
checkpoint = ModelCheckpoint(filepath='org.weights.best.hdf5', save_best_only=True, verbose=1)

# Training the model for 40 epochs
history = model.fit(X_train, Y_train, batch_size=16, epochs=40, 
                    validation_data=(X_valid, Y_valid), verbose=2, callbacks=[checkpoint])


# In[ ]:


model.load_weights('org.weights.best.hdf5')

(loss, accuracy) = model.evaluate(X_test, Y_test, batch_size=16, verbose=0)

print('Accuracy on test data: {:.3f}%'.format(accuracy * 100))

conf = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]

tot = [0,0,0,0,0,0,0]
count = 0
for i in (X_test):
    ans = model.predict(X_test[count:count+1],verbose = 1)
    maxx_ind = 0
    maxx = ans[0][0]
    r = 0
    for j in range(1,7):
        if Y_test[count][j]==1:
            r = j
        if ans[0][j] > maxx:
            maxx = ans[0][j]
            maxx_ind = j
    j = maxx_ind
    tot[r] = tot[r] + 1
    conf[r][j] = conf[r][j] + 1
    count = count + 1

print(conf)
print("")
print(tot)



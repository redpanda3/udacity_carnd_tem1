
# coding: utf-8

# In[1]:

##import everything here
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D, MaxPooling2D, Convolution2D
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import numpy as np
import sklearn
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[3]:

samples = []

def read_log(file):
    global samples
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)


# In[4]:




file_0 = 'try0/driving_log_0.csv'
file_1 = 'try0/driving_log_1.csv'
    
read_log(file_0)
read_log(file_1)


# In[ ]:




# In[5]:

##add_flipped_data



def add_flip_data():
    global samples
    add_flip_samples=[]
    for sample in samples:
        #uncomment here for the first run
        #image_center = cv2.imread('./try0/IMG/'+sample[0].split('\\')[-1])
        #image_left = cv2.imread('./try0/IMG/'+sample[1].split('\\')[-1])
        #image_right = cv2.imread('./try0/IMG/'+sample[2].split('\\')[-1])  
        #cv2.imwrite('./try0/IMG/'+'flip_'+sample[0].split('\\')[-1], cv2.flip(image_center, 1))
        #cv2.imwrite('./try0/IMG/'+'flip_'+sample[1].split('\\')[-1], cv2.flip(image_left, 1))
        #cv2.imwrite('./try0/IMG/'+'flip_'+sample[2].split('\\')[-1], cv2.flip(image_right, 1))
        add_flip_samples.append(["\\"+'flip_'+sample[0].split('\\')[-1], "\\"+'flip_'+sample[1].split('\\')[-1], "\\"+'flip_'+sample[2].split('\\')[-1], -1.0*float(sample[3]), sample[4], sample[5], sample[6]])  
    samples = samples + add_flip_samples

add_flip_data()
        
 


# In[ ]:


    


# In[6]:

##add left and right data



def add_lr_data():
    global samples
    add_lr_samples=[]
    for sample in samples:
        add_lr_samples.append(["\\"+sample[1].split('\\')[-1], "none", "none", float(sample[3])+0.20, sample[4], sample[5], sample[6]])
        add_lr_samples.append(["\\"+sample[2].split('\\')[-1], "none", "none", float(sample[3])-0.20, sample[4], sample[5], sample[6]])
    samples = samples + add_lr_samples    


# In[7]:

add_lr_data()


# In[8]:

print(len(samples))


# In[13]:

##balance data

##nbins and max_examples are hyperparameters

import matplotlib.pyplot as plt

histo = []

def balance_data(nbins = 2000, max_examples = 200):
    global samples,histo
    samples = np.array(samples)
    balanced = np.empty([0, samples.shape[1]], dtype=samples.dtype)
    for i in range((-1)*nbins, nbins):
        begin = 1.0*i/nbins
        end = begin + 1.0 / nbins
        extracted = samples[(samples[:,3].astype(float) >= begin) & (samples[:,3].astype(float) < end)]
        np.random.shuffle(extracted)
        extracted = extracted[0:max_examples, :]
        histo.append(len(extracted))
        balanced = np.concatenate((balanced, extracted), axis=0)  
    return balanced


# In[14]:

input_data = balance_data()

plt.plot(range(len(histo)), histo)


# In[16]:

plt.show()


# In[ ]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(input_data, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './try0/IMG/'+batch_sample[0].split('\\')[-1]
                image = cv2.imread(name)
                angle = batch_sample[3]
                images.append(image)
                angles.append(angle)
                


                # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)





# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[ ]:




# In[ ]:




model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,0), (0,0))))
model.add(Convolution2D(8, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.summary()

print("done")

model.compile(loss='mse', optimizer='adam')
best_model = ModelCheckpoint('model_best1.h5', verbose=2, save_best_only=True)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,validation_steps = len(validation_samples), epochs=1, callbacks=[best_model] )


# In[71]:

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()


# In[72]:

##train the network


# In[82]:

##write back
model.save('model_last.h5')


# In[ ]:




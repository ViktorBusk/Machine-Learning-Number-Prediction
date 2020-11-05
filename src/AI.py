import tensorflow as tf
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class NumberAI:
    '''
    `NumberAI` a neuralnetwork using `tf.keras.Model` to create a the arcitecture for the AI.

    `NumberAI` provides training using `tf.keras.Model` and image prediction using `numpy.array`.
               
    ```python
    # Note that you must first call `fit`,
    # for the model to able to call `predict` on some input data.
    
    number_ai = NumberAI()
    number_ai.fit()
    number_ai.predict(some_numpy_input_image_array_1_x_784)
    ```
    '''
    def __init__(self):
        '''
        Creates a `NumberAI` object instance.
        '''
        # Handle datasets
        self.mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
        (self.x_train, self.y_train),(self.x_test, self.y_test) = self.mnist.load_data() # Load data
        self.threshold_value = 200
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.__reshape_data() # Reshape the data

        # Build neural network
        self.model = tf.keras.models.Sequential()
        self.__create_layers()
        self.history = None 
        self.test_loss = None
        self.test_acc = None

        # Augument data
        #self.__data_augmentation()
        #self.__shuffle_train()
    
    def __shapex(self, X, threshold): 
        '''
        Convert all the pixels for all the rgb images to either be 0 or 1 based on a given threshold value.

        Arguments:
            X: The x_dataset to shape.
            threshold: The threshold value, in range 0 to 255 (inclusive). 
                       A higher value means less of the brighter pixels are shown.
        
        Raises:
            ValueError: If the threshold value is not within range of 0 to 255 (inclusive).
        '''
        if threshold < 0 or threshold > 255:
            raise ValueError('The given threshold, ' + str(threshold) + ', was out of range, must be between 0 and 255 (inclusive)')
        XX = np.empty_like(X)
        XX[X< threshold] = 0
        XX[X>=threshold] = 1
        XX = XX.reshape(*XX.shape, 1) 
        return XX

    def __shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

    def __data_augmentation(self, augment_size=5000): 
        print("Augumenting data...")
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.05, 
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False)

        image_generator.fit(self.x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[randidx].copy()
        y_augmented = self.y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                    batch_size=augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0] 

    def __show_image(self, dataset, index = 0):
        '''
        Shows an image of the dataset using mathplotlib (for debug purposes).

        Arguments:
            dataset: The dataset that containts the image.
            index: The index of the image in the given dataset.

        Raises:
            ValueError: If the index is not within range of the dataset.
        '''
        if index < 0 or index >= len(dataset): # Validate index
            raise ValueError('The given index, ' + str(index) +  ', was out of range, \n' +
            ' must be between 0 and ' + str(len(dataset) - 1) + ' (inclusive) for this dataset')
        
        else:
            plt.imshow(dataset[index], cmap="gray") # Import the image
            plt.show() # Show the image

    def __reshape_data(self):
        '''
        Modify data to be compatible with user drawing input.
        '''
        self.x_train = self.__shapex(self.x_train, self.threshold_value)
        self.x_test = self.__shapex(self.x_test, self.threshold_value)
        
    def __create_layers(self): 
        '''
        Create the structure for the neural network.
        '''
        # Add the Flatten Layer
        self.model.add(tf.keras.layers.Flatten())
       
        # Build the input and the hidden layers
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
       
        # Add the Output Layer
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        
    def fit(self):
        '''
        Train the neuralnetwork on the datasets.
        '''
        # Compile the model
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Start training process
        self.history = self.model.fit(x=self.x_train, y=self.y_train, epochs=5)

        # Evaluate the model performance
        self.test_loss, self.test_acc = self.model.evaluate(x=self.x_test, y=self.y_test)
        
        # Print out the model accuracy 
        print('\nTest accuracy:', self.test_acc)

    def predict(self, input_image): 
        '''
        Make the AI predict a number from an image. 
        
        This can only be called if `fit` has been called before.

        Arguments:
            input_image: A 1D numpy array with the shape [1 x 784] used as input for the neural network.
        '''
        if self.history == None:
            return ('AI has not been fit. Call the fit function first!')
       
        else:
            # Make prediction based on input image
            prediction = self.model.predict(input_image)
            
            # Concatinate strings
            prediction_class = str(np.argmax(prediction))
            prediction_probability = str(round(float(np.max(prediction)) * 100, 2)) + " %"
            output_strings = [prediction_class, prediction_probability, np.max(prediction)]

            return output_strings
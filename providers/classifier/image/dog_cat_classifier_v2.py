from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2

class DogCatClassifier():

    def __init__(self, *args, **kwargs):
        pass
    
    def preprocessing(self):
        train_dir = 'dataset/dog_cat/train'
        validation_dir = 'dataset/dog_cat/validation'
      
        # All images will be rescaled by 1./255
        # train_datagen = ImageDataGenerator(rescale=1./255)
        # Adding rescale, rotation_range, width_shift_range, height_shift_range,
        # shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Flow training images in batches of 20 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=20,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary'
        )

        # Flow validation images in batches of 20 using test_datagen generator
        validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary'
        )
        return train_generator, validation_generator


    def create_model(self):
        local_weights_file = 'dataset/model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model = InceptionV3(
            input_shape=(150, 150, 3), include_top=False, weights=None)
        pre_trained_model.load_weights(local_weights_file)
        

        for layer in pre_trained_model.layers:
            layer.trainable = False

        last_layer = pre_trained_model.get_layer('mixed7')
        last_output = last_layer.output
        print('last layer output shape:', last_layer.output_shape)
        # pre_trained_model.summary()
        # Flatten the output layer to 1 dimension
        x = layers.Flatten()(last_output)
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        # Add a dropout rate of 0.2
        x = layers.Dropout(0.2)(x)
        # Add a final sigmoid layer for classification
        x = layers.Dense(1, activation='sigmoid')(x)

        # Configure and compile the model
        model = Model(pre_trained_model.input, x)
        model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(lr=0.0001),
                    metrics=['acc'])
        return model

    def train(self):

        train_generator, validation_generator = self.preprocessing()
        
        model = self.create_model()
        # model.summary()

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,  # 2000 images = batch_size * steps
            epochs=15,
            validation_data=validation_generator,
            validation_steps=50,  # 1000 images = batch_size * steps
            verbose=1
        )

        # Retrieve a list of accuracy results on training and test data
        # sets for each training epoch
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        # Retrieve a list of list results on training and test data
        # sets for each training epoch
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Get number of epochs
        epochs = range(len(acc))

        # Plot training and validation accuracy per epoch
        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('Training and validation accuracy')

        plt.figure()

        # Plot training and validation loss per epoch
        plt.plot(epochs, loss)
        plt.plot(epochs, val_loss)
        plt.title('Training and validation loss')
        plt.show()
        print('train classifier')
        input()

    def classify(self):
        pass

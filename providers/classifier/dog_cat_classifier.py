from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt

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
        # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
        # the three color channels: R, G, and B
        img_input = layers.Input(shape=(150, 150, 3))

        # First convolution extracts 16 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(16, 3, activation='relu')(img_input)
        x = layers.MaxPooling2D(2)(x)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Third convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Convolution2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Flatten feature map to a 1-dim tensor
        x = layers.Flatten()(x)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = layers.Dense(512, activation='relu')(x)

        # Add a dropout rate of 0.5
        x = layers.Dropout(0.5)(x)

        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(1, activation='sigmoid')(x)

        # Configure and compile the model
        model = Model(img_input, output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.001),
            metrics=['acc']
        )
        return model

    def create_model_70(self):
        # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
        # the three color channels: R, G, and B
        img_input = layers.Input(shape=(150, 150, 3))

        # First convolution extracts 16 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(16, 3, activation='relu')(img_input)
        x = layers.MaxPooling2D(2)(x)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Third convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Flatten feature map to a 1-dim tensor so we can add fully connected layers
        x = layers.Flatten()(x)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = layers.Dense(512, activation='relu')(x)

        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(1, activation='sigmoid')(x)

        # Create model:
        # input = input feature map
        # output = input feature map + stacked convolution/maxpooling layers + fully 
        # connected layer + sigmoid output layer
        model = Model(img_input, output)
        model.compile(
            loss='binary_crossentropy', 
            optimizer=RMSprop(lr=0.001), 
            metrics=['acc']
        )
        
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
        print('classify object')
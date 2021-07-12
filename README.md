[image1]: assets/1.png 
[image2]: assets/2.png 
[image3]: assets/3.png 
[image4]: assets/4.png 
[image5]: assets/5.png 
[image6]: assets/6.png 
[image7]: assets/7.png 
[image8]: assets/8.png 
[image9]: assets/9.png 
[image10]: assets/10.png 
[image11]: assets/11.png 
[image12]: assets/12.png 
[image13]: assets/13.png 
[image14]: assets/14.png 
[image15]: assets/15.png 
[image16]: assets/16.png 
[image17]: assets/17.png 
[image18]: assets/18.png 
[image19]: assets/19.png 
[image20]: assets/20.png 
[image21]: assets/21.png 
[image22]: assets/22.png 
[image23]: assets/23.png 
[image24]: assets/24.png 
[image25]: assets/25.png 
[image26]: assets/26.png 
[image27]: assets/27.png 
[image28]: assets/28.png 
[image29]: assets/29.png 
[image30]: assets/31.png 


# Generative Adversarial Networks
Overview of Generative Adversarial Networks techniques.

## Content 
- [Essential GAN Theory](#esent_gan)
- [The Quick, Draw! Dataset](#quick_draw)
- [The Discriminator Network](#dis_net)
- [The Generator Network](#gen_net)
- [The Adversarial Network](#adver_net)
- [GAN Training](#gan_train)

- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)


# Essential GAN Theory <a id="esent_gan"></a>
Two Deep Learning networks in competition: 
- **Generator** vs. **Discriminator**
- **Generator**: generates imitations or counterfeits (of images). It tries to transform random noise as input into a real looking image.
- **Discriminator**: tries to distiguish between real and fake images

During Training: Both models compete against each other.
- Generator gets better and better in producing real looking images
- Discriminator gets better and better in distiguishing between real and fake images.

Goal:
- Generator creates images which will be identified as real images by the Discriminator

    ![image1]
    S.316, Abb 12-1

### Discriminator Training
- **Generator**: 
    - creates imitations of images.
    - only inferences (only feed forward, no backpropagation).
- **Discriminator**:
    - uses information from Generator.
    - **learns** to distinguish fake from real images.

    ![image2]
    S.317, Abb 12-2

### Generator Training
- **Discriminator**:
    - judges the imitations of the Generator.
    - only inferences (only feed forward, no backpropagation).
- **Generator**: 
    - uses information from Discriminator.
    - **learns** to deceive better and better the Discriminator.
    - Goal: Discriminator should classify fake images as real images.

    ![image3]
    S.318, Abb 12-3

### Discriminator Training in detail: ONLY THE DISCRIMINATOR WILL LEARN
- Generator produces fake images via inference (black). These fake images are shuffled with real images.
- Discriminator makes a prediction (y_hat), if the image is real.
- Loss calculation via cross entropy costs (comparison of prediction y_hat with label y).
- Via Backpropagation-Tuning of the Discriminator's parameters costs will be minimized. Hence, the Discriminator model should get better to distinguish between real and fake images.

### Generator Training in detail: ONLY THE GENERATOR WILL LEARN
- Generator receives a random noise vector **z** as input and creates a fake image as output.
- Those produced fake images will be passed on the Discriminator. IMPORTANT: The Generator lies to the Discriminator and declares its images with label=1 (real image).
- Discriminator judges via inference (black), if those images are real or fake.
- Loss calculation via cross entropy costs (comparison of prediction y_hat with label y) (green).
- Via Backpropagation-Tuning of the Generator's parameters costs will be minimized. Hence, the Generator model should get better and better to produce real looking images.

### At the end of the Training
- Discriminator and Generator were trained via an interplay and each network is fully optimized for its own task.
- We can discard the Discrimintor network now. The Generator network is our final product.
- We can put in random noise and will get a real looking image.
- We can put **special z-values** into the Generator (i.e. special coordinates in the latent room) to create images with special features (e.g. person with certain age, gender, person with glasses etc.)


## The Quick, Draw! Dataset <a id="quick_draw"></a> 
- 50 million drawings out of 345 categories
- GAN in this repo will be trained with category apple.
- Open Jupyter Notebook ```generative_adversarial_network.ipynb```
    ### Load Dependencies
    ```
    # for data input and output:
    import numpy as np
    import os

    # for deep learning: 
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Conv2D, Dropout
    from keras.layers import BatchNormalization, Flatten
    from keras.layers import Activation
    from keras.layers import Reshape # new! 
    from keras.layers import Conv2DTranspose, UpSampling2D # new! 
    from keras.optimizers import RMSprop # new! 

    # for plotting: 
    import pandas as pd
    from matplotlib import pyplot as plt
    %matplotlib inline
    ```
    ### Load data
    - data.shape with two dimensions, 1st dim: number of images, 2nd dim: number of image pixels (784)
    - divide data by 255 scale pixels between 0 and 1.
    - via reshape method 1x784 pixel will be transformed to 28x28.
    - let's store image width and height in img_w and img_h
    ```
    # from google.colab import drive
    # drive.mount('/content/gdrive')
    # os.chdir('/content/gdrive/My Drive/Colab Notebooks/quickdraw_data')
    input_images = "../quickdraw_data/apple.npy"
    data = np.load(input_images) # 28x28 (sound familiar?) grayscale bitmap in numpy .npy format; images are centered
    ```
    ```
    data.shape

    RESULTS:
    ------------
    (144722, 784)
    ```
    ```
    data[4242]

    RESULTS:
    ------------
    array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,  36,  79,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0, 134, 238,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0, 119, 254,   4,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0, 101, 255,  21,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  82, 255,  39,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  64,
        255,  57,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,  46, 255,  76,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,  28, 255,  94,   0,   2,  24,  44,   9,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,
            67, 135, 203, 253, 255, 255, 255, 245, 238, 253, 255, 255, 234,
        127,  19,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  53,
        185, 246, 255, 246, 185, 127, 119, 120, 251, 197, 136, 124,  98,
            84, 169, 252, 213,   8,   0,   0,   0,   0,   0,   0,   0,   0,
            47, 239, 222, 135,  67,   8,   0,   0,   0,   0, 201, 112,   0,
            0,   0,   0,   0,  78, 255,  65,   0,   0,   0,   0,   0,   0,
            0,   0, 197, 223,  25,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,  24, 255, 100,   0,   0,   0,   0,
            0,   0,   0,  11, 250, 123,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   1, 244, 134,   0,   0,
            0,   0,   0,   0,   0,  54, 255,  71,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,  12,   2,   0,   0, 211, 169,
            0,   0,   0,   0,   0,   0,   0,  58, 255, 137,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   1, 229, 125,   0,   0,
        176, 203,   0,   0,   0,   0,   0,   0,   0,  58, 255, 213,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  33, 255, 199,
            0,   0, 142, 237,   0,   0,   0,   0,   0,   0,   0,  51, 255,
        229,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  94,
        255, 251,  15,   0, 108, 255,  17,   0,   0,   0,   0,   0,   0,
            29, 255, 244,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0, 197, 238, 255,  78,   0,  80, 255,  47,   0,   0,   0,   0,
            0,   0,   6, 248, 255,   5,   0,   0,   0,   0,   0,   0,   0,
            0,   0,  48, 255, 107, 236, 178,   0, 138, 246,  12,   0,   0,
            0,   0,   0,   0,   0, 194, 255,  28,   0,   0,   0,   0,   0,
            0,   0,   0,   0, 154, 241,  12, 105, 255,  89, 223, 173,   0,
            0,   0,   0,   0,   0,   0,   0, 130, 255,  80,   0,   0,   0,
            0,   0,   0,   0,   0,  16, 244, 147,   0,   1, 192, 248, 255,
            88,   0,   0,   0,   0,   0,   0,   0,   0,  31, 243, 151,   0,
            0,   0,   0,   0,   0,   0,  35, 204, 251,  41,   0,   0,  21,
        162, 176,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0, 143,
        249,  39,   0,   0,   0,   0,   0,  88, 241, 232,  68,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,  21, 230, 232,  65,  19,  23, 111, 212, 255, 188,  24,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,  33, 216, 255, 255, 255, 254, 192,  90,   2,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,  16,  81, 105, 109,  35,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0], dtype=uint8)
    ```
    ```
    data = data/255
    data = np.reshape(data,(data.shape[0],28,28,1)) # fourth dimension is color
    img_w,img_h = data.shape[1:3]
    data.shape

    RESULTS:
    ------------
    (144722, 28, 28, 1)
    ```
    ```
    plt.imshow(data[4242,:,:,0], cmap='Greys')
    ```
    ![image4]

    Typical 'real' image

## The Discriminator Network <a id="dis_net"></a> 
- A simple Convolutional Network

    ![image5]
    S.324, Abb 12-7

- It contains Conv2D layers and Model class
    ```
    def build_discriminator(depth=64, p=0.4):

        # Define inputs
        image = Input((img_w,img_h,1))
        
        # Convolutional layers
        conv1 = Conv2D(depth*1, 5, strides=2, 
                    padding='same', activation='relu')(image)
        conv1 = Dropout(p)(conv1)
        
        conv2 = Conv2D(depth*2, 5, strides=2, 
                    padding='same', activation='relu')(conv1)
        conv2 = Dropout(p)(conv2)
        
        conv3 = Conv2D(depth*4, 5, strides=2, 
                    padding='same', activation='relu')(conv2)
        conv3 = Dropout(p)(conv3)
        
        conv4 = Conv2D(depth*8, 5, strides=1, 
                    padding='same', activation='relu')(conv3)
        conv4 = Flatten()(Dropout(p)(conv4))
        
        # Output layer
        prediction = Dense(1, activation='sigmoid')(conv4)
        
        # Model definition
        model = Model(inputs=image, outputs=prediction)
        
        return model
    ```
    ```
    discriminator = build_discriminator()
    ```
    ```
    discriminator.summary()

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 28, 28, 1)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        1664      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 7, 7, 128)         204928    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 4, 4, 256)         819456    
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 4, 4, 256)         0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 4, 4, 512)         3277312   
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 4, 4, 512)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8192)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 8193      
    =================================================================
    Total params: 4,311,553
    Trainable params: 4,311,553
    Non-trainable params: 0
    ```
    ### Compile the model
    ```
    discriminator.compile(loss='binary_crossentropy', 
                      optimizer=RMSprop(lr=0.0008, 
                                        decay=6e-8, 
                                        clipvalue=1.0), 
                      metrics=['accuracy'])
    ```
    ### Exlanation
    - input image size: 28x28
    - 4 hidden layers (Conv layers)
    - Number filters doubles from layer to layer: 64 --> 128 --> 256 --> 512
    - kernel size: constant at 5x5
    - Stride: 
        - for the first three Conv layer --> 2x2 --> width and height of activation map are cut in half by each Conv layer.
        - for the last Conv layer --> 1x1 --> no change in width and height of the activation map.
    - Dropout: On each Conv layer a Dropout of 40% will be applied.
    - Flatten: Flatten last three dimenasional Conv layer (preparation for the fully connected layer)
    - Output layer with binary classification --> fully connected layer has only one Sigmoid neuron as output.
    - We use binary crossentropy to evaluate the loss
    - Use RMSProp as an Optimizer. 
        - decay: hyper parameter to describe the decay rate of the learning rate
        - clipvalue: hyper parameter, sets a limit for the gradient (default: 1.0)

## The Generator Network <a id="gen_net"></a> 
- A more complex DeConvolutional Network

    ![image6]
    S. 326, Abb.12-8

- It contains UpSampling2D and Conv2DTranspose() methods 
    ```
    z_dimensions = 32

    def build_generator(latent_dim=z_dimensions, 
                        depth=64, p=0.4):
        
        # Define inputs
        noise = Input((latent_dim,))
        
        # First dense layer
        dense1 = Dense(7*7*depth)(noise)
        dense1 = BatchNormalization(momentum=0.9)(dense1) # default momentum for moving average is 0.99
        dense1 = Activation(activation='relu')(dense1)
        dense1 = Reshape((7,7,depth))(dense1)
        dense1 = Dropout(p)(dense1)
        
        # De-Convolutional layers
        conv1 = UpSampling2D()(dense1)
        conv1 = Conv2DTranspose(int(depth/2), 
                                kernel_size=5, padding='same', 
                                activation=None,)(conv1)
        conv1 = BatchNormalization(momentum=0.9)(conv1)
        conv1 = Activation(activation='relu')(conv1)
        
        conv2 = UpSampling2D()(conv1)
        conv2 = Conv2DTranspose(int(depth/4), 
                                kernel_size=5, padding='same', 
                                activation=None,)(conv2)
        conv2 = BatchNormalization(momentum=0.9)(conv2)
        conv2 = Activation(activation='relu')(conv2)
        
        conv3 = Conv2DTranspose(int(depth/8), 
                                kernel_size=5, padding='same', 
                                activation=None,)(conv2)
        conv3 = BatchNormalization(momentum=0.9)(conv3)
        conv3 = Activation(activation='relu')(conv3)

        # Output layer
        image = Conv2D(1, kernel_size=5, padding='same', 
                    activation='sigmoid')(conv3)

        # Model definition    
        model = Model(inputs=noise, outputs=image)
        
        return model
    ```
    ```
    generator = build_generator()
    ```
    ```
    generator.summary()

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         (None, 32)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 3136)              103488    
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 3136)              12544     
    _________________________________________________________________
    activation_1 (Activation)    (None, 3136)              0         
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 7, 7, 64)          0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 7, 7, 64)          0         
    _________________________________________________________________
    up_sampling2d_1 (UpSampling2 (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_transpose_1 (Conv2DTr (None, 14, 14, 32)        51232     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 14, 14, 32)        128       
    _________________________________________________________________
    activation_2 (Activation)    (None, 14, 14, 32)        0         
    _________________________________________________________________
    up_sampling2d_2 (UpSampling2 (None, 28, 28, 32)        0         
    _________________________________________________________________
    conv2d_transpose_2 (Conv2DTr (None, 28, 28, 16)        12816     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 28, 28, 16)        64        
    _________________________________________________________________
    activation_3 (Activation)    (None, 28, 28, 16)        0         
    _________________________________________________________________
    conv2d_transpose_3 (Conv2DTr (None, 28, 28, 8)         3208      
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 28, 28, 8)         32        
    _________________________________________________________________
    activation_4 (Activation)    (None, 28, 28, 8)         0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 28, 28, 1)         201       
    =================================================================
    Total params: 183,713
    Trainable params: 177,329
    Non-trainable params: 6,384
    ```
    ### Explanation
    - Let's use 32 noise dimensions. If higher noise vector can store more information (what could enhance quality of GAN). But it needs more computational effort.
    - First hidden layer is a fully connected layer with dimension 1x3136. 3136 neurons were chosen due to an easy transformation to 7x7x64.  
    - Reshape((7,7,depth)) transforms the noise vector into a 2D array which can be used by the Conv2DTranspose layers.
    - Conv2DTranspose are the opposite to Conv2D layers.
    - Instead of recognizing features and providing a feature map wich locates features in the image they take an activation map and sort features locally. 
    - There are three Conv2DTranspose layers. 1st layer has 32, 2nd has 16 and 3rd has 8 filter. 
    - Amount of filter reduces successively but size grows successively via Upsampling layers. Each time height and width of activation map will be doubled.
    - All three DeConvolutional layers have the following properties:
        - kernel size: 5x5
        - Stride: 1x1
        - Padding same (in order to keep the size of activation maps after DeConvolution)
        - ReLU activation function
        - BatchNormalization
    - Through multiple layers of deconvolution the noise input will be transformed to real looking images.
    - Output layer: A Conv2D layer which transforms the 28x28x8 activation maps to a single 28x28x1 image.
    - Sigmoid function ensures that pixel values are within the range 0 and 1 (like real images). 

## The Adversarial Network <a id="adver_net"></a> 
- Let's combine the Training processes for Discriminator and Generator

    ![image7]
    S.329, Abb 12-9

- **Discriminator Network** is already compiled and ready for Training.
- **Generator Network** must be compiled as an **Adversarial Network**
    ```
    z = Input(shape=(z_dimensions,))
    img = generator(z)

    discriminator.trainable = False

    pred = discriminator(img)

    adversarial_model = Model(z, pred)

    adversarial_model.compile(loss='binary_crossentropy', 
                          optimizer=RMSprop(lr=0.0004, 
                                            decay=3e-8, 
                                            clipvalue=1.0), 
                          metrics=['accuracy'])
    ```
    ### Explanation
    - Input() to define z as an array of noise values with length 32.
    - img: a 28x28 image
    - discriminator.trainable = False --> In case of Generator Training weights of Discriminator weights must be freezed 
    - pred = discriminator(img): Transfer the faked img image to the discrimnator. Save the result as pred.
    - adversarial_model: with the Model class of Keras we design the the Adversarial model.
    - CONSIDER: Discriminator network is only *not trainable* when it is part of the Adversarial Network.
## GAN Training <a id="gan_train"></a> 
- Let's focus on the function train() 
    ### Start Training
    ```
    def train(epochs=2000, batch=128, z_dim=z_dimensions):
    
        d_metrics = []
        a_metrics = []
        
        running_d_loss = 0
        running_d_acc = 0
        running_a_loss = 0
        running_a_acc = 0
        
        for i in range(epochs):
            
            # sample real images: 
            real_imgs = np.reshape(
                data[np.random.choice(data.shape[0],
                                    batch,
                                    replace=False)],
                (batch,28,28,1))
            
            # generate fake images: 
            fake_imgs = generator.predict(
                np.random.uniform(-1.0, 1.0, 
                                size=[batch, z_dim]))
            
            # concatenate images as discriminator inputs:
            x = np.concatenate((real_imgs,fake_imgs))
            
            # assign y labels for discriminator: 
            y = np.ones([2*batch,1])
            y[batch:,:] = 0
            
            # train discriminator: 
            d_metrics.append(
                discriminator.train_on_batch(x,y)
            )
            running_d_loss += d_metrics[-1][0]
            running_d_acc += d_metrics[-1][1]
            
            # adversarial net's noise input and "real" y: 
            noise = np.random.uniform(-1.0, 1.0, 
                                    size=[batch, z_dim])
            y = np.ones([batch,1])
            
            # train adversarial net: 
            a_metrics.append(
                adversarial_model.train_on_batch(noise,y)
            ) 
            running_a_loss += a_metrics[-1][0]
            running_a_acc += a_metrics[-1][1]
            
            # periodically print progress & fake images: 
            if (i+1)%100 == 0:

                print('Epoch #{}'.format(i))
                log_mesg = "%d: [D loss: %f, acc: %f]" % \
                (i, running_d_loss/i, running_d_acc/i)
                log_mesg = "%s  [A loss: %f, acc: %f]" % \
                (log_mesg, running_a_loss/i, running_a_acc/i)
                print(log_mesg)

                noise = np.random.uniform(-1.0, 1.0, 
                                        size=[16, z_dim])
                gen_imgs = generator.predict(noise)

                plt.figure(figsize=(5,5))

                for k in range(gen_imgs.shape[0]):
                    plt.subplot(4, 4, k+1)
                    plt.imshow(gen_imgs[k, :, :, 0], 
                            cmap='gray')
                    plt.axis('off')
                    
                plt.tight_layout()
                plt.show()
        
        return a_metrics, d_metrics 
    ```
    ```
    a_metrics_complete, d_metrics_complete = train()
    ```
    ### Explanation
    - d_metrics, a_metrics, runningd_loss, etc. are used for los and accuracy metrics.
    - for loop over epochs. In each epoch we choose a batch of 128 apple images
    - In each epoch we switch between Discriminator and Generator Training
    - In order to train the Discriminator:
        - We choose a batch of 128 real images
        - We generate 128 fake images. We construct noise vectors **z** uniformly distributed between [-1.0, 1.0]. We hand over those vectors to the predict method of the Generator. The Generator performs an inference and generates images without updatin its weights.
        - We concat the real with th fak images to a single Variable x. Varibale x is the input for the discriminator.
        - We create a label array y. Real images are set to 1 (first half of y), fake images to 0 (second half of y).
        - In order to train the Disciminator: hand over x and y to the train_on_batch method of the model (method from Model class)
        - After each epoch Discriminator loss and accuracy will be appended to d_metrics list.
    - In order to train the Generator:
        - We transfer random noise vectors (stored in variable noise) and a label y=1 (all images are real) to the train_on_batch method of the Adversarial modell. 
        - The Generator component of the Adversarial model transforms the noise input in faked images which will be sent automatically to the Discrimininator component.
        - As the Discriminator weights are frozen it only outputs its judgement if the provided images are real or fake. Via cross entropy loss and backpropagation the parameters of the Generator network will be updated. By minimizing the loss the Generator should learn to produce real looking images, which are mistakenly classified as real (y_hat=1) by the Discriminator.
        - After each epoch Adversarial loss and accuracy will be appended to a_metrics list.
    - After every 100 epochs:
        - We output the actual epoch.
        - We print a log message with loss and accuracy of the Discriminator and the Adversarial Network.
        - We choose randomly 16 noise vectors and use the predict method of the Generator in order to create fake images which are stored in gen_imgs.
        - We draw 16 fake images in a 4x4 pattern in order to observe the generator image quality during training.
    - At the end: We return the lists with the metrics of the Discriminator and the Adversarial models.

    ### Results
    - after 100 epochs:

        ![image8]

    - after 2000 epochs:

        ![image9]

    - Loss of Adversarial and Discriminator:

        ![image10]

    - Accuracy of Adversarial and Discriminator:

        ![image11]


## Setup Instructions <a id="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a id="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a id="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Generative-Adversarial-Networks.git

- Change Directory
```
$ cd Generative-Adversarial-Networks
```

- Create a new Python environment, e.g. gan. Inside Git Bash (Terminal) write:
```
$ conda create --id gan
```

- Activate the installed environment via
```
$ conda activate gan
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a id="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a id="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Important web sites - Deep Learning
* Deep Learning - illustriert - [GitHub Repo](https://github.com/the-deep-learners/deep-learning-illustrated)
* Jason Yosinski - [Visualize what kernels are doing](https://www.youtube.com/watch?v=AgkfIQ4IGaM)

Further Resources
* Read about the [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) model. Why train an A.I. to talk, when you can train it to sing ;)? In April 2017, researchers used a variant of the WaveNet model to generate songs. The original paper and demo can be found [here](https://arxiv.org/pdf/1609.03499.pdf).
* Learn about CNNs [for text classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/). You might like to sign up for the author's [Deep Learning Newsletter!](https://www.getrevue.co/profile/wildml)
* Read about Facebook's novel [CNN approach for language translation](https://engineering.fb.com/2017/05/09/ml-applications/a-novel-approach-to-neural-machine-translation/) that achieves state-of-the-art accuracy at nine times the speed of RNN models.
* Play [Atari games with a CNN and reinforcement learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning). If you would like to play around with some beginner code (for deep reinforcement learning), you're encouraged to check out Andrej Karpathy's [post](http://karpathy.github.io/2016/05/31/rl/).
* Play [pictionary](https://quickdraw.withgoogle.com/#) with a CNN! Also check out all of the other cool implementations on the [A.I. Experiments](https://experiments.withgoogle.com/collection/ai) website. Be sure not to miss [AutoDraw](https://www.autodraw.com/)!
* Read more about [AlphaGo]. Check out [this article](https://www.technologyreview.com/2017/04/28/106009/finding-solace-in-defeat-by-artificial-intelligence/), which asks the question: If mastering Go “requires human intuition,” what is it like to have a piece of one’s humanity challenged?
* Check out these really cool videos with drones that are powered by CNNs.
    - Here's an interview with a startup - [Intelligent Flying Machines (IFM)](https://www.youtube.com/watch?v=AMDiR61f86Y).
    - Outdoor autonomous navigation is typically accomplished through the use of the [global positioning system (GPS)](www.droneomega.com/gps-drone-navigation-works/), but here's a demo with a CNN-powered [autonomous drone](https://www.youtube.com/watch?v=wSFYOw4VIYY).

* If you're excited about using CNNs in self-driving cars, you're encouraged to check out:
    - Udacity [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013), where we classify signs in the German Traffic Sign dataset in this project.
    - Udacity [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t), where we classify house numbers from the Street View House Numbers dataset in this project.
    - This series of blog posts that details how to train a CNN in Python to produce a self-driving A.I. to play Grand Theft Auto V.

* Check out some additional applications not mentioned in the video.
    - Some of the world's most famous paintings have been [turned into 3D](https://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1) for the visually impaired. Although the article does not mention how this was done, we note that it is possible to use a CNN to [predict depth](https://cs.nyu.edu/~deigen/depth/) from a single image.
    - Check out [this research](https://ai.googleblog.com/2017/03/assisting-pathologists-in-detecting.html) that uses CNNs to localize breast cancer.
    - CNNs are used to [save endangered species](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)!
    - An app called [FaceApp](https://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/) uses a CNN to make you smile in a picture or change genders.


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


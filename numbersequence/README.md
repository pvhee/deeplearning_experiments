# Number Sequence Classifier
> *Objective*: Build a live camera app that can interpret number strings in real-world images.

This is Udacity's final assignment (Lesson 7) for the Deep Learning course.

### Step 1: construct number sequence data set from notMNIST

First we generate a new data set by concatenating notMNIST letters together into sequences. This artificially constructed test set will allow us to model our network without having to deal with messy data in our initial architecture.

    python -m numbersequence.notmnist_sequence
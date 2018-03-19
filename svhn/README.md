# SVHN Number Classifier
> *Objective*: Classify Street View House Numbers

This is part of Udacity's final assignment (Lesson 7) for the Deep Learning course.

### Dataset

Dataset: http://ufldl.stanford.edu/housenumbers/

This is split up as following:

```
('Training set', (30061, 64, 64, 3), (30061, 6, 11))
('Validation set', (3341, 64, 64, 3), (3341, 6, 11))
('Test set', (13068, 64, 64, 3), (13068, 6, 11))
```

### ConvNet training

To train - or rerun a trained network on a saved model - run

    python -m svhn.model
   
The model is saved automatically to file after training (check MODEL_FILE for filename)
and you reuse this model for predicting a random batch of our test data by changing the value of LOAD_MODEL_FLAG
    
### ConvNet predictions
    
To evaluate a list of sample jpg images (64x64x3), run

    python -m svhn.predict 
    
Note that we took a sample list of images from pictures and resized those manually via https://www.imgonline.com.ua/eng/resize-image-result.php

### Google Cloud ML Predictions

To use Google Cloud ML, we first need to export our Keras model to an Estimator, then save this into a TF SavedModel.

    python -m svhn.export_model
    
This will export an inference graph (including input & output mappings) to a time-stamped directory in `export/`

Upload this directory to a Google Cloud bucket and create a model and version for this. We've listed the steps in the following script

    cd svhn
    ./gcloud_export.sh
    
You need to adapt `EXPORT_VERSION` and `MODEL_VERSION` every time you want to export and upload a new model.

We now need to make sure our input (a 64x64x3 image) is formatted as a JSON file. Run the following script to create such a file

    python -m svhn.input_data_gcloud
   
To test this, we invoke the model on Google Cloud ML directly

     gcloud ml-engine predict --model svhn_digits --version v7 --json-instances data.json

Let's now make this prediction from Python. First, we need a service account to authenticate against Cloud ML.

    export GOOGLE_APPLICATION_CREDENTIALS='....json'

We can then call a script to make a prediction.

    python -m svhn.predict_gcloud

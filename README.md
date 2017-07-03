# Udacity Deep Learning exercises
> See https://www.udacity.com/course/deep-learning--ud730

## Installation & Usage

Run [using Docker](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity), like

	docker run -p 8888:8888 --name tensorflow-udacity -it -v ~/code/udacity/udacity_deeplearning:/notebooks gcr.io/tensorflow/udacity-assignments:1.0.0

Note that we mounted our local directory into the Docker image so that we can save progress and commit this back to github.

You can later return to it using:

	docker start -ai tensorflow-udacity

Then access (on Mac) via

	http://0.0.0.0:8888

## Standalone packages

We've implemented a standalone notMNIST parser for use in e.g. google cloud ML in `trainer`.

To run this, follow [the Google Cloud ML tutorial](https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction), then test locally  using

	gcloud ml-engine local train --module-name trainer.task --package-path trainer -- --train-file notMNIST.pickle

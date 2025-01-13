# Singularity PyTorch Project

This project provides a Singularity container for an image processing application built using PyTorch. The application fetches images from a database, preprocesses them, and trains a convolutional neural network (CNN) to classify the images.

## Project Structure

```
singularity-pytorch-project
├── singularity.def
├── imageProcessor.py
└── README.md
```

- **singularity.def**: Defines the Singularity container configuration, including the base image and environment setup for Python and PyTorch.
- **imageProcessor.py**: Contains the implementation of the image processing application, including data fetching, preprocessing, model definition, training, and evaluation.

## Requirements

- Singularity
- Python 3.x
- PyTorch
- SQLite
- PIL
- NumPy

## Building the Singularity Container

To build the Singularity container, navigate to the project directory and run the following command:

```
sudo singularity build singularity-pytorch-project.sif singularity.def
```

## Running the Application

Once the container is built, you can run the image processing application using the following command:

```
singularity exec singularity-pytorch-project.sif python imageProcessor.py
```

Make sure to update the database path in `imageProcessor.py` to point to your SQLite database containing the images and labels.

## Usage

The application will fetch images from the specified database, preprocess them, and train a CNN model. After training, it will evaluate the model's performance and print the accuracy.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
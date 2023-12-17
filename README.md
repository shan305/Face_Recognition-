# Face Recognition Python Project

## Overview

This Python script utilizes OpenCV and TensorFlow/Keras to perform face recognition on video frames. The script loads a pre-trained model, detects faces in the frames, and recognizes them using the trained model.

## Features

- **Modular Structure:** The script is modularized for better organization and maintainability.

- **Model Training:** The script includes a function to train a face recognition model using Convolutional Neural Networks (CNNs).

- **Video Processing:** The script processes video frames, detects faces, and makes predictions on the recognition status of each face.

- **Output Visualization:** Recognized faces are displayed in the output frames with rectangles drawn around them.

- **Output Video Export:** The script exports an output video with recognized faces highlighted.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- OpenCV
- NumPy
- TensorFlow
- Matplotlib
-mageio[ffmpeg]
You can install the dependencies using the following command:

```bash
pip install opencv-python numpy tensorflow matplotlib
pip install 'imageio[ffmpeg]'
```

## Usage

1. **Clone Repository:**

   ```bash
   git clone https://github.com/shan305/Face_Recognition-.git
   cd face-recognition-script
   ```

2. **Run the Script:**

   ```bash
   python face_recognition_script.py
   ```

   Make sure to replace `face_recognition_script.py` with the actual filename if it's different.

3. **Input Video:**

   Provide the path to the input video when prompted.

4. **Output Video:**

   The output video with recognized faces will be saved in the specified output folder.

## Configuration

Adjust the following parameters in the script according to your requirements:

- `video_path`: Path to the input video file.
- `output_directory`: Path to the directory where the output video will be saved.
- `output_filename`: Name of the output video file.
- `model_path`: Path to save/load the trained face recognition model.


## Considertaion

Here is how to address overfitting and improve generalization:

**Validation Set:** Split dataset into training and validation sets. Train the model on the training set and monitor its performance on the validation set. If the model performs well on the training set but poorly on the validation set, it might be overfitting.

**Data Augmentation:** Apply data augmentation techniques to artificially increase the diversity of training set. This can include random rotations, flips, and zooms on the training images.

**Reduce Model Complexity:** If overfitting persists, consider reducing the complexity of your model. This could involve reducing the number of layers, neurons, or adding dropout layers to introduce regularization.

**Learning Rate:** Experiment with adjusting the learning rate. Too high of a learning rate may cause the model to converge too quickly, potentially overshooting the optimal weights.

**Evaluate on Test Data:** After training, evaluate the model on a separate test dataset that it has never seen before. This provides a more realistic estimate of the model's performance on new, unseen data.

**Note** finding the right balance often involves experimentation and tuning hyperparameters. If the model is still not performing as expected, we need to further investigate data and training process.




## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)


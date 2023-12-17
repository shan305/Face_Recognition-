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

You can install the dependencies using the following command:

```bash
pip install opencv-python numpy tensorflow matplotlib
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

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)


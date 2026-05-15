# Webcam Background Control with ML

This project is a desktop computer vision application built with Python, OpenCV, and Tkinter. The main focus is the webcam-based machine learning workflow used to process camera input in real time. A lightweight Ollama-powered language layer is included only as a command interface for changing backgrounds and colors.

## Features

- Live webcam capture.
- Real-time camera frame processing.
- Tkinter desktop interface.
- Background switching using local image assets.
- Solid color background generation.
- Natural-language command interpretation through Ollama.

## How to get the weights

The trained weights are not stored in this repository because they are too large for GitHub. To create them locally, run:

```bash
python finetune.py
```

after collecting your training data with the data collection script.

If you prefer not to retrain, you can also download the `.pth` file from this Google Drive link: https://drive.google.com/drive/folders/19gRvjIRMqryVBl0cpzZiYt3IvBnNO-Wr?usp=drive_link

## How to get the dataset

The main training dataset is hosted separately on Kaggle. Download it from Kaggle and place it in the correct dataset folder before training.

Personal frames are collected locally by running:

```bash
python autocollect.py
```

This script captures webcam frames for the custom training set.

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install Ollama on your machine and make sure it is running.

3. Start the application:

```bash
python main.py
```

## How it works

The application captures webcam frames and processes them in real time. The ML pipeline handles the camera-based functionality, while the command layer maps user input to actions such as switching an image background, applying a solid color background, or resetting the view.

If the user enters a place name or scene, the app chooses the best matching available background image. If the user enters a color name, the app uses that to generate a solid color background.

## Project structure

```text
project-folder/
├── main.py
├── llm_controller.py
├── finetune.py
├── autocollect.py
├── background.py
├── collect_data.py
├── segmentation.py
├── requirements.txt
├── README.md
├── backgrounds/
│   ├── tokyo.jpg
│   ├── rome.jpg
│   ├── timisoara.jpg
│   └── ...
└── dataset/
    └── ...
```

## Troubleshooting

- If the webcam does not open, check that no other app is using it.
- If Ollama is not responding, make sure the Ollama service is running.
- If a command returns `unknown`, try a clearer color or place name.
- If a background image does not switch, verify the file exists in the backgrounds folder.
- If colors look wrong, check that OpenCV BGR ordering is handled correctly.

## Privacy note

Any personal webcam training images were removed before publishing the repository. Only the code, documentation, and safe demo assets are included in the public project.

## Future improvements

- Better model accuracy.
- Faster or more stable webcam inference.
- Optional export of training results.
- Browser integration for use in online meetings / calls
- A better dataset for training
- A cleaner GUI
- Integration into a simple browser extension

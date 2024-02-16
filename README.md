
# Implementing an invisibility cloak using YOLO 8 in Computer Vision

In this repo we will learn how to create our own invisibility cloak using YOLO8 library. This project takes advantes of a field called Computer Vision that allows us to play with images, pre-proceses and large trained models.

We need to have already installed the following technical requirements in our env:
- Python >= 3.11.3
- pip >= 23.1.2



## Installing minimun  technical requirements:

**Installing Python3.1.5 in our system:**

Check Your Current Python Version:
```bash
  python3 --version
```

Install Dependencies: 
```bash
  sudo apt update
  sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
```

Download Python Source Code:
```bash
  wget https://www.python.org/ftp/python/3.1.5/Python-3.1.5.tgz
```

Extract Source Code:
```bash
  tar -xvf Python-3.1.5.tgz
```

Configure and Compile:
```bash
  cd Python-3.1.5
  ./configure
```
Compile Python
```bash
  make
```
Install Python: 
```bash
  sudo make install
```
Verify Installation: 
```bash
  python3.1 --version
```
This command should display the Python version you installed.

**Installing pip >= 23.1.2 in our system**

Install Dependencies: 
```bash
  sudo apt update
  sudo apt install python3-pip
```

Upgrade pip: After installing pip, we can upgrade it to the desired version using pip itself.

```bash
  pip install --upgrade pip==23.1.2
```

Verify Installation: 
```bash
  pip --version
```
This command will display the pip version installed on your system.

**Installing ultralytics in our env:**

Install with pip:
```bash
  pip install yolov5
```

Check Installation:
```bash
  print("YOLOv8 is installed.") if __import__('yolov5', globals(), locals(), [], 0) else print("YOLOv8 is not installed.")
```
This script will print out "YOLOv8 is installed" if the YOLOv8 library was installed right.

## Project structure:

Our project is bult in the following folder structure:

- [inivisibility-cloak]
  - [assets]
  - [models]
  - [snippets]

**inivisibility-cloak:** main folder of our application, contains our main.py file.

**assets:** This folder contains our .JPG, .PNG files. These files are going to be use for testing our inivisibility cloak. 

**models:** This folder locates our main model already trained.

**snippets:** This snippets can help us to undertand how our Machine Learning code works with some example such as segmentation and extraction.


## How our main.py file actually works:

Imports the OpenCV library, which is used for image processing tasks.
```bash
  import cv2
```

Imports the YOLO object detection model from the Ultralytics library.
```bash
  from ultralytics import YOLO
```

Imports the NumPy library, which is used for numerical operations.
```bash
  import numpy as np
```

Imports the operating system module, which is used for file operations.
```bash
  import os:
```


Defines a function load_file that takes two arguments: route (file path) and error_msg (error message to display if file does not exist).

```bash
  def load_file(route, error_msg):
    if os.path.exists(route):
        return cv2.resize(cv2.imread(route), (capHeight, capWidth))
    else:
        print(f"Error: {error_msg} '{route}' does not exist")
        return None
```

- Checks if the file exists at the given path using os.path.exists().

- If the file exists, it loads the image using cv2.imread(), resizes it using cv2.resize(), and returns the resized image.

- If the file does not exist, it prints an error message and returns None.

Initializes a video capture object cap using cv2.VideoCapture(0) to capture video from the default camera (webcam).

```bash
  cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        exit()
    else:
        capWidth, capHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.  CAP_PROP_FRAME_HEIGHT))
```

- Checks if the capture device is opened successfully using cap.isOpened().

- Retrieves the frame width and height from the capture device using cap.get(cv2.CAP_PROP_FRAME_WIDTH) and cap.get(cv2.CAP_PROP_FRAME_HEIGHT).

Loads a background image from the file './assets/background.jpg' using the load_file function.

```bash
  background = load_file('./assets/background.jpg', 'Background file.')
  model = YOLO('./models/20231117_best.pt') if os.path.exists('./models/20231117_best.pt') else None
```
- Checks if a YOLO object detection model file './models/20231117_best.pt' exists, and if so, initializes the YOLO model using Ultralytics library's YOLO class.

Enters an infinite loop (while True) to continuously process video frames from the webcam.
```bash
  while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=capWidth, conf=0.78) if model else None
    masks = results[0].masks if results else None
    poligon = np.zeros((capWidth, capHeight), dtype="uint8") if masks is None else (masks.data[0].cpu().numpy().astype("uint8")*255)
    poligon = cv2.resize(poligon, (capHeight, capWidth))
    background_masked = cv2.bitwise_and(background, background, mask=poligon) if background is not None else None
    frame = cv2.resize(frame, (capHeight, capWidth))
    final_result = cv2.add(frame, background_masked) if background_masked is not None else frame

    cv2.imshow('Invisibility Cloak', final_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
- Reads a frame from the capture device using cap.read().
- If the frame is read successfully:
    - Uses the YOLO model to predict objects in the frame if the model is available.
    - Extracts masks from the prediction results.
    - Creates a binary mask image (poligon) based on the predicted masks.
    - Applies bitwise AND operation to mask the background image with the binary mask    (background_masked).
    - Adds the masked background to the original frame to create the final result (final_result).
    - Displays the final result in a window titled 'Invisibility Cloak' using cv2.imshow().
    - Checks for the 'q' key press to exit the loop (if cv2.waitKey(1) & 0xFF == ord('q')).

Cleanup
```bash
  cap.release()
  cv2.destroyAllWindows()
```


## Let's run our app:

Navigate to the main folder:
```bash
  cd ./inivisibility-cloak
```

Run our main.py file using our python instalation:
```bash
  python main.py
```

## How to use our app:

Inside our assets folder, there are two main assets:
- background.jpg
- red_cloak.jpg

These files are our main assets across the application, these names **MUST ALWAYS REMAIN** the same and **NEVER** be changed. However we could change what is inside then.

**background.jpg** is an image that we need to adapt to what our background will be. This image needs to be as clear as possible and will always remain the same trought out our tests. 
**red_cloak.jpg** is what our model is going to detect and extract out of the image and replace by the background.

After running our python command to start the application a new video windows will show up, remember always be pointing the camera to the same background uploaded to the assets folder. 

Our red cloak will allow us to hide inside, so it will be replaced by our background.

After setting up our camara and running our application, we can start playing with our Invisibility cloak! Enjoy!

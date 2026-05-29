# TFM - Ruben Fernandez Gonzalez
This is a repository for the implementation of an intelligent system for the detection, identification, and analysis of vehicles in real-world environments (on-board cameras and fixed points such as toll booths) using Deep Learning. The system will integrate computer vision and deep learning techniques to detect and identify vehicles in real-time video.


## UV Installation

The required packages will be controlled by the Python package manager UV, which can be installed via *pip* if installed. Windows have a different method of installation than macOS and Linux, which share the same one (as well as the commands):

<details>
    <summary>pip installation</summary>
 
### Open a new terminal, copy and paste the next command:
```bash
pip install uv
```
</details>


<details>
    <summary>macOS and Linux installation</summary>

### Open a new terminal, copy and paste the next command:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
</details>


<details>
    <summary>Windows installation</summary>

### Open a new terminal, copy and paste the next command:   
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
</details>



## UV Environment creation and activation

After UV have finished installing, we proceed to create a new environment. This project packages will be detached from other environment or projects on the computer. You can use any name on the environment, here it will be used "muriap".
```bash
uv venv muriap --python 3.12
```

The environment is activated through different commands depending on the OS:
<details>
    <summary>macOS and Linux environment activation</summary>
 
### Copy and paste the next command:
```bash
source .muriap/bin/activate
```
</details>


<details>
    <summary>Windows environment activation</summary>
 
### Copy and paste the next command:
```bash
.\muriap\Scripts\activate
```
</details>



## Package installation

The Python packages needed for the programs to work are gonna be installed from the file named requirements.txt. ADDITIONAL_NOTE: If this command gives an error on the torch library, execute the next command with your specific version of CUDA:
```bash
uv pip install -r requirements.txt
```

NOTE: The environment (and the packages installed) has been tested using CUDA 13.2 so if you are using another version consider reinstalling PyTorch with the following command (and changing the CUDA version):
```bash
uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132 
```


___


## Dataset download and structurize for model training

### Vehicle datasets

#### BDD-100K dataset

The dataset can be download manually from the following URL: [BDD100K](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) or following the next instructions to use the Kaggle CLI:

1. First, sign up on Kaggle's [website](https://www.kaggle.com/account/login): 

2. After login, you can download your Kaggle API credentials at https://www.kaggle.com/settings/api by clicking on the "Generate New Token" button under the "API" section.

3. Copy the API token (first alphanumeric sequence) and export it via terminal: 

```bash
export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx
```

**ADDITIONAL NOTE: If you encounter any error such as KeyError: 'username', execute the next command with your kaggle username and the API token generated previously:**
```bash
echo '{"username":"your_username","key":"your_apitoken"}' > ~/.kaggle/kaggle.json
```

4. Download the dataset:
```bash
uv run kaggle datasets download solesensei/solesensei_bdd100k -p datasets --unzip -o -q
```

As there is no need for image segmentation, the folder bdd100k_seg can be deleted.
```bash
rmdir -rf bdd100k_seg
```

5. Execute the script to convert it to yolo format:
```bash
uv run .\scripts\bdd100k.py
```

6. To convert it to COCO format, the library **yolococo** exports everything to a .json file so it can be used to train models with this format.
```bash
uv run yolococo yolo2coco --images datasets/bdd100k_yolo/train/images/ --labels datasets/bdd100k_yolo/train/labels --classes classes.txt  --bbox-round 3 --file-name-mode name --out datasets/bdd100k_yolo/train/train.json

uv run yolococo yolo2coco --images datasets/bdd100k_yolo/val/images --labels datasets/bdd100k_yolo/val/labels --classes classes.txt  --bbox-round 3 --file-name-mode name --out datasets/bdd100k_yolo/val/val.json
```


After all of these changes, the complete dataset should look like these:

<pre>
    ├── README.md
    └── datasets
        └── bdd100k_yolo
            ├── train
                └── train.json
            ├── val
                └── val.json
            └── yolo.yaml
    └── models
</pre>



### License plate datasets

#### UC3M-LP

1. Enter the following link and download the dataset: https://edatos.consorciomadrono.es/dataset.xhtml?persistentId=doi:10.21950/OS5W4Z. The downloaded file, **UC3M-LP.zip**, must be extracted on the datasets folder.
```bash
unzip UC3M-LP.zip
```

2. Due to the fact that there is a premade script already done, we can use it for the YOLO training. We start cloning the dataset:
```bash
git clone https://github.com/ramajoballester/UC3M-LP.git
cd UC3M-LP
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the script to transform the dataset to YOLO format. It will create 2 versions of the dataset, one for LP detection from the whole image and another one for LP recognition from the cropped LP region. The script will resize the images to the specified dimensions and save the resulting images and labels in in new directories. You can specify the desired dimensions for the images as arguments of the script.
```bash
python3 scripts/labels2yolo.py . 640 320
```


___


## Tesseract-OCR installation

Although the package for using tesseract is installed by the package manager UV, this OCR still needs to be installed 
The required packages will be controlled by the Python package manager UV, which can be installed via *pip* if installed. Windows have a different method of installation than macOS and Linux, which share the same one (as well as the commands).

<details>
    <summary>macOS installation</summary>
 
### Open a new terminal, copy and paste the next command:
```bash
brew install tesseract
```

### Download .traineddata files for non-English languages (optional, depends on the language of the license plates):
```bash
brew install tesseract-lang
```

### Test installation:
```bash
tesseract --version
```
</details>


<details>
    <summary>macOS installation</summary>
 
### Open a new terminal, copy and paste the next command:
```bash
sudo apt install tesseract-ocr
```

### Download .traineddata files for non-English languages (optional, depends on the language of the license plates):
```bash
sudo apt install tesseract-ocr-all
```

### Test installation:
```bash
tesseract --version
```
</details>


<details>
    <summary>Windows installation</summary>

### UB Mannheim installation
Tesseract cannot be natively compiled easily on Windows, but the official pre-compiled binaries can be downloaded and installed from the UB Mannheim University project. Navigate to the following url and install the .exe: https://github.com/UB-Mannheim/tesseract/wiki. There are also older versions for 32 and 64 bit Windows available. 
</details>


___


## Deploy web interface 

Run the application with the following command:
```bash
uv run streamlit run .\web_app\st_yolo.py   
```

NOTE: If there are errors in several elements like **TypeError: Failed to fetch dynamically imported module** clear the cache of Streamlit with the following command:
```bash
uv run streamlit cache clear
```

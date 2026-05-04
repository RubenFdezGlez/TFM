# MURIA-Practicas
This is a repository for the implementation of an intelligent system for the detection, identification, and analysis of vehicles in real-world environments (on-board cameras and fixed points such as toll booths) using Deep Learning. The system will integrate computer vision and deep learning techniques to detect and identify vehicles in real-time video.


## Installation

### UV installation
The required packages will be controlled by the Python package manager UV, which can be installed via *pip* if installed. Windows have a different method of installation than macOS and Linux, which share the same one (as well as the commands).
<details>
 <summary>pip installation</summary>
 
 #### Open a new terminal, copy and paste the next command:
 ```bash
     pip install uv
 ```
</details>

<details>
 <summary>macOS and Linux installation</summary>
 
  #### Open a new terminal, copy and paste the next command:
 ```bash
     wget -qO- https://astral.sh/uv/install.sh | sh
 ```
</details>

<details>
 <summary>Windows installation</summary>

 #### Open a new terminal, copy and paste the next command:   
 ```bash
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
 ```
</details>

### Environment creation and activation

After UV have finished installing, we proceed to create a new environment. This project packages will be detached from other environment or projects on the computer. You can use any name on the environment, here it will be used "muriap".
```bash
    uv venv muriap
```

The environment is activated through different commands depending on the OS:
<details>
 <summary>macOS and Linux environment activation</summary>
 
 #### Copy and paste the next command:
 ```bash
     source .muriap/bin/activate
 ```
</details>

<details>
 <summary>Windows environment activation</summary>
 
 #### Copy and paste the next command:
 ```bash
     .\muriap\Scripts\activate
 ```
</details>


### Package installation

The Python packages needed for the programs to work are gonna be installed from the file named requirements.txt.
```bash
    uv pip install -r requirements.txt
```

### Dataset download for model training

The dataset can be download manually from the following URL: [BDD100K](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) or following the next instructions to use the Kaggle CLI:

1. First, sign up on Kaggle's [website](https://www.kaggle.com/account/login): 

2. After login, you can download your Kaggle API credentials at https://www.kaggle.com/settings/api by clicking on the "Generate New Token" button under the "API" section.

3. Copy the API token (first alphanumeric sequence) and export it via terminal: 

```bash
    export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx
```

As there is no need for image segmentation, the folder bdd100k_seg can be deleted.
```bash
    rmdir -rf bdd100k_seg
```

After finishing downloading, the dataset should follow the next structure:

<pre>
    ├── bdd100k
    ├── bdd100k_labels_release
    ├── README.md
    ├── yolo26n.py
    ├── st_yolo.py
    ├── README.md
    └── datasets
        └── train1.yaml
    └── models
        └── best.pt
</pre>


5. ### Deploy web interface 

Run the application with the following command:
```bash
    uv run streamlit run st_yolo.py   
```
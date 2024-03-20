# Event Extraction in Vietnamese text

This project focuses on implementing deep learning methods for Event Extraction tasks in Vietnamese text. The goal is to analyze Vietnamese sentences and identify triggers (words or phrases strongly expressing an event) along with corresponding arguments (words or phrases representing objects involved in the event). The project utilizes state-of-the-art methods applied to the [ACE 2005 dataset](https://catalog.ldc.upenn.edu/LDC2006T06), including:

- A pipeline module based on CNN ([Nguyen et al., 2015](https://aclanthology.org/P15-2060/))
- A pipeline module based on BiLSTM
- A pipeline module based on PhoBERT by adapting it to the Machine Reading Comprehension task ([Liu et al., 2021](https://aclanthology.org/2020.emnlp-main.128/))
- A joint module based on BiLSTM ([Nguyen et al., 2016](https://aclanthology.org/N16-1034/))

# How to run


To run the project, ensure you have Python 3.7 installed. Create a conda environment using the provided command:

```
conda create -n ee python=3.7
conda activate ee
```

Install all dependencies listed in the ```requirements.txt``` file:
```
pip install -r requirements.txt
```

Additionally, you'll need to download two weight files for the pipeline module based on PhoBERT from the provided URLs and place them in the ```weight``` folder.
- PhoBERT for Trigger Detection: https://drive.google.com/file/d/1mMfm9VBuMQ_8G9Jbx6JIZYKu70uzKha7/view?usp=sharing
- PhoBERT for Argument Detection: https://drive.google.com/file/d/1zMv5wb9XKW3gW_K6yIfrguSKlZqrs7B5/view?usp=sharing


After setting up the environment and downloading the necessary files, run the application using Streamlit package by executing the command:

```
streamlit run main.py
```

This will launch the application, and a URL (http://localhost:8501/) will appear in the terminal. Open this URL in your browser to use the application.

# Example

![alt text](https://github.com/tqgminh/event_extraction/blob/main/img/example.png?raw=true)

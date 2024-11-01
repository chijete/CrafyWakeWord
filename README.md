# CrafyWakeWord
CrafyWakeWord it's a library focused on AI-based wake word recognition.

# NEW VERSION ⚠️🚀
We have launched CrafyWakeWord2: the new version of CrafyWakeWord that offers better results.

Link: https://github.com/chijete/CrafyWakeWord2

## ⭐ Features and functions
- Custom wake word recognition.
- Multiple language support.
- Models portable to other platforms.
- TensorFlow and TensorFlow.js supported.
- Step by step explanation.

## Demo
You can see an online demo here:
- English demo: https://chijete.github.io/CrafyWakeWord_demo/en/
- Spanish demo: https://chijete.github.io/CrafyWakeWord_demo/es/

## Use pre-trained models
You can download pre-trained models in multiple languages ​​from this repository: https://github.com/chijete/CrafyWakeWord_models

**Tip:** If you want to download a single model and not clone the entire repository, you can use this tool to download a single folder from a git repository: https://download-directory.github.io/

## Create your own model
With this tool you can create your custom wake word detection model.
For example, you can create a model to detect when the user says the word "banana", and then run your own code accordingly.

### Prerequisites
- Have [Python](https://www.python.org/downloads/ "Python") 3 installed.
- Have [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/ "Miniconda") or [Anaconda](https://www.anaconda.com/download# "Anaconda") installed.
- Have a verified Google Cloud account (we will use the [Google Cloud Text-to-Speech API](https://console.cloud.google.com/apis/library/texttospeech.googleapis.com "Google Cloud Text-to-Speech API") to improve the dataset, more information below; the free plan is enough).

### 1. Download voice dataset
The first step is to obtain a dataset of transcribed audios.
In this library we will use Mozilla Common Voice to obtain the dataset.

Follow these steps:
1. Access to https://commonvoice.mozilla.org/en/datasets
2. Select the target language from the Language selector.
3. Select the last "Common Voice Corpus" version (Do not select "Delta Segment").
4. Enter an email, accept the terms and download the file.

### 2. Clone this repository
1. Clone this repository to a folder on your computer using [git](https://git-scm.com/ "git"), or download and unzip this repository using Github's "Code > Download ZIP" option.
2. Unzip the downloaded Mozilla Common Voice file and copy the "cv-corpus-..." folder to the folder where you cloned the repository.

### 3. Install dependencies
Run this commands in your terminal (conda activate first) or Anaconda terminal:
- `pip install librosa textgrid torchsummary ffmpeg-python pocketsphinx fastprogress chardet PyAudio clang pgvector hdbscan initdb speechbrain`
- `pip install --upgrade google-cloud-texttospeech`
- `pip install --only-binary :all: pynini` or `conda install conda-forge::pynini`
- `conda install -c conda-forge kalpy`
- `pip install montreal-forced-aligner`
- `conda install -c conda-forge sox`
- `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` (Installing PyTorch with CPU, complete instructions on https://pytorch.org/get-started/locally/ - you can use GPU acceleration).
- `pip install ffmpeg onnx tensorflow numpy onnx_tf tensorflow_probability ipython`

Install PostgreSQL from https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
When the installation is finished, add PostgreSQL to the System Path:

**In Windows:**
1. Open the Windows Control Panel.
2. Click on "System and Security".
3. Select "System".
4. In the "Advanced system settings" window, click on the "Environment Variables" button under the "Advanced" tab.
5. In the "System variables" section, look for the "Path" variable and click "Edit...".
6. Add the path to the PostgreSQL directory to the end of the list. For example, the path might be something like `"C:\Program Files\PostgreSQL\version\bin"` (replace "version" with the version of PostgreSQL you installed).

**When finished, close the terminal and reopen it to apply the changes.**

### 4. Download aligner model
We will use [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/ "Montreal Forced Aligner") to align the audio files from the Mozilla Common Voice dataset.
Follow these steps:
1. Search for an Acoustic model for your model's target language here: https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
2. On the Acoustic model details page, in the Installation section, click "download from the release page".
3. At the bottom of the page on Github, in the Assets section, click on the zip file (the first one in the list) to download it.
4. Return to the Acoustic model page, and in the Pronunciation dictionaries section, click on the first one in the list.
5. On the Pronunciation dictionary details page, in the Installation section, click "download from the release page".
6. At the bottom of the page on Github, in the Assets section, click on the dict file (the first one in the list) to download it.
7. Copy the two downloaded files to the `mfa` folder within the directory where you cloned the repository.

### 5. Edit config file
Edit `your_config.json` file:
- `"common_voice_datapath"` is the path, relative to the root directory, where the downloaded Mozilla Common Voice files are located. Example: `"common_voice_datapath": "corpus/cv-corpus-15.0-2023-09-08/en/"`
- `"wake_words"` is the list of words that your model will learn to recognize.
- `"google_credentials_file"` is the path, relative to the root directory, where your Google Cloud acccess credentials file is located. You can learn how to get your account credentials JSON file in this help article: https://cloud.google.com/iam/docs/keys-create-delete#creating . You can paste the credentials file in the root directory where you cloned the repository.
- `"mfa_DICTIONARY_PATH"` is the path, relative to the root directory, where your downloaded Montreal Forced Aligner Pronunciation dictionary file is located.
- `"mfa_ACOUSTIC_MODEL_PATH"` is the path, relative to the root directory, where your downloaded Montreal Forced Aligner Acoustic model file is located.
- `"dataset_language"` is the [ISO 639-1 code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes "ISO 639-1 code") of the target language. Example: `"en"`
- `"window_size_ms"` is the number of milliseconds of the model's listening time window.
- `"train_epochs"` is the number of epochs for which the model will be trained.
- `"add_vanilla_noise_to_negative_dataset"` determines whether to add the base noise to the negative dataset.
- `"voices_generation_with_google"` determines whether or not to generate synthetic voices with Google Cloud.
- `"custom_dataset_path"` (string or empty string) the path to the directory of your custom dataset. You can get more information in the "Custom datasets" section.
- `"tts_generated_clips"` config of clips generation with the Google Cloud Text-to-Speech API.
	- `"rate"` Speed ​​range of the voices of the generated audios (start, stop and step for np.arange). Min 0.25, max 4.0.
	- `"pitch"` Pitch ​​range of the voices of the generated audios (start, stop and step for np.arange). Min -20.0, max 20.0.

### 6. Prepare and train the model
Run these commands within your conda environment:
- `python dataset_generation.py` (sorts Mozilla Common Voice audio files)
- `python align.py` (runs Montreal Forced Aligner to align the audios)
- `python align_manage.py` (accommodates the results of Montreal Forced Aligner)
- `python train_model.py` (generates additional training data using the Google Cloud Text-to-Speech API and applying noise, and prepares and trains the final model)

### 7. Test the model
You can test the detection of wake words by running the `use_model.py` file and saying the words in order as shown in the console. (It is necessary to have a microphone connected).

**Note:** This tutorial is primarily designed for Windows.

## Port a model to another platform
The resulting file when creating a model with CrafyWakeWord is a PyTorch model file (.pt).
You can port this model to other platforms such as ONNX or TensorFlow.

### Port model to ONNX
1. Verify that the PyTorch model is located in the path `dataset/model_trained.pt`
2. Run this command in the root of the directory where you cloned the repository: `python convert_to_onnx.py`
3. The ONNX model will be saved in `dataset/onnx_model_trained.onnx`

### Port model to TensorFlow
1. Verify that the PyTorch model is located in the path `dataset/model_trained.pt`
2. Run this command in the root of the directory where you cloned the repository: `python convert_to_onnx.py`
3. Run this command: `python convert_onnx_to_tf.py`
4. The TensorFlow model will be saved in `dataset/tf_model_trained`, and the TensorFlow Lite model will be saved in `dataset/tf_model_trained.tflite`

#### Port model to TensorFlow.js
After porting the model to TensorFlow, run the following commands:
1. `conda create -n tfjsconverter python=3.6.8` (only in first execution)
2. `conda activate tfjsconverter`
3. `pip install tensorflowjs[wizard]` (only in first execution)
4. `tensorflowjs_wizard`
	1. ? Please provide the path of model file or the directory that contains model files. `dataset/tf_model_trained`
	2. ? What is your input model format? `Tensorflow Saved Model *`
	3. ? What is tags for the saved model? `serve`
	4. ? What is signature name of the model? `serving_default`
	5. ? Do you want to compress the model? `No compression (Higher accuracy)`
	6. ? Please enter shard size (in bytes) of the weight files? `4194304`
	7. ? Do you want to skip op validation? `No`
	8. ? Do you want to strip debug ops? `Yes`
	9. ? Do you want to enable Control Flow V2 ops? `Yes`
	10. ? Do you want to provide metadata? **ENTER**
	11. ? Which directory do you want to save the converted model in? `dataset/web_model`
5. The TensorFlow.js model will be saved in `dataset/web_model`

## Train another model

Before training a new model for the same trigger words, make a copy and delete the following files/directories:

- `dataset/tf_model_trained/`
- `dataset/web_model/`
- `dataset/model_data.json`
- `dataset/model_trained.pt`
- `dataset/onnx_model_trained.onnx`
- `dataset/tf_model_trained.tflite`
- `dataset/zmuv.pt.bin`

Before training a completely new model, make sure to make a copy and delete the following files/directories:

- `dataset/`

You may need to update the contents of the following folders if you change the language:

- `corpus/`
- `mfa/`

## Custom datasets

To improve model training you can add a custom dataset.

The dataset must have a format similar to Mozilla Common Voice: an audio dataset with its corresponding transcription.

To add a custom dataset you must create a directory in the root of the project with the following structure:

- `clips/` **(mandatory)** a directory containing all the audio clips in the dataset in MP3 format.
- `train.csv` **(mandatory)** a table in CSV format with the columns "path" and "sentence". In the "path" column a string must be entered with the full name of the audio clip file (example: "audio_123.mp3"), audio clips must be saved inside the `clips/` folder; and in the "sentence" column a string must be entered with the complete transcription of the audio clip (example: "Ducks can fly"). The audio clips that will be used for training must be listed in this file.
- `dev.csv` **(optional)** same structure as `train.csv`. The audio clips that will be used for dev must be listed in this file.
- `test.csv` **(optional)** same structure as `train.csv`. The audio clips that will be used for test must be listed in this file.

To use the custom dataset, before performing the training steps, the value of `"custom_dataset_path"` in `your_config.json` must be set to the path of the directory where the custom dataset is located (relative to the root directory). Example: "custom_dataset/". If you want not to use a custom dataset, then set the value of `"custom_dataset_path"` to an empty string.

## Credits and thanks

This library was developed following these instructions: https://github.com/rajashekar/WakeWordDetector/

We thank Rajashekar very much for his excellent work and explanation captured here: https://www.rajashekar.org/wake-word/

Additional thanks to:
- [Mozilla Common Voice](https://commonvoice.mozilla.org/ "Mozilla Common Voice")
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/ "Montreal Forced Aligner")
- [Google Cloud Text-to-Speech API](https://cloud.google.com/text-to-speech "Google Cloud Text-to-Speech API")
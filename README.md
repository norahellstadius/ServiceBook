# ServiceBook

## Description 
Contains code to train and evaluate a binary classifier with a Resnet50 backbone 

## Application 
The classified to used to destinguish images containing a service book and not

## Setup 

### 1. Clone the Repository

```bash
git clone git@github.com:norahellstadius/ServiceBook.git
```

### 2. Setup the enviorment 

#### a. Create a virtual environment

**Note**: make sure its called `.env` (dont forget `.`) since the `.gitignore` file tracks this exact file name. 

```bash
python3 -m venv .env
```

#### b. Activate the virtual environment:

```bash
source .env/bin/activate
```

#### c. Install requirements:

Now that your virtual environment is active, you can install the requirements using pip.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage 

## 1. Place data in folder in following structure

├── data
│   ├── label_1
│   │   ├── image1.png
│   │   ├── image2.png
│   │   ├── image3.png
│   ├── label_2
│   │   ├── image4.png
│   │   ├── image5.png
│   │   ├── image6.png

Here the label_1 and label_2 can be anything of your choose. For example `service book` and `non service book` or `1` anf `0`.


### 2. Train

#### a. Go to the correct directory 
Go to `src` directory 

```bash 
cd src
```

#### b. Run the training script 

Command syntex: 
```bash
python3 train.py --input_data_dir <INPUT_DATA_DIR> --split_data_dir <SPLIT_DATA_DIRR> [--batch_size BATCH_SIZE] [--epochs EPOCHS]
```
##### Required Arguments

- `--input_data_dir`: The path of the directory where the data is located
- `--split_data_dir`: The path of the directory where you want the (train, val, test) split data to be stored

##### Optional Arguments

- `--batch_size`: Batch size.
- `--epochs`: The number of epochs during training

For example to run the training script with the default batch size (4) and number of epochs (5), write the below in comand line 

```bash 
python3 train.py --input_data_dir "PATH/TO/DATA" --split_data_dir "PATH/TO/SAVE/SPLIT/DATA"
```

After training is finnish the path of where the models are saved is printed in terminal. Two models are saved a model which runs for the number of epochs (pre model) and anothe model which run for an additional 3 more epochs (post)
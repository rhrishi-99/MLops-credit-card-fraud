# MLops - Credit Card Fraud Detection

A machine learning operations project for credit card fraud detection using MLflow for experiment tracking and model management.

## Project Structure

```
codecode/
├── main.py                 # Main entry point
├── journal.md              # Project journal/notes
├── data/
│   └── creditcard.csv      # Credit card transaction dataset
├── mlruns/                 # MLflow experiment runs and artifacts
└── src/
    ├── pipeline/
    │   ├── preprocess.py   # Data preprocessing pipeline
    │   └── train.py        # Model training logic
    └── registry/
        └── promote.py      # Model promotion utilities
```

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
cd c:\Users\Rhrishi\Sem6\MLops
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset

The project uses the Credit Card Fraud Detection dataset. Download it using:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
```

Extract the dataset to the `data/` folder.

## Usage

### Run the Pipeline

Execute the main training pipeline:
```bash
python codecode/main.py
```

### Project Components

- **Preprocessing** (`src/pipeline/preprocess.py`): Handles data cleaning, normalization, and feature engineering
- **Training** (`src/pipeline/train.py`): Trains machine learning models with hyperparameter tuning
- **Registry** (`src/registry/promote.py`): Manages model promotion and versioning

## MLflow Integration

Model experiments are tracked using MLflow. View experiments and runs:
```bash
mlflow ui
```

This will start the MLflow UI on `http://localhost:5000`

## Directory Details

- `mlruns/`: Contains MLflow run artifacts, models, and metadata
- `data/`: Stores dataset files
- `src/`: Source code for pipeline and model management

## Notes

See `journal.md` for project progress and development notes.

## License

[Add license information]

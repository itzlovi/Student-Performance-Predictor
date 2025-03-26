# Student Performance Next Level Predictor

## Project Overview
This machine learning project predicts a student's next performance level based on various input features.

## Features
- Predict student's next performance level
- Interactive Streamlit web application
- Custom Equal Feature Importance Classifier

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python -m src.model_training
```

### Running the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Project Structure
- `data/`: Contains raw data
- `models/`: Saved machine learning models
- `src/`: Source code for data processing and model training
- `app/`: Streamlit web application
- `tests/`: Unit tests for the project

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
[MIT]

## Contributors  

- **Parminder Singh**: Led the model training process and efficiently managed API handling to ensure seamless communication between components.  
- **Lovepreet**: Responsible for model testing, designing and developing the frontend, implementing backend functionalities, and integrating APIs.  
- **Shakti Maan**: Conducted in-depth research, providing valuable insights and foundational knowledge to support the project development.  

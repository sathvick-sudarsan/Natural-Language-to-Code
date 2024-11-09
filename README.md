# Natural-Language-to-Code

This project aims to translate English text into Python code using a Seq2Seq model with attention as well as a transformer model. It provides an interface for users to input natural language questions and receive the corresponding Python code.

## Project Structure

- `app.py`: Main application file to run the Streamlit app.
- `app_seq2seq.py`: Core application logic for handling Seq2Seq model interactions.
- `baseline_seq2seq.py`: Defines a baseline Seq2Seq model.
- `evaluate.py`: Contains functions for evaluating the model’s performance.
- `transformer_model.py`: Implementation of the Transformer model.
- `utils.py`: Data preprocessing and loading that is used across the project.

## Setup Instructions

1. **Clone the Repository**: 
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   
2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt

3. **Host The App**:
   ```bash
   python -m streamlit run app.py

   
   

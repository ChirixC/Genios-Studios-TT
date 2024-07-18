# Genios Studio
## Data Science Technical Test

## Table of Contents

- [Genios Studio](#genios-studio)
  - [Data Science Technical Test](#data-science-technical-test)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [SetUp](#setup)
    - [Installation](#installation)
    - [Environment](#environment)

## Overview

This project analyzes social media data using various Python scripts (`clean_data.ipynb`, `read_data.ipynb`, `eda.ipynb`, and `transformers.ipynb`) to perform tasks such as data cleaning, exploratory data analysis (EDA), and sentiment analysis using BERT for sequence classification.

## Project Structure

The project consists of the following files:

- `clean_data.ipynb`: Cleans and preprocesses raw social media data.
- `read_data.ipynb`: Reads and formats raw data into a structured CSV.
- `eda.ipynb`: Conducts exploratory data analysis (EDA) on the cleaned data.
- `transformers.ipynb`: Implements sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers).

## Requirements

To run the project, ensure you have the following installed:

- Python (version >= 3.x)
- Jupyter Notebook
- pandas
- nltk
- emoji
- seaborn
- matplotlib
- scikit-learn
- transformers (Hugging Face library)
- wordcloud


## SetUp
### Installation
Clone the repository and install the required packages:

```
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```
### Environment

Ensure you have Python 3.7+ installed. It's recommended to use a virtual environment (venv or conda) for managing dependencies.
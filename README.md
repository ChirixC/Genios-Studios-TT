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
  - [File Descriptions](#file-descriptions)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Key Insights](#key-insights)

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

## File Descriptions

- **read_data.ipynb**
This notebook reads and preprocesses the raw data into a cleaned CSV format suitable for analysis and modeling.

- **clean_data.ipynb**
This notebook cleans the raw social media data, including removing nulls, duplicates, normalizing text, and extracting features like emojis and sentiment.

- **eda.ipynb**
This notebook performs Exploratory Data Analysis (EDA) on the cleaned dataset, visualizing message lengths, word frequencies, sentiment distributions, temporal analysis, emoji presence, and user-specific insights.

- **transformers.ipynb**
This notebook implements a machine learning model using BERT for sentiment analysis on the preprocessed data.

## Exploratory Data Analysis (EDA)
### Key Insights

During the exploratory analysis of the dataset, several trends and patterns emerged that shed light on the sentiment and impact of emojis in the comments. Of these trends, I want to highlight the following three:

 1. **Sentiment Analysis:** A bar plot illustrates that a significant majority of comments exhibit positive sentiment. This observation underscores the predominantly positive nature of the dataset.

![Sentiment Distribution](assets/output_sentiment_messages.png)

  1. **Common Words:** A word cloud visualization provides a clear depiction of the most frequently used words in the comments, emphasizing that larger words are predominantly positive in nature. This graphical representation enhances understanding of the prevalent sentiment conveyed through the comments.

![Most Common Words](assets/output_words.png)

  3. **Emoji Impact:** Contrary to the assumption that more emojis indicate greater positivity, a bar plot analysis demonstrates that the presence of emojis varies significantly across messages. While positive messages often incorporate emojis, negative messages tend to utilize them less frequently.

![Emojis Impact](assets/output_has_emoji.png)
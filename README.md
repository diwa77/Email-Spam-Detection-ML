# Email Spam Detection Using Machine Learning

## Project Overview
Email Spam Detection is a Machine Learning project that classifies email messages into two categories: Spam and Ham (legitimate email).

The project uses TF-IDF feature extraction and compares two machine learning algorithms: Multinomial Naive Bayes and Support Vector Machine (SVM).

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLTK
- Natural Language Processing (NLP)
- TF-IDF Vectorization

## Project Workflow
1. Load the email dataset using Pandas.
2. Explore the dataset and check email labels.
3. Perform text preprocessing demonstrations, including cleaning, tokenization and stopword removal.
4. Convert email messages into numerical features using TF-IDF.
5. Split the dataset into training and testing sets.
6. Train Naive Bayes and SVM classifiers.
7. Evaluate both models using Accuracy, Precision, Recall and F1-score.
8. Generate confusion matrices and predict new email messages.

## Dataset
- File: emails.csv
- Total records: 12
- Spam emails: 6
- Ham emails: 6
- Columns: message, label

*Note:* This is a small dataset used for initial project implementation and testing. More data is required for reliable performance evaluation.

## Model Evaluation

| Metric | Naive Bayes | SVM |
|---|---:|---:|
| Accuracy | 66.67% | 100% |
| Precision | 50% | 100% |
| Recall | 100% | 100% |
| F1-score | 66.67% | 100% |

*Note:* These results are based on a test set of only 3 emails and should not be considered representative of real-world performance.

## Project Structure
text
Email-Spam-Detection/
│
├── main.py
├── emails.csv
└── README.md


## Installation and Execution

Install the required libraries:

bash
pip install pandas scikit-learn nltk


Run the project:

bash
python main.py


## Future Improvements
- Use a larger and more diverse email dataset.
- Improve text preprocessing.
- Perform more robust model evaluation.
- Test the model on unseen email messages.
- Develop a simple user interface.

## Author
Diwakar Naidu

B.Tech – Artificial Intelligence and Data Science
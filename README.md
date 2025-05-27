# Task 1: Customer Support Ticket Classifier

## Overview

This notebook implements a machine learning pipeline to classify customer support tickets by issue type and urgency level, extract entities (product, dates, complaint keywords), and provide a Gradio interface for single and batch ticket processing. The project includes visualizations for ticket distributions, feature importances, and confusion matrices to evaluate model performance.

## Dependencies

- Python 3.8+
- Libraries: pandas, nltk, scikit-learn, textblob, xgboost, imblearn, gradio, matplotlib, seaborn
- Install: `pip install pandas nltk scikit-learn xgboost imblearn gradio matplotlib seaborn`
- NLTK data: punkt, stopwords, wordnet

## How to Run

1. Place `ai_dev_assignment_tickets_complex_1000.csv` in the same directory (note: the dataset is a CSV file with tab-separated values, not an Excel file as initially stated).
2. Run all notebook sections in sequence.
3. Launch the Gradio app to interact with the classifier (requires internet for hosted mode).
4. Visualizations include ticket distributions, confusion matrices, and feature importance for Random Forest.

## Approach

**Data Preparation**:

- Loaded CSV file with pandas using a tab separator.
- Handled missing values by imputing defaults: `ticket_text` as an empty string, `issue_type` as "General Inquiry", `urgency_level` as "Medium".
- Preprocessed text by applying lowercase, removing special characters, tokenizing, removing custom stopwords (excluding complaint terms), and lemmatizing.
- Encoded target variables (`issue_type` and `urgency_level`) into numerical form using `LabelEncoder`.

**Feature Engineering**:

- Applied TF-IDF vectorization to `processed_text` with bigrams and a minimum document frequency threshold.
- Used `OneHotEncoder` for the `product` column.
- Created binary features for issue types (e.g., `has_billing_issue`, `has_installation_issue`) and urgency indicators (e.g., `has_urgent_keywords`).
- Initially explored numerical features like ticket length and sentiment score, but simplified to binary features to improve model performance.
- Combined features using `ColumnTransformer` with `StandardScaler` for binary features.

**Model Training**:

- Trained multiple models (SVM, Random Forest, XGBoost, Logistic Regression) for both issue type and urgency level classification.
- Used Stratified K-Fold cross-validation and GridSearchCV for hyperparameter tuning (e.g., tuning `C` for SVM, `max_depth` for Random Forest).
- Addressed class imbalance with SMOTE, applying a custom sampling strategy to oversample minority classes like "Installation Issue".
- Evaluated models using classification reports and confusion matrices.
- Visualized feature importance for Random Forest to identify key features like `has_installation_issue`.

**Entity Extraction**:

- Extracted products using a regex pattern based on known product names from the dataset.
- Extracted dates using regex for formats like YYYY-MM-DD.
- Extracted complaint keywords from a predefined list.

**Integration**:

- Developed a `process_ticket` function to predict issue type, urgency level, and extract entities for a single ticket.
- Returned results as a JSON string with predictions from all models and extracted entities.

**Gradio Interface**:

- Built a web app for single and batch ticket processing.
- Supports batch processing with comma-separated ticket descriptions (e.g., "Ticket 1, Ticket 2").
- Outputs formatted results with product, issue type predictions, urgency level predictions, and entities for each ticket.

**Evaluation and Visualizations**:

- Visualized ticket distributions for issue type and urgency level using seaborn bar plots.
- Generated confusion matrices for each model to identify misclassifications.
- Plotted feature importance for Random Forest to highlight the impact of binary features.

## Design Choices

- **Preprocessing**: Excluded complaint terms from stopwords to retain important words for classification.
- **Features**: Focused on TF-IDF, one-hot-encoded `product`, and binary features after numerical features like ticket length showed limited improvement.
- **Models**: Employed an ensemble of models with cross-validation for robustness; added Naive Bayes for its strength in text classification.
- **Class Imbalance**: Used SMOTE and custom class weights to handle imbalanced classes like "Installation Issue".
- **Entity Extraction**: Relied on regex for simplicity, leveraging known patterns in the dataset.

## Limitations

- **Missing Data**: Imputing "General Inquiry" or "Medium" for missing labels may bias predictions toward these classes.
- **Class Imbalance**: Despite SMOTE, minority classes like "Wrong Item" or "High" urgency may still be underrepresented in some folds.
- **Feature Set**: Simplified by removing numerical features, potentially discarding useful signals for some tickets.
- **Entity Extraction**: Regex-based extraction may miss complex date formats (e.g., "March 3rd") or products not in the predefined list.
- **Scalability**: Batch processing in Gradio is limited to comma-separated inputs; a more robust UI could improve usability.

## Challenges

- **Class Imbalance**: Addressed with SMOTE and custom class weights, but initial predictions were biased toward majority classes like "General Inquiry".
- **Feature Engineering**: Simplified from numerical features to binary features to reduce overfitting and improve performance.
- **Preprocessing**: Defined custom stopwords to exclude complaint terms, ensuring retention of important words.
- **Model Tuning**: Used GridSearchCV with Stratified K-Fold for better performance, keeping parameter grids small to manage training time.
- **Error Handling**: Resolved issues like missing columns and undefined variables by ensuring proper feature creation and variable definitions

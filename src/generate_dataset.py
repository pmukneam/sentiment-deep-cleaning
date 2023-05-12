# Import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download the NLTK stopwords if not already downloaded
nltk.download("stopwords")
# Set the English stopwords
stop_words = set(stopwords.words("english"))


def preprocess_review_text(text):
    # Check if the input is not a string (e.g., missing value, float, etc.), return an empty string
    if not isinstance(text, str):
        return ""

    # Convert the text to lowercase
    text = text.lower()
    # Remove all characters except letters and spaces
    text = re.sub(r"[^a-z\s]+", "", text)
    # Split the text into words
    words = text.split()
    # Remove stopwords from the list of words
    filtered_words = [word for word in words if word not in stop_words]
    # Join the filtered words back into a string and remove any trailing spaces
    return " ".join(filtered_words).strip()


def prepare_dataset(file_path, num_rows):
    # Read the CSV file, selecting only the 'review_text' and 'review_score' columns, and the specified number of rows
    df = pd.read_csv(file_path, usecols=["review_score", "review_text"], nrows=num_rows)
    # Apply the preprocess_review_text function to the 'review_text' column and store the results in a new column 'filtered_review_text'
    df["filtered_review_text"] = df["review_text"].apply(preprocess_review_text)

    # Remove duplicate rows
    df.drop_duplicates(subset=["filtered_review_text"], inplace=True)

    # Remove NAs Row
    df.dropna(subset=["filtered_review_text"], inplace=True)

    # Remove rows where 'filtered_review_text' is empty
    df = df[df["filtered_review_text"].str.len() > 0]

    # Shuffle the DataFrame and reset the index
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def filter_dataframe(df, n, pos):
    # Filter positive and negative reviews
    df_positive = df[df['review_score'] == 1]
    df_negative = df[df['review_score'] == -1]

    # Calculate the number of positive and negative reviews
    n_positive = round(n * pos)
    n_negative = n - n_positive

    # Select n_positive from df_positive and n_negative from df_negative
    df_positive_sample = df_positive.sample(n_positive)
    df_negative_sample = df_negative.sample(n_negative)

    # Concatenate the two dataframes
    result = pd.concat([df_positive_sample, df_negative_sample])

    return result

#------------------------------------------------------------ Main ------------------------------------------------------------


raw_file_path = 'dataset/raw/dataset.csv'
processed_df = prepare_dataset(raw_file_path, 500000)
filtered_df = filter_dataframe(processed_df, 10, 0.65)
filtered_df.to_csv('dataset/train/filtered_dataset_10.csv', index=False)

# -*- coding: utf-8 -*-
"""deep-clean-pre-trained.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wfWSico6C_J6WngAJl8-TqX4Bpaoxn8Q
"""

# Install required libraries
! pip install -q transformers accelerate sentencepiece gradio pandas nltk

"""#**Model: GPT-2**"""

# Importing models
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, AutoTokenizer, TFAutoModelForCausalLM

"""- **gpt2**"""

# First time run
model = TFAutoModelForCausalLM.from_pretrained('gpt2')
model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/GPT-2")

# Load model
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#model = TFGPT2LMHeadModel.from_pretrained()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TFAutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/GPT-2")
model.config.pad_token_id = model.config.eos_token_id

text_1 = "Keeps only sentimental words: 'I love and like this game'"
text_2 = "Extracting words that imply sentiment of the given review: I love and like this game'"

# Encode
encoded_input_1 = tokenizer(text_1, return_tensors='tf')
encoded_input_2 = tokenizer(text_2, return_tensors='tf')

# generate outputs and decode
output_1 = model.generate(**encoded_input_1, early_stopping=True)
decoded_1 = tokenizer.decode(output_1[0])

output_2 = model.generate(**encoded_input_2, early_stopping=True)
decoded_2 = tokenizer.decode(output_2[0])

print(decoded_1)
print('')
print(decoded_2)

"""**Generate Review**"""

# Load libraries
import pandas as pd
import nltk
nltk.download('punkt')

# Load dataset
file_path = '/content/filtered_dataset_10.csv'
df = pd.read_csv(file_path)
df

# Define Variables
prefix_1 = "Keeps only sentimental words: "
prefix_2 = "Extracting words that imply sentiment of the given review: "
max_input_length = 20

def generate_review_gpt2(review_text):
  prefix = "Keeps only sentimental words: "
  # Format input
  input = prefix + review_text
  # Encode
  encoded_input = tokenizer(input,
                            max_length=max_input_length, 
                            truncation=True, 
                            return_tensors='tf')

  # generate outputs and decode
  output = model.generate(**encoded_input, 
                          early_stopping=True,
                          max_length=25,
                          repetition_penalty = 5.2,
                          top_k = 80)
  decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]


  return decoded

df["gpt2"] = df["review_text"].apply(generate_review_gpt2)

df

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Deep Clean/gpt2_demo.csv', index=False)

"""- **gpt2-xl**

"""

# First time run
model = TFAutoModelForCausalLM.from_pretrained('gpt2-xl')
model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/GPT-2-xl")

# Load model
tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
model = TFAutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/GPT-2-xl")
model.config.pad_token_id = model.config.eos_token_id

text_1 = "Keeps only sentimental words: 'I love and like this game'"
text_2 = "Extracting words that imply sentiment of the given review: I love and like this game'"

# Encode
encoded_input_1 = tokenizer(text_1, return_tensors='tf')
encoded_input_2 = tokenizer(text_2, return_tensors='tf')

# generate outputs and decode
output_1 = model.generate(**encoded_input_1, early_stopping=True)
decoded_1 = tokenizer.decode(output_1[0])

output_2 = model.generate(**encoded_input_2, early_stopping=True)
decoded_2 = tokenizer.decode(output_2[0])

print(decoded_1)
print('')
print(decoded_2)

"""**Generate Review**"""

# Load libraries
import pandas as pd
import nltk
nltk.download('punkt')

# Load dataset
file_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/filtered_dataset.csv'
df = pd.read_csv(file_path)

# Define Variables
prefix_1 = "Keeps only sentimental words: "
prefix_2 = "Extracting words that imply sentiment of the given review: "
max_input_length = 20

def generate_review_gpt2_xl(review_text):
  prefix = "Keeps only sentimental words: "
  # Format input
  input = prefix + review_text
  # Encode
  encoded_input = tokenizer(input,
                            max_length=max_input_length, 
                            truncation=True, 
                            return_tensors='tf')

  # generate outputs and decode
  output = model.generate(**encoded_input, 
                          early_stopping=True,
                          max_length=25,
                          repetition_penalty = 5.2,
                          top_k = 80)
  decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]


  return decoded

df["gpt2-xl"] = df["review_text"].apply(generate_review_gpt2_xl)

df

df
df.to_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Deep Clean/gpt2-xl.csv', index=False)



"""# **Model: T5**"""

# Importing models
from transformers import T5Tokenizer, T5ForConditionalGeneration

"""- **t5-base**"""

# First time run
model = T5ForConditionalGeneration.from_pretrained("t5-base").to("cuda")
model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/t5-base")

# Load model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/t5-base").to("cuda")

text_1 = "Keeps only sentimental words: 'I love and like this game'"
text_2 = "Extracting words that imply sentiment of the given review: I love and like this game'"

# Encode
encoded_input_1 = tokenizer(text_1, return_tensors="pt").input_ids.to("cuda")
encoded_input_2 = tokenizer(text_2, return_tensors="pt").input_ids.to("cuda")

# generate outputs and decode
output_1 = model.generate(encoded_input_1, early_stopping=True)
decoded_1 = tokenizer.decode(output_1[0], skip_special_tokens=True)

output_2 = model.generate(encoded_input_2, early_stopping=True)
decoded_2 = tokenizer.decode(output_2[0], skip_special_tokens=True)

print(decoded_1)
print('')
print(decoded_2)

"""**Generate Rewview**"""

# Load libraries
import pandas as pd
import nltk
nltk.download('punkt')

# Load dataset
file_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/filtered_dataset.csv'
df = pd.read_csv(file_path)

# Define Variables
prefix_1 = "Keeps only sentimental words: "
prefix_2 = "Extracting words that imply sentiment of the given review: "
max_input_length = 50

def generate_review_t5_base(review_text):
  prefix = "Keeps only sentimental words: "
  # Format input
  input = prefix + review_text
  # Encode
  encoded_input = tokenizer(input,
                            max_length=max_input_length, 
                            truncation=True, 
                            return_tensors='pt').input_ids.to("cuda")

  # generate outputs and decode
  output = model.generate(encoded_input, 
                          early_stopping=True,
                          max_length=64)
  decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

  #decoded = nltk.sent_tokenize(decoded.strip())[0]

  return decoded


df

df["t5-base-prefix-1"] = df["review_text"].apply(generate_review_t5_base)

df

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Deep Clean/t5-base.csv', index=False)

"""- **t5-XL**"""

# First time run
model = T5ForConditionalGeneration.from_pretrained("t5-3b").to("cuda")
model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/t5-3b")

# Load model
tokenizer = T5Tokenizer.from_pretrained("t5-3b")
model = T5ForConditionalGeneration.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/t5-3b").to("cuda")

text_1 = "Keeps only sentimental words: 'I love and like this game'"
text_2 = "Keeps words that imply sentiment of the given review: I love and like this game'"

# Encode
encoded_input_1 = tokenizer(text_1, return_tensors="pt").input_ids.to("cuda")
encoded_input_2 = tokenizer(text_2, return_tensors="pt").input_ids.to("cuda")

# generate outputs and decode
output_1 = model.generate(encoded_input_1, early_stopping=True)
decoded_1 = tokenizer.decode(output_1[0], skip_special_tokens=True)

output_2 = model.generate(encoded_input_2, early_stopping=True)
decoded_2 = tokenizer.decode(output_2[0], skip_special_tokens=True)

print(decoded_1)
print('')
print(decoded_2)

"""**Generate Review**"""

# Load libraries
import pandas as pd
import nltk
nltk.download('punkt')

# Load dataset
file_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/filtered_dataset.csv'
df = pd.read_csv(file_path)

# Define Variables
prefix_1 = "Keeps only sentimental words: "
prefix_2 = "Extracting words that imply sentiment of the given review: "
max_input_length = 50

def generate_review_t5_3b(review_text):
  prefix = "Keeps only sentimental words: "
  # Format input
  input = prefix + review_text
  # Encode
  encoded_input = tokenizer(input,
                            max_length=max_input_length, 
                            truncation=True, 
                            return_tensors='pt').input_ids.to("cuda")

  # generate outputs and decode
  output = model.generate(encoded_input, 
                          early_stopping=True,
                          max_length=64)
  decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

  #decoded = nltk.sent_tokenize(decoded.strip())[0]

  return decoded


df

df["t5-3b-prefix-1"] = df["review_text"].apply(generate_review_t5_3b)

df

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Deep Clean/t5-3b.csv', index=False)

"""- **flan-t5-base**"""

# First time run
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to("cuda")
model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/flan-t5-base")

# Load model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/flan-t5-base").to("cuda")

text_1 = "Keeps only adjective words: 'I love and like this game'"
text_2 = "keeps only words that imply sentiment of the given review: I love and like this game'"

# Encode
encoded_input_1 = tokenizer(text_1, return_tensors="pt").input_ids.to("cuda")
encoded_input_2 = tokenizer(text_2, return_tensors="pt").input_ids.to("cuda")

# generate outputs and decode
output_1 = model.generate(encoded_input_1, early_stopping=True)
decoded_1 = tokenizer.decode(output_1[0], skip_special_tokens=True)

output_2 = model.generate(encoded_input_2, early_stopping=True)
decoded_2 = tokenizer.decode(output_2[0], skip_special_tokens=True)

print(decoded_1)
print('')
print(decoded_2)

"""**Generate Review**"""

# Load libraries
import pandas as pd
import nltk
nltk.download('punkt')

# Load dataset
file_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/filtered_dataset.csv'
df = pd.read_csv(file_path)

# Define Variables
prefix_1 = "Keeps only sentimental words: "
prefix_2 = "Extracting words that imply sentiment of the given review: "
max_input_length = 50

def generate_review_flan_t5_base(review_text):
  prefix = "Keeps only sentimental words: "
  # Format input
  input = prefix + review_text
  # Encode
  encoded_input = tokenizer(input,
                            max_length=max_input_length, 
                            truncation=True, 
                            return_tensors='pt').input_ids.to("cuda")

  # generate outputs and decode
  output = model.generate(encoded_input, 
                          early_stopping=True,
                          max_length=64)
  decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

  #decoded = nltk.sent_tokenize(decoded.strip())[0]

  return decoded


df

df["flan-t5-base-prefix-1"] = df["review_text"].apply(generate_review_flan_t5_base)

df

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Deep Clean/flan-t5-base.csv', index=False)

"""- **flan-t5-xl**"""



# First time run
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl").to("cuda")
model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/flan-t5-xl")

# Load model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Model/flan-t5-xl").to("cuda")

print(decoded_1)
print('')
print(decoded_2)

# Load libraries
import pandas as pd
import nltk
nltk.download('punkt')

# Load dataset
file_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/filtered_dataset.csv'
df = pd.read_csv(file_path)

# Define Variables
prefix_1 = "Keeps only sentimental words: "
prefix_2 = "Extracting words that imply sentiment of the given review: "
max_input_length = 50

def generate_review_flan_t5_xl(review_text):
  prefix = "Keeps only sentimental words: "
  # Format input
  input = prefix + review_text
  # Encode
  encoded_input = tokenizer(input,
                            max_length=max_input_length, 
                            truncation=True, 
                            return_tensors='pt').input_ids.to("cuda")

  # generate outputs and decode
  output = model.generate(encoded_input, 
                          early_stopping=True,
                          max_length=64)
  decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

  #decoded = nltk.sent_tokenize(decoded.strip())[0]

  return decoded


df

df["flan-t5-xl-prefix-1"] = df["review_text"].apply(generate_review_flan_t5_xl)

df

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Deep Clean/flan-t5-xl.csv', index=False)
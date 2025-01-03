#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install chardet


# In[3]:


import chardet


# In[ ]:


import pandas as pd


# In[4]:


# Read a small portion of the file to detect encoding
with open("steemit_tsv.zip", "rb") as f:
    raw_data = f.read(10000)  # Read first 10KB of data
result = chardet.detect(raw_data)
print(result)  # Output will contain the detected encoding


# In[12]:


import pandas as pd
import os

# Path to the folder containing the .tsv file
folder_path = "steemit_tsv.zip_extracted"
file_name = "steemit_2024-07-21.tsv"
file_path = os.path.join(folder_path, file_name)

# Read the file into a DataFrame
try:
    # Read the file, skipping invalid lines
    df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip", encoding="utf-8")

    # Rename columns for easier access
    df.columns = [f"index{i}" for i in range(df.shape[1])]

    # Filter rows where index2 is "comment"
    filtered_df = df[df["index2"].str.lower() == "comment"]

    # Select relevant columns: index1, index9, index10, index11
    selected_columns = filtered_df[["index1", "index9", "index10", "index11"]]

    # Display only the first ten rows
    first_ten_comments = selected_columns.head(10)
    print("First Ten Comments:")
    print(first_ten_comments)

except FileNotFoundError:
    print(f"File '{file_name}' not found in the folder '{folder_path}'. Please check the file path.")
except pd.errors.ParserError as e:
    print(f"Error parsing the file: {e}")


# In[12]:


import pandas as pd
import os

# Path to the folder containing the .tsv file
folder_path = "steemit_tsv.zip_extracted"
file_name = "steemit_2024-07-21.tsv"
file_path = os.path.join(folder_path, file_name)

# Read the file into a DataFrame
try:
    # Read the file, skipping invalid lines
    df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip", encoding="utf-8")

    # Rename columns for easier access
    df.columns = [f"index{i}" for i in range(df.shape[1])]

    # Filter rows where index2 is "comment"
    filtered_df = df[df["index2"].str.lower() == "comment"]

    # Select relevant columns: index1, index9, index10, index11
    selected_columns = filtered_df[["index1", "index9", "index10", "index11"]]

    # Display only the first ten rows
    first_ten_comments = selected_columns.head(10)
    print("First Ten Comments:")
    print(first_ten_comments)

except FileNotFoundError:
    print(f"File '{file_name}' not found in the folder '{folder_path}'. Please check the file path.")
except pd.errors.ParserError as e:
    print(f"Error parsing the file: {e}")


# In[12]:


import pandas as pd
import os

# Path to the folder containing the .tsv file
folder_path = "steemit_tsv.zip_extracted"
file_name = "steemit_2024-07-21.tsv"
file_path = os.path.join(folder_path, file_name)

# Read the file into a DataFrame
try:
    # Read the file, skipping invalid lines
    df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip", encoding="utf-8")

    # Rename columns for easier access
    df.columns = [f"index{i}" for i in range(df.shape[1])]

    # Filter rows where index2 is "comment"
    filtered_df = df[df["index2"].str.lower() == "comment"]

    # Select relevant columns: index1, index9, index10, index11
    selected_columns = filtered_df[["index1", "index9", "index10", "index11"]]

    # Display only the first ten rows
    first_ten_comments = selected_columns.head(10)
    print("First Ten Comments:")
    print(first_ten_comments)

except FileNotFoundError:
    print(f"File '{file_name}' not found in the folder '{folder_path}'. Please check the file path.")
except pd.errors.ParserError as e:
    print(f"Error parsing the file: {e}")


# In[12]:


import pandas as pd
import os

# Path to the folder containing the .tsv file
folder_path = "steemit_tsv.zip_extracted"
file_name = "steemit_2024-07-21.tsv"
file_path = os.path.join(folder_path, file_name)

# Read the file into a DataFrame
try:
    # Read the file, skipping invalid lines
    df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip", encoding="utf-8")

    # Rename columns for easier access
    df.columns = [f"index{i}" for i in range(df.shape[1])]

    # Filter rows where index2 is "comment"
    filtered_df = df[df["index2"].str.lower() == "comment"]

    # Select relevant columns: index1, index9, index10, index11
    selected_columns = filtered_df[["index1", "index9", "index10", "index11"]]

    # Display only the first ten rows
    first_ten_comments = selected_columns.head(10)
    print("First Ten Comments:")
    print(first_ten_comments)

except FileNotFoundError:
    print(f"File '{file_name}' not found in the folder '{folder_path}'. Please check the file path.")
except pd.errors.ParserError as e:
    print(f"Error parsing the file: {e}")


# In[ ]:





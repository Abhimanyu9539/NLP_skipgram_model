import os
import re
import config
import argparse
import pandas as pd
from tqdm import tqdm
from Source.utils import save_file
from nltk.tokenize import word_tokenize


def main(args_):
    print("Reading the data file...")
    data = pd.read_csv(os.path.join(args_.input_path, args_.file_name))
    data.dropna(subset=[args_.col_name], inplace=True)
    input_text = data[args_.col_name]
    # Convert text to lower case
    print("Converting text to lower case...")
    input_text = [i.lower() for i in tqdm(input_text)]
    # Remove punctuations except apostrophe
    print("Removing punctuations in text...")
    input_text = [re.sub(r"[^\w\d'\s]+", " ", i) for i in tqdm(input_text)]
    # Remove digits
    print("Removing digits in text...")
    input_text = [re.sub("\d+", "", i) for i in tqdm(input_text)]
    # Remove more than one consecutive instance of 'x'
    print("Removing 'xxxx...' in text")
    input_text = [re.sub(r'[x]{2,}', "", i) for i in tqdm(input_text)]
    # Replace multiple spaces with single space
    print("Removing additional spaces in text...")
    input_text = [re.sub(' +', ' ', i) for i in tqdm(input_text)]
    # Tokenize the text
    print("Tokenizing the text...")
    tokens = [word_tokenize(t) for t in tqdm(input_text)]
    print("Saving tokens...")
    save_file(os.path.join(args_.output_path, args_.token_file), tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default=config.file_name,
                        help="Input file name")
    parser.add_argument("--col_name", type=str, default=config.col_name,
                        help="Text column name")
    parser.add_argument("--input_path", type=str, default=config.input_folder,
                        help="Input folder name")
    parser.add_argument("--output_path", type=str, default=config.output_folder,
                        help="Output folder name")
    parser.add_argument("--token_file", type=str, default=config.token_file,
                        help="File containing word tokens")
    args = parser.parse_args()
    main(args)

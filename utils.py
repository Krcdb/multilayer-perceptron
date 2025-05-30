import os
import sys
import pandas as pd


def load_csv_df(input):
  if not os.path.isfile(input):
    print(f"Error: File '{input}' does not exist.")
    sys.exit(1)
  
  try:
    df = pd.read_csv(input)
  except Exception as e:
    print(f"Error reading file '{input}': {e}")
    sys.exit(1)
    
  return df
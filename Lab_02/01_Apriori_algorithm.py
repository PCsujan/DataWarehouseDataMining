# Required Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Read the text file
def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Parse the text into transaction list
def parse(text):
    lines = text.strip().split("\n")
    transactions = []
    for line in lines[1:]:  # Skip header
        items = line.split(",")[1:]  # Skip transaction ID
        cleaned = [item.strip() for item in items if item.strip()]
        transactions.append(cleaned)
    return transactions

# Convert transactions into DataFrame
def encode(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
def apply_apriori(df, min_support=0.1, min_conf=0.3):
    freq_items = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    return freq_items, rules

# --- MAIN PROGRAM STARTS HERE ---

# Read and parse files
sports_text = read_text("data/sports.txt")
space_text = read_text("data/space.txt")

sports_transactions = parse(sports_text)
space_transactions = parse(space_text)

sports_df = encode(sports_transactions)
space_df = encode(space_transactions)

# Apply Apriori
sports_freq, sports_rules = apply_apriori(sports_df)
space_freq, space_rules = apply_apriori(space_df)

# Output for Sports
print("\n Sports - Frequent Itemsets:\n", sports_freq)
if sports_rules.empty:
    print("\n No Sports Association Rules Generated. Try lowering support/confidence.")
else:
    print("\n Sports - Association Rules:\n", sports_rules[['antecedents','consequents','support','confidence','lift']])

# Output for Space
print("\n Space - Frequent Itemsets:\n", space_freq)
if space_rules.empty:
    print("\n No Space Association Rules Generated. Try lowering support/confidence.")
else:
    print("\n Space - Association Rules:\n", space_rules[['antecedents','consequents','support','confidence','lift']])

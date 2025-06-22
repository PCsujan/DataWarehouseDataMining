import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# --- Read and Parse ---

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def parse(text):
    lines = text.strip().split("\n")
    transactions = []
    for line in lines[1:]:  # skip header
        items = line.split(",")[1:]  # skip TID
        cleaned = [item.strip() for item in items if item.strip()]
        transactions.append(cleaned)
    return transactions

def encode(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

# --- Apply Apriori and FP-Growth ---

def apply_apriori(df, min_support=0.1, min_conf=0.3):
    freq = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    return rules

def apply_fpgrowth(df, min_support=0.1, min_conf=0.3):
    freq = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    return rules

# --- Comparison Printer ---

def compare_results(name, apriori_rules, fp_rules):
    print(f"---- {name} ----")
    print(f"Apriori Rules Count: {len(apriori_rules)}")
    print(f"FP-Growth Rules Count: {len(fp_rules)}\n")

# --- Load and Run ---

# Sports
sports_text = read_text("data/sports.txt")
sports_df = encode(parse(sports_text))
sports_rules = apply_apriori(sports_df)
sports_fp_rules = apply_fpgrowth(sports_df)

# Space
space_text = read_text("data/space.txt")
space_df = encode(parse(space_text))
space_rules = apply_apriori(space_df)
space_fp_rules = apply_fpgrowth(space_df)

# Compare Output
compare_results("Sports", sports_rules, sports_fp_rules)
compare_results("Space", space_rules, space_fp_rules)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Step 1: Read file
def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Step 2: Parse transactions
def parse(text):
    lines = text.strip().split('\n')
    transactions = []
    for line in lines[1:]:  # skip header
        items = line.split(',')[1:]  # skip transaction ID
        cleaned = [item.strip() for item in items if item.strip()]
        if cleaned:
            transactions.append(cleaned)
    return transactions

# Step 3: Encode transactions
def encode(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)

# Step 4: Apply Apriori
def apply_apriori(df, min_support=0.1, min_conf=0.3):
    freq = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    return freq, rules

# Step 5: Apply FP-Growth
def apply_fpgrowth(df, min_support=0.1, min_conf=0.3):
    freq = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    return freq, rules

# --------- RUN FOR SPORTS ---------
sports_text = read_text("data/sports.txt")
sports_transactions = parse(sports_text)
sports_df = encode(sports_transactions)

sports_apriori_freq, sports_apriori_rules = apply_apriori(sports_df)
sports_fp_freq, sports_fp_rules = apply_fpgrowth(sports_df)

print(" Sports - Apriori Frequent Itemsets:\n", sports_apriori_freq)
if sports_apriori_rules.empty:
    print(" No Apriori rules for Sports.\n")
else:
    print("\n Sports - Apriori Association Rules:\n", sports_apriori_rules[['antecedents','consequents','support','confidence','lift']])

print("\n Sports - FP-Growth Frequent Itemsets:\n", sports_fp_freq)
if sports_fp_rules.empty:
    print(" No FP-Growth rules for Sports.\n")
else:
    print("\n Sports - FP-Growth Association Rules:\n", sports_fp_rules[['antecedents','consequents','support','confidence','lift']])

# --------- RUN FOR SPACE ---------
space_text = read_text("data/space.txt")
space_transactions = parse(space_text)
space_df = encode(space_transactions)

space_apriori_freq, space_apriori_rules = apply_apriori(space_df)
space_fp_freq, space_fp_rules = apply_fpgrowth(space_df)

print("\n Space - Apriori Frequent Itemsets:\n", space_apriori_freq)
if space_apriori_rules.empty:
    print(" No Apriori rules for Space.\n")
else:
    print("\n Space - Apriori Association Rules:\n", space_apriori_rules[['antecedents','consequents','support','confidence','lift']])

print("\n Space - FP-Growth Frequent Itemsets:\n", space_fp_freq)
if space_fp_rules.empty:
    print(" No FP-Growth rules for Space.\n")
else:
    print("\n Space - FP-Growth Association Rules:\n", space_fp_rules[['antecedents','consequents','support','confidence','lift']])

pip install mlxtend --quiet

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Transactions dataset
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'sugar'],
    ['milk', 'bread', 'sugar'],
    ['milk', 'bread', 'butter', 'sugar'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter'],
    ['bread', 'sugar', 'coffee'],
    ['milk', 'bread', 'coffee'],
    ['milk', 'bread', 'sugar', 'butter']
]

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets
frequent_items = apriori(df, min_support=0.3, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.6)

# Display results
print("Frequent Itemsets:\n", frequent_items)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

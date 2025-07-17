import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load the data
df = pd.read_csv(r"C:\Users\HP\Documents\PROJECT\spam.csv", encoding='latin-1')

# Clean up and keep only necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Show basic info
print("Label distribution:")
print(df['label'].value_counts())
print("\nSample messages:")
print("Ham:", df[df['label'] == 'ham']['message'].iloc[0])
print("Spam:", df[df['label'] == 'spam']['message'].iloc[0])

# Step 2: Split into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("\nTraining set size:", len(train_df))
print("Test set size:", len(test_df))

# Step 3: Define keyword-based classifier
# Build a list of spammy keywords (students can expand this list)
spam_keywords = ['free', 'win', 'urgent', 'prize', 'cash', 'claim', 'winner', 'congratulations', 'click', 'now', 'offer']

def keyword_classifier(text):
    text = text.lower()
    return 'spam' if any(word in text for word in spam_keywords) else 'ham'

# Step 4: Apply classifier to test data
test_df['predicted'] = test_df['message'].apply(keyword_classifier)

# Step 5: Evaluate the keyword-based classifier
print("\n--- Keyword-based Classifier Evaluation ---")
print(classification_report(test_df['label'], test_df['predicted']))

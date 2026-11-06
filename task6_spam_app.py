import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# ================= 1. DATA =================
spam = [
    "Congratulations! You have won a free iPhone. Click here to claim now",
    "WIN cash prize of Rs 50000. Reply YES to claim your reward",
    "You have been selected for a free holiday. Call now to claim",
    "URGENT! Your account will be blocked. Click the link to verify now",
    "Get a free loan approved instantly, no documents needed, apply now",
    "Claim your free gift card today, limited offer, click the link",
    "Earn money from home, Rs 5000 daily, join now for free",
    "You won the lottery! Send your bank details to claim the prize",
    "Free recharge offer! Click this link to get 100% cashback now",
    "Lowest price offer, buy now and win a free gift, hurry",
    "Your number won a lucky draw, call this number to claim prize",
    "Exclusive deal just for you, 90% discount, click to buy now",
    "Final notice: claim your reward points before they expire, click now",
    "Free entry to win a car, text WIN to 55555 now",
    "Hot singles in your area, click here for free",
    "Congratulations, you are our lucky winner, claim cash prize today",
    "Get rich quick, invest Rs 1000 and earn Rs 10000 in a week",
    "Your KYC is pending, update now by clicking this link or account blocked",
    "Limited time offer! Free subscription, click to activate now",
    "You have a pending prize of Rs 1 lakh, reply with your details",
]
ham = [
    "Hi, are we still meeting for lunch tomorrow?",
    "Please send me the project report by evening",
    "Can you call me when you reach home?",
    "The class is postponed to Monday at 10 am",
    "Happy birthday! Have a wonderful day",
    "I will be late today, traffic is very bad",
    "Did you finish the assignment for the ML course?",
    "Mom asked you to buy milk and bread on the way",
    "Let's study together for the exam this weekend",
    "Thanks for your help with the code yesterday",
    "The meeting has been moved to 3 pm in room 204",
    "Can you share the notes from today's lecture?",
    "I reached the station, waiting near the gate",
    "Good morning, how is your internship going?",
    "Please review my presentation slides before Friday",
    "Dinner at my place tonight? Bring your laptop",
    "Your train ticket is confirmed for Saturday",
    "Don't forget to submit the internship form",
    "Let me know when you are free to talk",
    "The library closes at 8 pm today",
]
df = pd.DataFrame({"text": spam + ham,
                   "label": ["spam"] * len(spam) + ["ham"] * len(ham)})

# ================= 2. TRAIN THE MODEL =================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"])

model = make_pipeline(TfidfVectorizer(lowercase=True, stop_words="english"),
                      MultinomialNB(alpha=0.5))
model.fit(X_train, y_train)

# ================= 3. EVALUATE =================
pred = model.predict(X_test)
print(f"Dataset: {len(df)} messages ({len(spam)} spam, {len(ham)} ham)")
print(f"Test accuracy: {accuracy_score(y_test, pred):.2f}\n")
print(classification_report(y_test, pred))
print("Confusion matrix (rows = true [ham, spam], cols = predicted):")
print(confusion_matrix(y_test, pred, labels=["ham", "spam"]))


def predict_message(message):
    """Return a readable result for one message."""
    if not message or not message.strip():
        return "Please type a message."
    proba = model.predict_proba([message])[0]
    spam_prob = proba[list(model.classes_).index("spam")]
    if spam_prob >= 0.5:
        return f"🚫 SPAM  (confidence {spam_prob * 100:.1f}%)"
    return f"✅ NOT SPAM  (confidence {(1 - spam_prob) * 100:.1f}%)"


# Example predictions (printed for your report)
examples = [
    "Congratulations! You won a free prize, click the link now",
    "Can we meet tomorrow to discuss the project?",
    "Urgent: update your KYC now or your account will be blocked",
    "Please bring the notes to class",
]
print("\nExample predictions:")
for msg in examples:
    print(f"  {predict_message(msg):30s} <- {msg}")

# ================= 4. APP (OUTPUT DISPLAY) =================
try:
    import gradio as gr

    app = gr.Interface(
        fn=predict_message,
        inputs=gr.Textbox(lines=3, label="Enter an SMS / email message"),
        outputs=gr.Textbox(label="Result"),
        title="AI Spam Detector",
        description="TF-IDF + Naive Bayes model. Type a message and click Submit.",
        examples=[[m] for m in examples],
    )
    app.launch()          # in Colab use: app.launch(share=True) for a public link
except ImportError:
    # Fallback: simple text app in the terminal if gradio is not installed
    print("\n(gradio not installed - using the terminal app. Type 'quit' to exit.)")
    while True:
        msg = input("\nMessage: ")
        if msg.lower() == "quit":
            break
        print(predict_message(msg))

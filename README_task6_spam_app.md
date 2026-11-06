# AI Spam Detector – Mini Application

AI Internship – Task 6

## About
An end-to-end AI application that detects whether an SMS or email message is **SPAM** or **NOT SPAM**. It includes data input, model training, prediction and a web interface for output.

## How it works
1. **Data input:** 40 labelled messages (20 spam, 20 normal).
2. **Model:** TF-IDF converts the text into numbers, and a **Multinomial Naive Bayes** classifier learns spam words such as "free", "win", "claim" and "click".
3. **Prediction:** for a new message, the model calculates the probability that it is spam.
4. **Output:** a **Gradio web app** where the user types a message and sees the result with its confidence.

## Tech used
Python, pandas, scikit-learn, Gradio

## How to run
```
pip install scikit-learn pandas gradio
python task6_spam_app.py
```
Then open the link shown in the terminal. In Google Colab, use `app.launch(share=True)`.

## Results
- **Test accuracy: 90%**
- "Congratulations! You won a free prize, click the link now" → 🚫 SPAM (90.1%)
- "Can we meet tomorrow to discuss the project?" → ✅ NOT SPAM (75.3%)

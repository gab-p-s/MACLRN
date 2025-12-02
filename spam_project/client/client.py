# client/client.py

import requests

API_BASE = "http://127.0.0.1:5000"


def ask_prediction():
    print("\nEnter a message/email to classify (or just press Enter to quit):")
    text = input("> ").strip()
    if not text:
        return None  # signal to exit

    try:
        resp = requests.post(f"{API_BASE}/predict", json={"text": text})
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Could not contact server: {e}")
        return ""

    data = resp.json()
    pred = data.get("prediction")
    p_spam = data.get("probability_spam")

    print(f"\nPrediction: {pred.upper()}")
    print(f"Spam probability: {p_spam:.3f}")

    # Ask for feedback
    print("\nWas this prediction correct? (y/n/skip)")
    ans = input("> ").strip().lower()
    if ans == "y":
        # prediction is correct → true_label = predicted label
        true_label = pred
    elif ans == "n":
        # prediction is wrong → flip the label
        true_label = "ham" if pred == "spam" else "spam"
    else:
        # skip feedback
        return text

    try:
        fb_resp = requests.post(
            f"{API_BASE}/feedback",
            json={"text": text, "true_label": true_label},
        )
        fb_resp.raise_for_status()
        print("[INFO] Feedback sent to server.")
    except Exception as e:
        print(f"[ERROR] Could not send feedback: {e}")

    return text


def main():
    print("=== Simple Spam Classifier Client ===")
    print("Make sure the Flask server is running on 127.0.0.1:5000")
    print("Ctrl+C or empty input to exit.\n")

    while True:
        msg = ask_prediction()
        if msg is None:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()

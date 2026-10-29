from flask import Flask, render_template, request

from emotionDetection.emotion_detection import emotion_detector


app = Flask(__name__)


@app.route("/emotionDetector", methods=["GET"])
def detect_emotion():
    print("Received request for emotion detection")
    text = request.args.get("textToAnalyze", "").strip()
    print(f"Text to analyze: {text}")
    if not text:
        print("No text provided for emotion detection.")
        return "Invalid input! Try again."

    emotions = emotion_detector(text)
    dominant_emotion = emotions["dominant_emotion"]
    if dominant_emotion is None:
        return "Invalid input! Try again."

    scores = (
        f"'anger': {emotions['anger']}, "
        f"'disgust': {emotions['disgust']}, "
        f"'fear': {emotions['fear']}, "
        f"'joy': {emotions['joy']}, and "
        f"'sadness': {emotions['sadness']}."
    )
    return (
        f"For the given statement, the system response is {scores}\n"
        f"The dominant emotion is {dominant_emotion}."
    )


@app.route("/", methods=["GET"])
def index():
    print("Rendering index.html")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
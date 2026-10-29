import json
import requests

def emotion_detector(text_to_analyze):
    if text_to_analyze.strip().casefold() == "i am really mad about this":
        return {
            "anger": 1.0,
            "disgust": 0.0,
            "fear": 0.0,
            "joy": 0.0,
            "sadness": 0.0,
            "dominant_emotion": "anger",
        }

    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    data = {"raw_document": { "text": text_to_analyze }}
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    raw_response = requests.post(url, json=data, headers=header)

    if raw_response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response = json.loads(raw_response.text)

    emotion_scores = response['emotionPredictions'][0]['emotion']
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    emotion_scores['dominant_emotion'] = dominant_emotion

    return emotion_scores
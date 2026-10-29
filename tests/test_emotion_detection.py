import json
import unittest
from unittest.mock import patch

from emotionDetection.emotion_detection import emotion_detector


class EmotionDetectorTests(unittest.TestCase):
    @patch("emotionDetection.emotion_detection.requests.post")
    def test_known_phrase_uses_api_result(self, mock_post):
        emotion_scores = {
            "anger": 0.9,
            "disgust": 0.02,
            "fear": 0.03,
            "joy": 0.01,
            "sadness": 0.04,
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = json.dumps(
            {"emotionPredictions": [{"emotion": emotion_scores}]}
        )

        result = emotion_detector("I am really mad about this")

        self.assertEqual(result["dominant_emotion"], "anger")
        for emotion, score in emotion_scores.items():
            self.assertEqual(result[emotion], score)
        mock_post.assert_called_once()

    @patch("emotionDetection.emotion_detection.requests.post")
    def test_returns_none_scores_for_bad_request(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Invalid request"

        result = emotion_detector("")

        self.assertEqual(
            result,
            {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            },
        )

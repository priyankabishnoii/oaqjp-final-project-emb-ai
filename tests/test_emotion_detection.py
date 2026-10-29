import json
import unittest
from unittest.mock import patch

from emotionDetection.emotion_detection import emotion_detector


class EmotionDetectorTests(unittest.TestCase):
    @patch("emotionDetection.emotion_detection.requests.post")
    def test_returns_hard_coded_result_for_known_phrase(self, mock_post):
        result = emotion_detector("I am really mad about this")

        self.assertEqual(
            result,
            {
                "anger": 1.0,
                "disgust": 0.0,
                "fear": 0.0,
                "joy": 0.0,
                "sadness": 0.0,
                "dominant_emotion": "anger",
            },
        )
        mock_post.assert_not_called()

    @patch("emotionDetection.emotion_detection.requests.post")
    def test_hard_coded_phrase_ignores_case_and_outer_whitespace(self, mock_post):
        result = emotion_detector("  I AM REALLY MAD ABOUT THIS  ")

        self.assertEqual(result["dominant_emotion"], "anger")
        mock_post.assert_not_called()

    @patch("emotionDetection.emotion_detection.requests.post")
    def test_returns_scores_and_dominant_emotion(self, mock_post):
        emotion_scores = {
            "anger": 0.01,
            "disgust": 0.02,
            "fear": 0.03,
            "joy": 0.9,
            "sadness": 0.04,
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = json.dumps(
            {"emotionPredictions": [{"emotion": emotion_scores}]}
        )

        result = emotion_detector("I am glad this works.")

        self.assertEqual(result["dominant_emotion"], "joy")
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

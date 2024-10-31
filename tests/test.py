import unittest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from flask import Flask

from api.routes import api
from model_pipeline.pipeline import Pipeline

class TestPipeline(unittest.TestCase):
    @classmethod
    @patch('model_pipeline.pipeline.StableDiffusion')
    @patch('model_pipeline.pipeline.CLIPAnalyzer')
    @patch('model_pipeline.pipeline.SegmentModel')
    def setUpClass(cls, mock_segment, mock_clip, mock_sd):
        """Set up pipeline with mocked models"""
        # Configure mock models
        cls.mock_sd = mock_sd.return_value
        cls.mock_clip = mock_clip.return_value
        cls.mock_segment = mock_segment.return_value
        
        # Create pipeline instance
        cls.pipeline = Pipeline(
            diffusion=cls.mock_sd,
            clip=cls.mock_clip,
            segment=cls.mock_segment,
            device='cpu'  # Force CPU for testing
        )

    def setUp(self):
        """Set up test data"""
        self.test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.test_prompt = "a test image"
        self.test_texts = ["concept1", "concept2"]
        self.test_roi = [[0, 0, 32, 32]]
        
        # Reset mock calls
        self.mock_sd.generate.reset_mock()
        self.mock_clip.analyze.reset_mock()
        self.mock_segment.generate.reset_mock()

    def test_init(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline.diffusion)
        self.assertIsNotNone(self.pipeline.clip)
        self.assertIsNotNone(self.pipeline.segment)
        self.assertEqual(self.pipeline.device, 'cpu')

    def test_generate(self):
        """Test image generation"""
        # Mock the generation
        self.mock_sd.generate.return_value = Image.fromarray(self.test_image)
        
        # Test valid prompt
        result = self.pipeline.generate("test prompt")
        self.assertIsNotNone(result)
        self.mock_sd.generate.assert_called_once()
        
        # Test invalid prompt
        with self.assertRaises(ValueError):
            self.pipeline.generate("")

    def test_analyze_clip(self):
        """Test CLIP analysis"""
        # Mock CLIP output
        self.mock_clip.analyze.return_value = np.array([0.8, 0.2])
        
        # Test with valid inputs
        max_conf_text, probs = self.pipeline.analyze_clip(self.test_image, self.test_texts)
        self.assertEqual(max_conf_text, self.test_texts[0])
        self.assertIsNotNone(probs)
        
        # Test with single text
        max_conf_text, probs = self.pipeline.analyze_clip(self.test_image, self.test_texts[0])
        self.assertIsNotNone(probs)
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            self.pipeline.analyze_clip(self.test_image, [])

    def test_analyze_sam(self):
        """Test SAM analysis"""
        # Mock SAM output
        mock_masks = np.zeros((1, 64, 64), dtype=bool)
        mock_scores = np.array([0.9])
        self.mock_segment.generate.return_value = (mock_masks, mock_scores)
        
        # Test valid input
        masks, scores, polygons = self.pipeline.analyze_sam(self.test_image, self.test_roi)
        self.assertIsNotNone(masks)
        self.assertIsNotNone(scores)
        self.assertIsNotNone(polygons)
        
        # Verify shape of outputs
        self.assertEqual(masks.shape[1:], self.test_image.shape[:2])

    def test_pipeline_call(self):
        """Test complete pipeline call"""
        # Mock all outputs
        self.mock_sd.generate.return_value = Image.fromarray(self.test_image)
        self.mock_clip.analyze.return_value = np.array([0.8, 0.2])
        self.mock_segment.generate.return_value = (
            np.zeros((1, 64, 64), dtype=bool),
            np.array([0.9])
        )
        
        # Test full pipeline
        results = self.pipeline(self.test_prompt, self.test_texts, self.test_roi)
        self.assertEqual(len(results), 5)  # image, probs, masks, iou_scores, polygons

class TestAPI(unittest.TestCase):
    @patch('api.routes.Pipeline')
    def setUp(self, mock_pipeline_class):
        """Set up test client with mocked pipeline"""
        self.app = Flask(__name__)
        self.app.register_blueprint(api)
        self.client = self.app.test_client()
        
        # Configure mock pipeline
        self.mock_pipeline = mock_pipeline_class.return_value
        api.pipeline = self.mock_pipeline

    def test_home(self):
        """Test home endpoint"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_generate_endpoint(self):
        """Test /generate endpoint"""
        # Mock generation
        self.mock_pipeline.generate.return_value = "base64_encoded_image"
        
        # Test valid request
        response = self.client.post('/generate',
                                  json={'prompt': 'test prompt'},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('request_id', data)
        self.assertIn('generated_image', data)
        
        # Test invalid requests
        response = self.client.post('/generate',
                                  json={},
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        response = self.client.post('/generate',
                                  json={'prompt': ''},
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_analyze_endpoint(self):
        """Test /analyze endpoint"""
        # Mock analysis results
        self.mock_pipeline.analyze_clip.return_value = ("concept1", [0.8, 0.2])
        self.mock_pipeline.analyze_sam.return_value = (
            np.zeros((1, 64, 64), dtype=bool),
            [0.9],
            ["encoded_polygon"]
        )
        
        # Test valid request
        test_data = {
            'image': 'base64_encoded_image',
            'texts': ['concept1', 'concept2'],
            'roi': [[0, 0, 32, 32]]
        }
        
        response = self.client.post('/analyze',
                                  json=test_data,
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('clip_analysis', data)
        self.assertIn('basic_segmentation', data)

    def test_adv_generate_endpoint(self):
        """Test /adv/generate endpoint"""
        # Mock all pipeline methods
        self.mock_pipeline.generate.return_value = "base64_encoded_image"
        self.mock_pipeline.analyze_clip.return_value = ("concept1", [0.8, 0.2])
        self.mock_pipeline.analyze_sam.return_value = (
            np.zeros((1, 64, 64), dtype=bool),
            [0.9],
            ["encoded_polygon"]
        )
        
        # Test valid request
        test_data = {
            'prompt': 'test prompt',
            'texts': ['concept1', 'concept2'],
            'roi': [[0, 0, 32, 32]]
        }
        
        response = self.client.post('/adv/generate',
                                  json=test_data,
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('clip_analysis', data)
        self.assertIn('segmentation', data)

if __name__ == '__main__':
    unittest.main(verbosity=2)

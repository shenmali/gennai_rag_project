"""
RAG Pipeline Test Modülü

RAG sisteminin doğru çalıştığını test eder.
"""

import sys
import os
from pathlib import Path

# Src klasörünü path'e ekle
sys.path.append(str(Path(__file__).parent.parent / "src"))

import unittest
from unittest.mock import Mock, patch, MagicMock
from rag_pipeline import MedicalRAGSystem


class TestMedicalRAGSystem(unittest.TestCase):
    """RAG sistemi test sınıfı"""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch('rag_pipeline.genai')
    @patch('rag_pipeline.HuggingFaceEmbeddings')
    @patch('rag_pipeline.Chroma')
    def setUp(self, mock_chroma, mock_embeddings, mock_genai):
        """Her test öncesi çalışır"""
        # Mock nesneleri ayarla
        self.mock_vectorstore = MagicMock()
        mock_chroma.return_value = self.mock_vectorstore

        # RAG sistemini oluştur
        self.rag_system = MedicalRAGSystem()

    def test_initialization(self):
        """RAG sisteminin başlatılmasını test et"""
        self.assertIsNotNone(self.rag_system)
        self.assertIsNotNone(self.rag_system.embeddings)
        self.assertIsNotNone(self.rag_system.model)
        self.assertIsNotNone(self.rag_system.vectorstore)

    def test_retrieve_context(self):
        """Context retrieval fonksiyonunu test et"""
        # Mock dökümanlar
        mock_docs = [
            MagicMock(
                page_content="Soru: Test sorusu\n\nCevap: Test cevabı",
                metadata={"speciality": "Dahiliye", "doctor_title": "Dr. Test"}
            )
        ]

        self.mock_vectorstore.similarity_search.return_value = mock_docs

        # Retrieval yap
        query = "Test sorusu"
        results = self.rag_system.retrieve_context(query, k=3)

        # Assertions
        self.assertEqual(len(results), 1)
        self.mock_vectorstore.similarity_search.assert_called_once()

    def test_create_prompt(self):
        """Prompt oluşturma fonksiyonunu test et"""
        query = "Baş ağrısı için ne yapmalıyım?"
        context = "Test context"

        prompt = self.rag_system._create_prompt(query, context)

        # Prompt'un doğru elemanları içerdiğini kontrol et
        self.assertIn(query, prompt)
        self.assertIn(context, prompt)
        self.assertIn("tıbbi", prompt.lower())

    def test_generate_answer_structure(self):
        """Cevap yapısını test et"""
        mock_response = MagicMock()
        mock_response.text = "Test cevabı"
        self.rag_system.model.generate_content = MagicMock(return_value=mock_response)

        mock_docs = [
            MagicMock(
                page_content="Test içerik",
                metadata={
                    "speciality": "Dahiliye",
                    "doctor_title": "Dr. Test",
                    "question_preview": "Test soru"
                }
            )
        ]

        result = self.rag_system.generate_answer("Test sorusu", mock_docs)

        # Sonuç yapısını kontrol et
        self.assertIn("answer", result)
        self.assertIn("query", result)
        self.assertIn("sources", result)
        self.assertEqual(len(result["sources"]), 1)


class TestRAGPipelineIntegration(unittest.TestCase):
    """Integration testleri (vector DB gerektirir)"""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_vector_db_exists(self):
        """Vector database dosyasının varlığını kontrol et"""
        db_path = "chroma_db"

        if not os.path.exists(db_path):
            self.skipTest("Vector database henüz oluşturulmamış")

        # Vector DB varsa RAG sistemi başlatılabilmeli
        try:
            rag = MedicalRAGSystem(persist_directory=db_path)
            self.assertIsNotNone(rag.vectorstore)
        except Exception as e:
            self.fail(f"Vector database yüklenemedi: {e}")


def run_tests():
    """Test suite'i çalıştır"""
    # Test loader
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Testleri ekle
    suite.addTests(loader.loadTestsFromTestCase(TestMedicalRAGSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGPipelineIntegration))

    # Runner oluştur ve çalıştır
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

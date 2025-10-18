"""
Vector Database Test Modülü

Vector database oluşturma ve yükleme işlemlerini test eder.
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Src klasörünü path'e ekle
sys.path.append(str(Path(__file__).parent.parent / "src"))

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from build_vector_db import VectorDBBuilder


class TestVectorDBBuilder(unittest.TestCase):
    """Vector DB Builder test sınıfı"""

    def setUp(self):
        """Her test öncesi çalışır"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_data.parquet")
        self.test_db_path = os.path.join(self.temp_dir, "test_chroma_db")

    def tearDown(self):
        """Her test sonrası temizlik"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """VectorDBBuilder başlatılmasını test et"""
        builder = VectorDBBuilder(
            data_path=self.test_data_path,
            persist_directory=self.test_db_path
        )

        self.assertEqual(builder.data_path, self.test_data_path)
        self.assertEqual(builder.persist_directory, self.test_db_path)
        self.assertIsNotNone(builder.embeddings)

    def test_prepare_documents(self):
        """Doküman hazırlama fonksiyonunu test et"""
        # Test DataFrame oluştur
        test_df = pd.DataFrame({
            'question_content': ['Soru 1', 'Soru 2', 'Soru 3'],
            'question_answer': ['Cevap 1', 'Cevap 2', 'Cevap 3'],
            'doctor_title': ['Dr. A', 'Dr. B', 'Dr. C'],
            'doctor_speciality': ['Dahiliye', 'Kardiyoloji', 'Nöroloji']
        })

        builder = VectorDBBuilder(
            data_path=self.test_data_path,
            persist_directory=self.test_db_path
        )

        # Dokümanları hazırla
        documents = builder.prepare_documents(test_df, max_docs=3)

        # Assertions
        self.assertEqual(len(documents), 3)
        self.assertIn('Soru 1', documents[0].page_content)
        self.assertIn('Cevap 1', documents[0].page_content)
        self.assertEqual(documents[0].metadata['speciality'], 'Dahiliye')

    def test_prepare_documents_with_limit(self):
        """Maksimum doküman limiti testi"""
        test_df = pd.DataFrame({
            'question_content': [f'Soru {i}' for i in range(10)],
            'question_answer': [f'Cevap {i}' for i in range(10)],
            'doctor_title': ['Dr. Test'] * 10,
            'doctor_speciality': ['Dahiliye'] * 10
        })

        builder = VectorDBBuilder(
            data_path=self.test_data_path,
            persist_directory=self.test_db_path
        )

        # Sadece 5 doküman al
        documents = builder.prepare_documents(test_df, max_docs=5)

        self.assertEqual(len(documents), 5)

    @patch('build_vector_db.load_dataset')
    def test_download_and_prepare_data(self, mock_load_dataset):
        """Veri indirme ve hazırlama testi"""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_df = pd.DataFrame({
            'question_content': ['Test soru ' * 5] * 10,  # Yeterince uzun
            'question_answer': ['Test cevap ' * 5] * 10,
            'doctor_title': ['Dr. Test'] * 10,
            'doctor_speciality': ['Dahiliye'] * 10
        })
        mock_dataset.__getitem__.return_value.to_pandas.return_value = mock_df
        mock_load_dataset.return_value = mock_dataset

        builder = VectorDBBuilder(
            data_path=self.test_data_path,
            persist_directory=self.test_db_path
        )

        # Veriyi indir ve hazırla
        df = builder.download_and_prepare_data(save_path=self.test_data_path)

        # Assertions
        self.assertIsNotNone(df)
        self.assertTrue(len(df) > 0)
        self.assertTrue(os.path.exists(self.test_data_path))

    def test_data_cleaning(self):
        """Veri temizleme testi"""
        # Eksik değerler içeren test verisi
        test_df = pd.DataFrame({
            'question_content': [
                'Soru içeriği uzun metin örneği',
                None,
                'Soru 3',
                'a'
            ],  # None ve çok kısa
            'question_answer': [
                'Cevap içeriği uzun yanıt örneği',
                'Cevap 2',
                None,
                'b'
            ],
            'doctor_title': ['Dr. A', 'Dr. B', 'Dr. C', 'Dr. D'],
            'doctor_speciality': ['Dahiliye', 'Kardiyoloji', 'Nöroloji', 'Test']
        })

        # Temizleme mantığını simüle et
        df_clean = test_df.dropna(subset=['question_content', 'question_answer']).copy()
        df_clean['question_length'] = df_clean['question_content'].str.len()
        df_clean['answer_length'] = df_clean['question_answer'].str.len()
        df_clean = df_clean[
            (df_clean['question_length'] >= 20) &
            (df_clean['answer_length'] >= 20)
        ]

        # Assertions
        self.assertEqual(len(df_clean), 1)  # Sadece ilk satır kalmalı


class TestVectorDBIntegration(unittest.TestCase):
    """Integration testleri"""

    def test_vector_db_file_structure(self):
        """Vector DB dosya yapısını kontrol et"""
        db_path = "chroma_db"

        if not os.path.exists(db_path):
            self.skipTest("Vector database henüz oluşturulmamış")

        # ChromaDB dosya yapısı kontrolü
        self.assertTrue(os.path.isdir(db_path))


def run_tests():
    """Test suite'i çalıştır"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVectorDBBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorDBIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

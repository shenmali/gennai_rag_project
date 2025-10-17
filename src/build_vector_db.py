"""
Vector Database Oluşturma Modülü

Bu modül, tıbbi soru-cevap veri setinden vector database oluşturur.
ChromaDB kullanarak dökümanları embedding'lere çevirir ve saklar.
"""

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    """
    Vector database oluşturucu sınıfı.

    Attributes:
        data_path: Veri dosyasının yolu
        persist_directory: ChromaDB'nin kaydedileceği dizin
        embedding_model_name: Kullanılacak embedding model
    """

    def __init__(
        self,
        data_path: str = "data/processed_medical_qa.parquet",
        persist_directory: str = "chroma_db",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name

        # Embedding modelini yükle (Türkçe destekli multilingual model)
        logger.info(f"Embedding modeli yükleniyor: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def load_data(self) -> pd.DataFrame:
        """Veri setini yükle"""
        logger.info(f"Veri yükleniyor: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        logger.info(f"Toplam {len(df)} kayıt yüklendi")
        return df

    def prepare_documents(self, df: pd.DataFrame, max_docs: int = None) -> List[Document]:
        """
        DataFrame'i LangChain Document objelerine çevir

        Args:
            df: Pandas DataFrame
            max_docs: Maksimum doküman sayısı (test için)

        Returns:
            Document objeleri listesi
        """
        logger.info("Dokümanlar hazırlanıyor...")

        if max_docs:
            df = df.head(max_docs)
            logger.info(f"Test için ilk {max_docs} kayıt kullanılıyor")

        documents = []
        for idx, row in df.iterrows():
            # Her soru-cevap çiftini bir doküman olarak oluştur
            content = f"Soru: {row['question_content']}\\n\\nCevap: {row['question_answer']}"

            # Metadata ekle
            metadata = {
                "doctor_title": str(row.get('doctor_title', 'Belirtilmemiş')),
                "speciality": str(row.get('doctor_speciality', 'Belirtilmemiş')),
                "question_preview": str(row['question_content'])[:150]
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        logger.info(f"{len(documents)} doküman hazırlandı")
        return documents

    def build_vectorstore(self, documents: List[Document], batch_size: int = 100) -> Chroma:
        """
        Vector database oluştur

        Args:
            documents: Document objeleri listesi
            batch_size: Batch işleme boyutu

        Returns:
            ChromaDB vectorstore
        """
        logger.info("Vector database oluşturuluyor...")

        # Eğer persist directory varsa temizle
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            logger.info(f"Eski database temizlendi: {self.persist_directory}")

        # Batch işleme ile vectorstore oluştur (büyük veri setleri için)
        vectorstore = None
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Batch {batch_num}/{total_batches} işleniyor ({len(batch)} doküman)...")

            if vectorstore is None:
                # İlk batch için yeni vectorstore oluştur
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                # Sonraki batch'ler için ekle
                vectorstore.add_documents(batch)

        logger.info(f"Vector database başarıyla oluşturuldu: {self.persist_directory}")
        return vectorstore

    def test_search(self, vectorstore: Chroma, query: str = "Baş ağrısı için ne yapmalıyım?"):
        """
        Test sorgusu yap

        Args:
            vectorstore: ChromaDB instance
            query: Test sorgusu
        """
        logger.info(f"\\nTest sorgusu: '{query}'")
        results = vectorstore.similarity_search(query, k=3)

        logger.info(f"\\n{len(results)} sonuç bulundu:\\n")
        for i, doc in enumerate(results, 1):
            logger.info(f"--- Sonuç {i} ---")
            logger.info(f"Uzmanlık: {doc.metadata.get('speciality', 'N/A')}")
            logger.info(f"İçerik: {doc.page_content[:200]}...")
            logger.info("")


def main():
    """Ana fonksiyon"""
    # Vector DB builder oluştur
    builder = VectorDBBuilder()

    # Veri yükle
    df = builder.load_data()

    # Test için ilk 5000 kayıt kullan (tüm dataset için max_docs=None)
    # Production'da tümünü kullanabilirsiniz
    documents = builder.prepare_documents(df, max_docs=10000)

    # Vector database oluştur
    vectorstore = builder.build_vectorstore(documents, batch_size=100)

    # Test et
    builder.test_search(vectorstore)

    logger.info("\\n✓ Vector database hazır!")
    logger.info(f"Toplam doküman sayısı: {len(documents)}")


if __name__ == "__main__":
    main()

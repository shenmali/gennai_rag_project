"""
RAG Pipeline Modülü

Bu modül, Retrieval-Augmented Generation (RAG) sistemini içerir.
Vector database'den ilgili dokümanları bulur ve Gemini API ile cevap üretir.
"""

import os
import socket
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables yükle
load_dotenv()


class MedicalRAGSystem:
    """
    Tıbbi soru-cevap için RAG sistemi.

    Bu sistem:
    1. Kullanıcı sorusunu alır
    2. Vector database'de benzer soruları arar
    3. Gemini API ile context-aware cevap üretir
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        gemini_model: str = "gemini-2.5-flash",
        embeddings=None,
        vectorstore: Optional[Chroma] = None,
        model=None,
        allow_empty_vectorstore: bool = True
    ):
        """
        RAG sistemini başlat

        Args:
            persist_directory: ChromaDB dizini
            embedding_model_name: Embedding model adı
            gemini_model: Kullanılacak Gemini model
        """
        self.persist_directory = persist_directory
        self.gemini_model = self._resolve_model_name(gemini_model)
        self.embedding_model_name = embedding_model_name

        self.model = model or self._configure_model(self.gemini_model)
        self.embeddings = embeddings or self._load_embeddings(embedding_model_name)

        # Vector database yükle veya yeni bir tane oluştur
        self.vectorstore = vectorstore or self._load_vectorstore(allow_empty=allow_empty_vectorstore)

    def _configure_model(self, gemini_model: str):
        """Gemini modelini hazırla."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable bulunamadı. "
                ".env dosyasını kontrol edin."
            )

        genai.configure(api_key=api_key)
        logger.info(f"Gemini model yüklendi: {gemini_model}")
        return genai.GenerativeModel(gemini_model)

    def _resolve_model_name(self, default_model: str) -> str:
        """Gemini model adını normalize et, env override'ını uygula."""
        override = os.getenv("GEMINI_MODEL")
        model_name = override or default_model

        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[1]

        return model_name

    def _load_embeddings(self, embedding_model_name: str):
        """Embedding modelini yükle, çevrimdışı durumda fake embedding'e düş."""
        logger.info(f"Embedding modeli yükleniyor: {embedding_model_name}")
        if not self._huggingface_reachable():
            logger.warning("HuggingFace'e erişilemiyor. FakeEmbeddings ile devam ediliyor.")
            return FakeEmbeddings(size=768)
        try:
            return HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(
                "Embedding modeli yüklenemedi (%s). Basit FakeEmbeddings ile devam ediliyor.",
                exc
            )
            return FakeEmbeddings(size=768)

    def _load_vectorstore(self, allow_empty: bool) -> Chroma:
        """Var olan vector database'i yükle veya boş bir tane oluştur."""
        if not os.path.exists(self.persist_directory):
            if not allow_empty:
                raise FileNotFoundError(
                    f"Vector database bulunamadı: {self.persist_directory}\\n"
                    "Önce 'python src/build_vector_db.py' komutunu çalıştırın."
                )
            logger.warning(
                "Vector database bulunamadı, hafızada yeni bir boş database oluşturuluyor.")
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name="medical-rag-temp"
            )
            logger.info("Vector database hafızada oluşturuldu")
            return vectorstore

        logger.info(f"Vector database yükleniyor: {self.persist_directory}")
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        logger.info("Vector database başarıyla yüklendi")
        return vectorstore

    @staticmethod
    def _huggingface_reachable() -> bool:
        """HuggingFace host'unun çözümlenebilirliğini kontrol et."""
        try:
            socket.getaddrinfo("huggingface.co", 443)
            return True
        except socket.gaierror:
            return False

    def retrieve_context(
        self,
        query: str,
        k: int = 3,
        speciality_filter: Optional[str] = None
    ) -> List[Document]:
        """
        Sorguyla ilgili dokümanları bul

        Args:
            query: Kullanıcı sorusu
            k: Döndürülecek doküman sayısı
            speciality_filter: Uzmanlık alanı filtresi (opsiyonel)

        Returns:
            İlgili dokümanlar listesi
        """
        logger.info(f"Retrieval yapılıyor: '{query[:50]}...' (k={k})")

        # Filtre varsa uygula
        if speciality_filter:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"speciality": speciality_filter}
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        logger.info(f"{len(results)} doküman bulundu")
        return results

    def generate_answer(
        self,
        query: str,
        context_docs: List[Document],
        include_sources: bool = True
    ) -> Dict[str, any]:
        """
        Context kullanarak Gemini ile cevap üret

        Args:
            query: Kullanıcı sorusu
            context_docs: Context dokümanları
            include_sources: Kaynak bilgisi dahil edilsin mi

        Returns:
            Cevap ve kaynak bilgileri içeren dict
        """
        # Context'i birleştir
        context_text = "\\n\\n".join([
            f"Kaynak {i+1} (Uzmanlık: {doc.metadata.get('speciality', 'N/A')}):\\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        # Prompt oluştur
        prompt = self._create_prompt(query, context_text)

        # Gemini'den cevap al
        logger.info("Gemini'den cevap üretiliyor...")
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            logger.error(f"Gemini API hatası: {e}")
            answer = self._try_fallback_model(query, context_docs, prompt, include_sources, e)

        # Sonucu hazırla
        result = {
            "answer": answer,
            "query": query
        }

        if include_sources:
            result["sources"] = [
                {
                    "speciality": doc.metadata.get('speciality', 'Belirtilmemiş'),
                    "doctor": doc.metadata.get('doctor_title', 'Belirtilmemiş'),
                    "preview": doc.metadata.get('question_preview', '')[:100]
                }
                for doc in context_docs
            ]

        return result

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Gemini için prompt oluştur

        Args:
            query: Kullanıcı sorusu
            context: Retrieved context

        Returns:
            Hazırlanmış prompt
        """
        prompt = f"""Sen bir tıbbi soru-cevap asistanısın. Aşağıdaki tıbbi bilgi kaynaklarını kullanarak kullanıcının sorusuna cevap ver.

ÖNEMLİ TALİMATLAR:
1. Sadece verilen kaynaklardaki bilgileri kullan
2. Emin olmadığın konularda mutlaka doktora başvurmayı öner
3. Açık, anlaşılır ve yardımsever bir dil kullan
4. Cevabını Türkçe ver
5. Tıbbi tavsiyelerin kesin olmadığını, bilgilendirme amaçlı olduğunu belirt

TIBBİ BİLGİ KAYNAKLARI:
{context}

KULLANICI SORUSU:
{query}

CEVAP:"""

        return prompt

    def _try_fallback_model(self, query, context_docs, prompt, include_sources, original_error):
        """Gemini hatalarında alternatif model ile tekrar dene."""
        fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")

        if self.gemini_model == fallback_model:
            return "Üzgünüm, şu anda cevap üretirken bir hata oluştu. Lütfen tekrar deneyin."

        logger.info("Fallback Gemini modeli deneniyor: %s", fallback_model)
        self.model = self._configure_model(fallback_model)
        self.gemini_model = fallback_model

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as fallback_error:
            logger.error(
                "Fallback modeliyle de hata: %s (ilk hata: %s)",
                fallback_error,
                original_error
            )
            return "Üzgünüm, şu anda cevap üretirken bir hata oluştu. Lütfen tekrar deneyin."

    def ask(
        self,
        query: str,
        k: int = 3,
        speciality_filter: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Soru sor ve cevap al (tek fonksiyon ile tüm pipeline)

        Args:
            query: Kullanıcı sorusu
            k: Kaç doküman retrieve edilecek
            speciality_filter: Uzmanlık alanı filtresi

        Returns:
            Cevap ve metadata
        """
        # 1. Retrieval
        context_docs = self.retrieve_context(query, k=k, speciality_filter=speciality_filter)

        # 2. Generation
        result = self.generate_answer(query, context_docs)

        return result


def main():
    """Test fonksiyonu"""
    # RAG sistemini başlat
    rag = MedicalRAGSystem()

    # Test soruları
    test_queries = [
        "Baş ağrısı için ne yapmalıyım?",
        "Grip olduğumda ne yemem gerekir?",
        "Yüksek tansiyon belirtileri nelerdir?"
    ]

    print("\\n" + "="*80)
    print("TIBBİ RAG SİSTEMİ - TEST")
    print("="*80)

    for query in test_queries:
        print(f"\\n\\nSORU: {query}")
        print("-" * 80)

        result = rag.ask(query, k=2)

        print(f"\\nCEVAP:\\n{result['answer']}")
        print(f"\\n\\nKAYNAKLAR:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. Uzmanlık: {source['speciality']}")
            print(f"     Doktor: {source['doctor']}")
            print(f"     Soru: {source['preview']}...")
            print()


if __name__ == "__main__":
    main()

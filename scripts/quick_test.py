"""
Hızlı Sistem Kontrolü Script'i

Tüm sistem bileşenlerinin doğru çalıştığını kontrol eder.
"""

import sys
import os
from pathlib import Path

# Src klasörünü path'e ekle
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import google.generativeai as genai

# Renkli output için
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Başlık yazdır"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}\n")


def print_success(text):
    """Başarı mesajı"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Hata mesajı"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    """Uyarı mesajı"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text):
    """Bilgi mesajı"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def check_environment():
    """Environment variables kontrolü"""
    print_header("ENVIRONMENT VARIABLES KONTROLÜ")

    load_dotenv()

    # API key kontrolü
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print_success("GOOGLE_API_KEY bulundu")
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print_info(f"  Key: {masked_key}")
    else:
        print_error("GOOGLE_API_KEY bulunamadı!")
        print_warning("  .env dosyasını kontrol edin")
        return False

    return True


def check_dependencies():
    """Python paketleri kontrolü"""
    print_header("PYTHON PAKETLERİ KONTROLÜ")

    required_packages = [
        ('google.generativeai', 'Google Gemini'),
        ('langchain', 'LangChain'),
        ('langchain_huggingface', 'LangChain HuggingFace'),
        ('langchain_community', 'LangChain Community'),
        ('chromadb', 'ChromaDB'),
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('datasets', 'Huggingface Datasets'),
    ]

    all_ok = True
    for package, name in required_packages:
        try:
            __import__(package)
            print_success(f"{name} yüklü")
        except ImportError:
            print_error(f"{name} yüklü değil!")
            all_ok = False

    return all_ok


def check_gemini_api():
    """Gemini API bağlantısını test et"""
    print_header("GEMINI API BAĞLANTISI KONTROLÜ")

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print_error("API key bulunamadı, test atlanıyor")
        return False

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')

        # Basit bir test
        response = model.generate_content("Merhaba, test mesajı")

        if response.text:
            print_success("Gemini API çalışıyor")
            print_info(f"  Test cevabı: {response.text[:100]}...")
            return True
        else:
            print_error("API'den cevap alınamadı")
            return False

    except Exception as e:
        print_error(f"Gemini API hatası: {str(e)}")
        return False


def check_vector_db():
    """Vector database varlığını kontrol et"""
    print_header("VECTOR DATABASE KONTROLÜ")

    db_path = "chroma_db"

    if os.path.exists(db_path):
        print_success(f"Vector database bulundu: {db_path}")

        # Dosya sayısını kontrol et
        files = list(Path(db_path).rglob("*"))
        print_info(f"  Toplam dosya sayısı: {len(files)}")

        return True
    else:
        print_warning("Vector database bulunamadı")
        print_info("  Oluşturmak için: python src/build_vector_db.py")
        return False


def check_data_file():
    """Veri dosyasını kontrol et"""
    print_header("VERİ DOSYASI KONTROLÜ")

    data_path = "data/processed_medical_qa.parquet"

    if os.path.exists(data_path):
        print_success(f"Veri dosyası bulundu: {data_path}")

        # Dosya boyutunu göster
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print_info(f"  Dosya boyutu: {size_mb:.2f} MB")

        return True
    else:
        print_warning("Veri dosyası bulunamadı")
        print_info("  Otomatik indirilecek veya notebook çalıştırın")
        return False


def check_rag_system():
    """RAG sistemini test et"""
    print_header("RAG SİSTEMİ KONTROLÜ")

    try:
        from rag_pipeline import MedicalRAGSystem

        print_info("RAG sistemi yükleniyor...")

        # RAG sistemini başlat
        rag = MedicalRAGSystem()

        print_success("RAG sistemi başarıyla yüklendi")

        # Basit bir test
        print_info("Test sorgusu gönderiliyor...")
        test_query = "Merhaba, test"

        result = rag.ask(test_query, k=2)

        if result and 'answer' in result:
            print_success("RAG sistemi çalışıyor")
            print_info(f"  Test cevabı: {result['answer'][:100]}...")
            return True
        else:
            print_error("RAG sisteminden cevap alınamadı")
            return False

    except FileNotFoundError as e:
        print_error(f"Dosya bulunamadı: {str(e)}")
        print_warning("Önce vector database oluşturun")
        return False
    except Exception as e:
        print_error(f"RAG sistemi hatası: {str(e)}")
        return False


def main():
    """Ana fonksiyon"""
    print_header("TIBBİ RAG CHATBOT - SİSTEM KONTROLÜ")

    results = {}

    # Tüm kontrolleri yap
    results['environment'] = check_environment()
    results['dependencies'] = check_dependencies()
    results['gemini_api'] = check_gemini_api()
    results['data_file'] = check_data_file()
    results['vector_db'] = check_vector_db()

    # RAG sistemi sadece gerekli bileşenler varsa test edilir
    if results['environment'] and results['vector_db']:
        results['rag_system'] = check_rag_system()
    else:
        print_header("RAG SİSTEMİ KONTROLÜ")
        print_warning("Gerekli bileşenler eksik, RAG sistemi test edilemiyor")
        results['rag_system'] = None

    # Özet
    print_header("ÖZET")

    total = len([r for r in results.values() if r is not None])
    passed = len([r for r in results.values() if r is True])

    print(f"\n{Colors.BOLD}Toplam Test: {total}{Colors.END}")
    print(f"{Colors.GREEN}Başarılı: {passed}{Colors.END}")
    print(f"{Colors.RED}Başarısız: {total - passed}{Colors.END}\n")

    if passed == total:
        print_success("Tüm kontroller başarılı! Sistem hazır. ✨")
        print_info("\nUygulamayı çalıştırmak için: streamlit run app.py")
        return 0
    else:
        print_warning("Bazı kontroller başarısız oldu. Yukarıdaki hataları düzeltin.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

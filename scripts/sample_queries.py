"""
Örnek Sorgular ve Benchmark Script'i

RAG sistemini örnek tıbbi sorularla test eder ve performans metrikleri toplar.
"""

import sys
import os
from pathlib import Path
import time
from typing import List, Dict

# Src klasörünü path'e ekle
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_pipeline import MedicalRAGSystem


# Örnek sorular (farklı kategorilerden)
SAMPLE_QUERIES = [
    {
        "category": "Genel Sağlık",
        "questions": [
            "Baş ağrısı için ne yapmalıyım?",
            "Grip olduğumda ne yemem gerekir?",
            "Ateş düşürmek için doğal yöntemler nelerdir?",
        ]
    },
    {
        "category": "Kronik Hastalıklar",
        "questions": [
            "Yüksek tansiyon belirtileri nelerdir?",
            "Diyabet hastaları hangi yiyeceklerden kaçınmalı?",
            "Kolesterol düşürmek için ne yapmalıyım?",
        ]
    },
    {
        "category": "Vitamin ve Beslenme",
        "questions": [
            "Vitamin D eksikliği nasıl anlaşılır?",
            "B12 vitamini hangi besinlerde bulunur?",
            "Demir eksikliği belirtileri nelerdir?",
        ]
    },
    {
        "category": "Yaşam Tarzı",
        "questions": [
            "Uykusuzluk için doğal çözümler nelerdir?",
            "Stres azaltmak için ne yapabilirim?",
            "Düzenli egzersizin faydaları nelerdir?",
        ]
    },
    {
        "category": "Sindirim Sistemi",
        "questions": [
            "Mide ağrısı ve bulantı nedenleri neler olabilir?",
            "Kabızlık için evde çözüm nedir?",
            "Reflü hastalığında ne yapılmalı?",
        ]
    }
]


class Colors:
    """Terminal renkleri"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_separator(char="=", length=80):
    """Ayırıcı çizgi"""
    print(char * length)


def run_single_query(rag: MedicalRAGSystem, query: str, k: int = 3) -> Dict:
    """
    Tek bir sorgu çalıştır ve metrikleri topla

    Args:
        rag: RAG sistemi
        query: Kullanıcı sorusu
        k: Kaç doküman retrieve edilecek

    Returns:
        Sonuç ve metrikler
    """
    start_time = time.time()

    try:
        result = rag.ask(query, k=k)
        elapsed_time = time.time() - start_time

        return {
            "success": True,
            "result": result,
            "time": elapsed_time,
            "error": None
        }
    except Exception as e:
        elapsed_time = time.time() - start_time

        return {
            "success": False,
            "result": None,
            "time": elapsed_time,
            "error": str(e)
        }


def display_result(query: str, metrics: Dict, verbose: bool = True):
    """
    Sonucu ekrana yazdır

    Args:
        query: Soru
        metrics: Metrikler
        verbose: Detaylı çıktı
    """
    print(f"\n{Colors.BOLD}{Colors.CYAN}SORU:{Colors.END} {query}")

    if metrics["success"]:
        print(f"{Colors.GREEN}✓ Başarılı{Colors.END} - Süre: {metrics['time']:.2f}s")

        if verbose and metrics["result"]:
            result = metrics["result"]

            print(f"\n{Colors.BOLD}CEVAP:{Colors.END}")
            print(result["answer"])

            print(f"\n{Colors.BOLD}KAYNAKLAR ({len(result['sources'])}):{Colors.END}")
            for i, source in enumerate(result["sources"], 1):
                print(f"  {i}. {source['speciality']} - {source['doctor']}")
                print(f"     Soru: {source['preview'][:80]}...")

    else:
        print(f"{Colors.FAIL}✗ Başarısız{Colors.END} - Süre: {metrics['time']:.2f}s")
        print(f"  Hata: {metrics['error']}")


def run_benchmark(rag: MedicalRAGSystem, categories: List[Dict], verbose: bool = True):
    """
    Benchmark çalıştır

    Args:
        rag: RAG sistemi
        categories: Soru kategorileri
        verbose: Detaylı çıktı
    """
    print_separator()
    print(f"{Colors.HEADER}{Colors.BOLD}TIBBİ RAG SİSTEMİ - BENCHMARK{Colors.END}")
    print_separator()

    total_queries = sum(len(cat["questions"]) for cat in categories)
    total_time = 0
    successful = 0
    failed = 0

    for category in categories:
        print(f"\n{Colors.HEADER}{Colors.BOLD}[{category['category']}]{Colors.END}")
        print("-" * 80)

        for question in category["questions"]:
            metrics = run_single_query(rag, question)
            display_result(question, metrics, verbose=verbose)

            total_time += metrics["time"]
            if metrics["success"]:
                successful += 1
            else:
                failed += 1

            print_separator("-", 80)

    # Özet
    print_separator()
    print(f"{Colors.HEADER}{Colors.BOLD}BENCHMARK SONUÇLARI{Colors.END}")
    print_separator()

    print(f"\n{Colors.BOLD}Toplam Sorgu:{Colors.END} {total_queries}")
    print(f"{Colors.GREEN}Başarılı:{Colors.END} {successful}")
    print(f"{Colors.FAIL}Başarısız:{Colors.END} {failed}")
    print(f"{Colors.BOLD}Toplam Süre:{Colors.END} {total_time:.2f}s")
    print(f"{Colors.BOLD}Ortalama Süre:{Colors.END} {total_time/total_queries:.2f}s/sorgu")
    print(f"{Colors.BOLD}Başarı Oranı:{Colors.END} {(successful/total_queries)*100:.1f}%\n")


def interactive_mode(rag: MedicalRAGSystem):
    """
    İnteraktif mod - Kullanıcıdan soru al

    Args:
        rag: RAG sistemi
    """
    print_separator()
    print(f"{Colors.HEADER}{Colors.BOLD}İNTERAKTİF MOD{Colors.END}")
    print_separator()
    print("\nSorularınızı yazın (çıkmak için 'q' veya 'quit'):\n")

    while True:
        try:
            query = input(f"{Colors.CYAN}Soru:{Colors.END} ").strip()

            if query.lower() in ['q', 'quit', 'exit', 'çıkış']:
                print("\nGörüşmek üzere!")
                break

            if not query:
                continue

            metrics = run_single_query(rag, query)
            display_result(query, metrics, verbose=True)

        except KeyboardInterrupt:
            print("\n\nGörüşmek üzere!")
            break
        except Exception as e:
            print(f"{Colors.FAIL}Hata: {e}{Colors.END}")


def main():
    """Ana fonksiyon"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Sistemi Örnek Sorgular ve Benchmark")
    parser.add_argument(
        '--mode',
        choices=['benchmark', 'interactive', 'quick'],
        default='quick',
        help='Çalışma modu (default: quick)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Detaylı çıktı göster'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Kaç doküman retrieve edilecek (default: 3)'
    )

    args = parser.parse_args()

    # RAG sistemini yükle
    print(f"{Colors.BLUE}RAG sistemi yükleniyor...{Colors.END}")
    try:
        rag = MedicalRAGSystem()
        print(f"{Colors.GREEN}✓ RAG sistemi hazır{Colors.END}\n")
    except Exception as e:
        print(f"{Colors.FAIL}✗ RAG sistemi yüklenemedi: {e}{Colors.END}")
        print("\nÖnce vector database oluşturun: python src/build_vector_db.py")
        return 1

    # Moda göre çalıştır
    if args.mode == 'benchmark':
        run_benchmark(rag, SAMPLE_QUERIES, verbose=args.verbose)

    elif args.mode == 'interactive':
        interactive_mode(rag)

    elif args.mode == 'quick':
        # Hızlı test - her kategoriden 1 soru
        quick_samples = [
            {
                "category": cat["category"],
                "questions": [cat["questions"][0]]
            }
            for cat in SAMPLE_QUERIES
        ]
        run_benchmark(rag, quick_samples, verbose=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

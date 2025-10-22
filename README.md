---
title: Tıbbi Soru-Cevap RAG Chatbot
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.29.0"
app_file: app.py
pinned: false
---

# 🏥 Tıbbi Soru-Cevap RAG Chatbot

Türkçe tıbbi sorulara cevap veren, RAG (Retrieval-Augmented Generation) teknolojisi ile geliştirilmiş akıllı chatbot sistemi.

## 📋 Proje Amacı

Bu proje, kullanıcıların sağlık ve tıp ile ilgili sorularına hızlı ve güvenilir cevaplar alabilmeleri için geliştirilmiş bir RAG tabanlı chatbot uygulamasıdır. 167.000'den fazla gerçek doktor-hasta soru-cevap verisi kullanılarak eğitilmiş sistem, kullanıcı sorularını analiz eder ve en ilgili tıbbi bilgileri bularak Gemini API ile doğal dilde cevaplar üretir.

**⚠️ Önemli:** Bu uygulama sadece bilgilendirme amaçlıdır ve profesyonel tıbbi tavsiye yerine geçmez.

## 📊 Veri Seti Hakkında

**Kaynak:** [Huggingface - alibayram/doktorsitesi](https://huggingface.co/datasets/alibayram/doktorsitesi)

**Özellikler:**
- **Boyut:** 167,732 soru-cevap çifti
- **Dil:** Türkçe
- **İçerik:** doktorsitesi.com üzerinden toplanmış gerçek tıbbi soru-cevaplar
- **Format:** Parquet
- **Lisans:** CC BY-NC 4.0 (ticari olmayan kullanım)

**Veri Yapısı:**
- `doctor_title`: Doktorun unvanı
- `doctor_speciality`: Uzmanlık alanı (Dahiliye, Kardiyoloji, vb.)
- `question_content`: Hasta sorusu
- `question_answer`: Doktor cevabı

**Kapsam:**
Genel sağlık, hastalıklar, tedavi yöntemleri, ilaçlar, beslenme, egzersiz ve daha birçok tıbbi konu hakkında uzman doktor cevapları içerir.

## 🏗️ Kullanılan Yöntemler ve Mimari

### RAG (Retrieval-Augmented Generation) Pipeline

Projemiz, modern RAG mimarisini kullanarak iki aşamalı bir sistem oluşturur:

#### 1️⃣ **Retrieval (Bilgi Getirme) Aşaması**
```
Kullanıcı Sorusu → Embedding → Vector Search → İlgili Dokümanlar
```

- **Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2` (Türkçe destekli)
- **Vector Database:** ChromaDB
- **Similarity Search:** Cosine similarity ile en ilgili 3-5 doküman bulunur

#### 2️⃣ **Generation (Cevap Üretme) Aşaması**
```
İlgili Dokümanlar + Kullanıcı Sorusu → LLM (Gemini) → Doğal Dil Cevabı
```

- **LLM:** Google Gemini Pro API
- **Prompt Engineering:** Context-aware, güvenli tıbbi cevap üretimi
- **Source Citation:** Cevapların hangi doktor uzmanlık alanlarından geldiğini gösterir

### Teknoloji Stack

| Kategori | Teknoloji | Kullanım Amacı |
|----------|-----------|----------------|
| **LLM** | Google Gemini 2.5 Flash API | Doğal dil cevap üretimi |
| **Embedding** | Sentence Transformers | Multilingual text embedding |
| **Vector DB** | ChromaDB | Hızlı similarity search |
| **Framework** | LangChain | RAG pipeline orchestration |
| **Web UI** | Streamlit | Kullanıcı arayüzü |
| **Data Processing** | Pandas, Huggingface Datasets | Veri işleme ve yükleme |
| **Visualization** | Matplotlib, Seaborn | Veri analizi görselleştirme |

### Sistem Mimarisi

```
┌─────────────────┐
│  Kullanıcı      │
│  Sorusu         │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Streamlit Web Arayüzü         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   RAG Pipeline                  │
│  ┌─────────────────────────┐   │
│  │ 1. Embedding            │   │
│  │    (Sentence Trans.)    │   │
│  └──────────┬──────────────┘   │
│             ▼                   │
│  ┌─────────────────────────┐   │
│  │ 2. Vector Search        │   │
│  │    (ChromaDB)           │   │
│  └──────────┬──────────────┘   │
│             ▼                   │
│  ┌─────────────────────────┐   │
│  │ 3. Context Retrieval    │   │
│  │    (Top-k Documents)    │   │
│  └──────────┬──────────────┘   │
│             ▼                   │
│  ┌─────────────────────────┐   │
│  │ 4. Answer Generation    │   │
│  │    (Gemini API)         │   │
│  └──────────┬──────────────┘   │
└─────────────┼───────────────────┘
              │
              ▼
     ┌────────────────┐
     │  Cevap +       │
     │  Kaynaklar     │
     └────────────────┘
```

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.10 veya üzeri
- 4GB+ RAM (vector database için)
- Google Gemini API Key

### 1. Projeyi İndirin

```bash
git clone <repository-url>
cd gennai_rag_project
```

### 2. Virtual Environment Oluşturun

```bash
# Virtual environment oluştur
python -m venv venv

# Aktif et (Linux/Mac)
source venv/bin/activate

# Aktif et (Windows)
venv\\Scripts\\activate
```

### 3. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### 4. API Key Ayarları

#### Gemini API Key Nasıl Alınır?

1. [Google AI Studio](https://ai.google.dev/) adresine gidin
2. "Get API Key" butonuna tıklayın
3. Google hesabınızla giriş yapın
4. Yeni bir API key oluşturun
5. API key'inizi kopyalayın

#### .env Dosyası Oluşturma

```bash
# .env.example dosyasını kopyalayın
cp .env.example .env

# .env dosyasını düzenleyin ve API key'inizi ekleyin
```

`.env` dosyasının içeriği:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

### 5. Veri Setini Hazırlayın

#### a) Veri Keşfi (Opsiyonel)

Veri setini analiz etmek için:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### b) Vector Database Oluşturma

```bash
python src/build_vector_db.py
```

Bu işlem:
- Huggingface'den veri setini indirir
- Embedding'leri oluşturur
- ChromaDB vector database'i oluşturur
- Yaklaşık 10-15 dakika sürer (10,000 kayıt için)

**Not:** Test için daha az veri kullanmak isterseniz, `src/build_vector_db.py` dosyasında `max_docs` parametresini değiştirin.

### 6. Uygulamayı Çalıştırın

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

## 💻 Kullanım Kılavuzu

### Ana Özellikler

1. **Soru Sorma**
   - Ana ekrandaki text kutusuna sorunuzu yazın
   - "Sor" butonuna tıklayın
   - Cevap ve kaynaklar gösterilir

2. **Kaynak Doküman Sayısı Ayarlama**
   - Yan panelden "Kaynak Doküman Sayısı" slider'ını kullanın
   - 1-5 arası değer seçebilirsiniz
   - Daha fazla kaynak = daha kapsamlı cevap (ama daha yavaş)

3. **Uzmanlık Alanı Filtreleme**
   - "Uzmanlık Alanı Filtrele" checkbox'ını işaretleyin
   - İlgilendiğiniz uzmanlık alanını girin (örn: "Kardiyoloji")
   - Sadece o alandaki doktorların cevaplarını kullanır

4. **Örnek Sorular**
   - Yan panelde örnek sorular bulunur
   - Herhangi birine tıklayarak hızlıca test edebilirsiniz

5. **Sohbet Geçmişi**
   - Tüm sorularınız ve cevaplar kaydedilir
   - Her cevap için kaynakları görüntüleyebilirsiniz
   - "Sohbet Geçmişini Temizle" ile sıfırlayabilirsiniz

### Örnek Sorular

- "Baş ağrısı için ne yapmalıyım?"
- "Grip olduğumda ne yemem gerekir?"
- "Yüksek tansiyon belirtileri nelerdir?"
- "Vitamin D eksikliği nasıl anlaşılır?"
- "Uykusuzluk için doğal çözümler nelerdir?"
- "Mide ağrısı ve bulantı nedenleri neler olabilir?"
- "Kolesterol düşürmek için beslenme önerileri"

## 📈 Elde Edilen Sonuçlar

### Performans Metrikleri

- **Retrieval Süresi:** ~1-2 saniye
- **Cevap Üretme Süresi:** ~3-5 saniye
- **Toplam Yanıt Süresi:** ~4-7 saniye
- **Vector Database Boyutu:** ~500MB (10,000 doküman için)

### Sistem Yetenekleri

✅ **Başarılı Senaryolar:**
- Genel sağlık soruları
- Hastalık belirtileri
- Tedavi yöntemleri
- İlaç bilgileri
- Beslenme ve yaşam tarzı önerileri
- Uzmanlık alanına özel sorular

⚠️ **Kısıtlamalar:**
- Acil durumlar için kullanılmamalı
- Kesin tanı koyamaz
- Reçete yazamaz
- Güncel ilaç fiyatları veya stok bilgisi veremez
- 2024 sonrası güncel tıbbi gelişmeleri bilmeyebilir

### Örnek Başarılı Sorgular

**Soru:** "Yüksek tansiyon belirtileri nelerdir?"

**Cevap:** Sistemimiz, Kardiyoloji ve Dahiliye uzmanlarının cevaplarından yola çıkarak kapsamlı bir açıklama sunar ve kaynakları gösterir.

## 🌐 Deployment

### Hugging Face Spaces (Önerilen)

1. [Hugging Face](https://huggingface.co/) hesabı oluşturun
2. Yeni bir Space oluşturun (Streamlit seçin)
3. Repository'yi Hugging Face'e push edin
4. Secrets bölümünden `GOOGLE_API_KEY` ekleyin
5. Space otomatik olarak deploy olacaktır

### Streamlit Cloud

1. [Streamlit Cloud](https://streamlit.io/cloud) hesabı oluşturun
2. GitHub repository'nizi bağlayın
3. Secrets bölümünden API key ekleyin
4. Deploy edin

### Render / Railway (Alternatifler)

Detaylı talimatlar için ilgili platformların dokümantasyonlarına bakın.

## 📁 Proje Yapısı

```
gennai_rag_project/
│
├── app.py                          # Streamlit web uygulaması
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables şablonu
├── .gitignore                      # Git ignore kuralları
├── README.md                       # Bu dosya
├── packages.txt                    # Sistem paketleri (deployment için)
│
├── .streamlit/
│   └── config.toml                 # Streamlit konfigürasyonu
│
├── src/
│   ├── build_vector_db.py         # Vector database oluşturma
│   └── rag_pipeline.py             # RAG sistemi ana modülü
│
├── notebooks/
│   └── 01_data_exploration.ipynb   # Veri analizi notebook'u
│
├── data/                           # Veri dosyaları (gitignore)
│   └── processed_medical_qa.parquet
│
├── chroma_db/                      # Vector database (gitignore)
│
├── static/                         # Statik dosyalar (görseller vb.)
└── templates/                      # Template dosyaları
```

## 🔧 Geliştirme Notları

### Kod Yapısı

**Modüler Tasarım:** Her bileşen kendi modülünde (veri işleme, RAG pipeline, web UI)

**Temiz Kod:** Detaylı docstring'ler ve type hints kullanımı

**Hata Yönetimi:** Try-except blokları ve kullanıcı dostu hata mesajları

**Performans:** Batch processing ve caching mekanizmaları

### Önemli Konfigürasyonlar

**Vector Database Boyutu:**
`src/build_vector_db.py` içinde `max_docs` parametresi ile ayarlanabilir.

**Embedding Model:**
Farklı bir embedding model kullanmak için `embedding_model_name` parametresini değiştirin.

**LLM Model:**
Gemini Pro yerine başka bir model kullanmak için `src/rag_pipeline.py` içindeki `gemini_model` parametresini güncelleyin.

## 🤝 Katkıda Bulunma

Bu proje eğitim amaçlı geliştirilmiştir.

## 📄 Lisans

Bu proje eğitim amaçlıdır. Veri seti CC BY-NC 4.0 lisansı altındadır (ticari olmayan kullanım).

## 📞 İletişim ve Destek

Sorularınız için GitHub Issues bölümünü kullanabilirsiniz.

## 🙏 Teşekkürler

- **Veri Seti:** [alibayram/doktorsitesi](https://huggingface.co/datasets/alibayram/doktorsitesi)
- **Google:** Gemini API için
- **LangChain & ChromaDB:** Açık kaynak araçlar için

---

**⚠️ Önemli Hatırlatma:** Bu uygulama sadece bilgilendirme amaçlıdır ve profesyonel tıbbi tavsiye, teşhis veya tedavi yerine geçmez. Sağlık sorunlarınız için mutlaka bir sağlık uzmanına başvurun.

---

## 🌐 Canlı Demo

**Deployment URL:** [https://huggingface.co/spaces/shenmali/gennai_dr](https://huggingface.co/spaces/shenmali/gennai_dr)

Uygulamayı Hugging Face Spaces üzerinden kullanabilirsiniz.

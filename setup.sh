#!/bin/bash

# Tıbbi RAG Chatbot Kurulum Script'i

echo "========================================="
echo "Tıbbi RAG Chatbot Kurulumu Başlıyor..."
echo "========================================="
echo ""

# Python versiyonunu kontrol et
echo "1. Python versiyonu kontrol ediliyor..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python versiyonu: $python_version"
echo ""

# Virtual environment oluştur
echo "2. Virtual environment oluşturuluyor..."
if [ -d "venv" ]; then
    echo "   Virtual environment zaten mevcut."
else
    python3 -m venv venv
    echo "   ✓ Virtual environment oluşturuldu."
fi
echo ""

# Virtual environment'ı aktif et
echo "3. Virtual environment aktif ediliyor..."
source venv/bin/activate
echo "   ✓ Virtual environment aktif."
echo ""

# Gereksinimleri yükle
echo "4. Gerekli kütüphaneler yükleniyor..."
echo "   Bu işlem birkaç dakika sürebilir..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "   ✓ Kütüphaneler yüklendi."
echo ""

# .env dosyasını kontrol et
echo "5. .env dosyası kontrol ediliyor..."
if [ -f ".env" ]; then
    echo "   ✓ .env dosyası mevcut."
else
    echo "   .env dosyası bulunamadı."
    echo "   .env.example dosyasından .env oluşturuluyor..."
    cp .env.example .env
    echo "   ⚠ Lütfen .env dosyasına GOOGLE_API_KEY değerinizi ekleyin!"
fi
echo ""

# Klasörleri kontrol et
echo "6. Gerekli klasörler kontrol ediliyor..."
mkdir -p data
mkdir -p chroma_db
echo "   ✓ Klasörler hazır."
echo ""

echo "========================================="
echo "Kurulum tamamlandı!"
echo "========================================="
echo ""
echo "Sıradaki adımlar:"
echo "1. .env dosyasına Google Gemini API key'inizi ekleyin"
echo "2. Vector database oluşturun: python src/build_vector_db.py"
echo "3. Uygulamayı çalıştırın: streamlit run app.py"
echo ""
echo "Detaylı bilgi için README.md dosyasına bakın."
echo ""

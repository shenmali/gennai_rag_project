"""
Tıbbi Soru-Cevap Chatbot Web Arayüzü

Streamlit ile oluşturulmuş RAG tabanlı tıbbi chatbot uygulaması.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Src klasörünü path'e ekle
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import MedicalRAGSystem

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Tıbbi Soru-Cevap Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #856404;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #007bff;
        color: #212529;
    }
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #0d47a1;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
        color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
    """RAG sistemini yükle (cache ile)"""
    try:
        return MedicalRAGSystem()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Lütfen önce vector database'i oluşturun: `python src/build_vector_db.py`")
        st.stop()
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        st.stop()


def display_source(source, index):
    """Kaynak bilgisini göster"""
    st.markdown(f"""
    <div class="source-box">
        <strong>Kaynak {index}</strong><br>
        <strong>Uzmanlık:</strong> {source['speciality']}<br>
        <strong>Doktor:</strong> {source['doctor']}<br>
        <strong>İlgili Soru:</strong> {source['preview']}...
    </div>
    """, unsafe_allow_html=True)


def main():
    """Ana uygulama"""

    # Başlık
    st.markdown('<p class="main-header">🏥 Tıbbi Soru-Cevap Chatbot</p>', unsafe_allow_html=True)

    # Uyarı mesajı
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>ÖNEMLİ UYARI:</strong> Bu chatbot sadece bilgilendirme amaçlıdır.
        Verilen bilgiler kesin tıbbi tavsiye yerine geçmez. Sağlık sorunlarınız için
        mutlaka bir sağlık uzmanına başvurun.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")

        # Kaç doküman retrieve edilecek
        k_docs = st.slider(
            "Kaynak Doküman Sayısı",
            min_value=1,
            max_value=5,
            value=3,
            help="Her sorgu için kaç benzer soru-cevap çifti kullanılacak"
        )

        # Uzmanlık alanı filtresi (opsiyonel)
        use_filter = st.checkbox("Uzmanlık Alanı Filtrele", value=False)
        speciality_filter = None

        if use_filter:
            speciality_filter = st.text_input(
                "Uzmanlık Alanı",
                placeholder="Örn: Dahiliye, Kardiyoloji"
            )

        st.markdown("---")

        # Hakkında
        st.header("ℹ️ Hakkında")
        st.markdown("""
        Bu chatbot, 167.000+ tıbbi soru-cevap verisi kullanılarak
        RAG (Retrieval-Augmented Generation) teknolojisi ile geliştirilmiştir.

        **Teknolojiler:**
        - Gemini API (LLM)
        - LangChain (RAG Framework)
        - ChromaDB (Vector Database)
        - Streamlit (Web Framework)

        **Veri Seti:**
        - Kaynak: Huggingface
        - Boyut: 167,732 kayıt
        - Dil: Türkçe
        """)

        st.markdown("---")

        # Örnek sorular
        st.header("💡 Örnek Sorular")
        example_questions = [
            "Baş ağrısı için ne yapmalıyım?",
            "Grip olduğumda ne yemem gerekir?",
            "Yüksek tansiyon belirtileri nelerdir?",
            "Vitamin D eksikliği nasıl anlaşılır?",
            "Uykusuzluk için doğal çözümler nelerdir?"
        ]

        for question in example_questions:
            if st.button(question, key=f"example_{question[:20]}"):
                st.session_state.current_question = question

    # Ana içerik alanı
    st.header("💬 Soru Sorun")

    # Session state'i başlat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""

    # RAG sistemini yükle
    with st.spinner("Sistem yükleniyor..."):
        rag_system = load_rag_system()

    st.success("✓ Sistem hazır! Sorularınızı sorabilirsiniz.")

    # Soru input
    col1, col2 = st.columns([6, 1])

    with col1:
        user_question = st.text_input(
            "Sorunuzu yazın:",
            value=st.session_state.current_question,
            placeholder="Örn: Mide ağrısı için ne yapmalıyım?",
            key="question_input"
        )

    with col2:
        st.write("")  # Boşluk için
        st.write("")  # Boşluk için
        ask_button = st.button("🔍 Sor", type="primary", use_container_width=True)

    # Soru sorulduğunda
    if ask_button and user_question:
        with st.spinner("Cevap hazırlanıyor..."):
            try:
                # RAG sisteminden cevap al
                result = rag_system.ask(
                    query=user_question,
                    k=k_docs,
                    speciality_filter=speciality_filter if use_filter and speciality_filter else None
                )

                # Sohbet geçmişine ekle
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": result['answer'],
                    "sources": result['sources']
                })

                # Input'u temizle
                st.session_state.current_question = ""

            except Exception as e:
                st.error(f"Hata oluştu: {e}")

    # Sohbet geçmişini göster
    if st.session_state.chat_history:
        st.markdown("---")
        st.header("📝 Sohbet Geçmişi")

        # En son soruyu en üstte göster
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>Soru:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Cevap:</strong><br>{chat['answer']}
            </div>
            """, unsafe_allow_html=True)

            # Kaynakları göster
            with st.expander("📚 Kaynakları Görüntüle"):
                for i, source in enumerate(chat['sources'], 1):
                    display_source(source, i)

            st.markdown("---")

        # Geçmişi temizle butonu
        if st.button("🗑️ Sohbet Geçmişini Temizle"):
            st.session_state.chat_history = []
            st.rerun()

    else:
        st.info("👆 Yukarıdaki kutuya sorunuzu yazın veya yan taraftaki örnek sorulardan birini seçin.")


if __name__ == "__main__":
    main()

# Agent Project

Bu proje, Deep Q-Learning algoritması ile bir borsa ajanı (trading agent) simülasyonu sunar. Amaç, geçmiş fiyat verilerine dayanarak al-sat kararları vererek karı maksimize etmektir.

## 🚀 Kurulum

```bash
git clone https://github.com/kullanici_adi/agent-project.git
cd agent-project
pip install -r requirements.txt
```

## 📁 Proje Yapısı

```bash
agent-project/
├── main.py                   # Eğitim döngüsünü başlatır
├── environment/
│   └── trading_env.py        # Al-sat simülasyon ortamı
├── models/
│   ├── dqn.py                # DQN modeli (PyTorch)
│   └── agent.py              # Ajan sınıfı (eğitim, karar)
├── data/
│   └── data_loader.py        # Veri çekme ve işleme
├── utils/
│   └── helpers.py            # Yardımcı fonksiyonlar
```

## 🧠 Kullanım

```bash
python main.py
```

## 🔧 Gereksinimler

- yfinance
- pandas
- numpy
- torch


# FashionMNIST Image Classification (PyTorch)

Fashion-MNIST veri seti üzerinde evrişimli sinir ağı (CNN) ile giyim eşyası sınıflandırması. Model 5 epoch sonunda test setinde yaklaşık **%89.5** doğruluk elde eder.

## Özellikler

- Özel `FashionClassifierModel` mimarisi (2 conv bloğu + tam bağlı katman)
- Veri artırma: `TrivialAugmentWide`, yatay çevirme
- Eğitim / test loss ve accuracy grafikleri
- Yerel görsel ile tahmin (`ankle_bot.jpg`)

## Sınıflar

| # | Sınıf |
|---|--------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Gereksinimler

- Python 3.10+
- PyTorch, torchvision ve bağımlılıklar (`requirements.txt`)

## Kurulum

```bash
git clone https://github.com/melikesoftware/Fashion-Mnist-CNN.git
cd Fashion-Mnist-CNN

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Kullanım

```bash
python main.py
```

İlk çalıştırmada FashionMNIST veri seti `data/` klasörüne indirilir (repo’ya dahil değildir).

Kendi görselinizle tahmin için proje köküne bir `.jpg` dosyası koyun ve `main.py` içindeki `Image.open("ankle_bot.jpg")` satırındaki dosya adını değiştirin.

## Model özeti

| Metrik | Değer |
|--------|--------|
| Toplam parametre | 48,554 |
| Batch size | 32 |
| Epoch | 5 |
| Optimizer | Adam (lr=0.001) |
| Son test accuracy | ~89.55% |

## Proje yapısı

```
.
├── main.py           # Eğitim, değerlendirme ve inference
├── ankle_bot.jpg     # Örnek test görseli
├── requirements.txt
└── data/             # Otomatik indirilir (.gitignore)
```


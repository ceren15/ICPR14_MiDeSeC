# ICPR14_MiDeSeC

# ICPR14 & MiDeSeC ile YOLOv8 Segmentasyon Eğitimi (Macenko Normalizasyonu + İki Aşamalı Fine-Tune)

Bu depo, **MiDeSeC** ve **ICPR14** veri setleri kullanılarak **YOLOv8n-seg** tabanlı segmentasyon modeli eğitimi için uçtan uca bir eğitim hattı (pipeline) sunar. Çalışma;  
(i) **Macenko renk normalizasyonu** ile görüntü renk dağılımlarının standardizasyonu,  
(ii) veri/etiket doğrulama (poligon overlay) modülü,  
(iii) **MiDeSeC + ICPR14 birleşik veri seti** üzerinde **iki aşamalı eğitim (stage-1 pretrain + stage-2 fine-tune)**,  
(iv) test değerlendirmesi ve görselleştirme adımlarını içerir.

> Not: Bu proje Colab/Drive dizin yapısı ile çalışacak şekilde hazırlanmıştır; yerelde çalıştırmak için yol (path) düzenlemeleri gerekebilir.

---

## İçerik
- [Amaç ve Kapsam](#amaç-ve-kapsam)
- [Yöntem](#yöntem)
- [Veri Setleri](#veri-setleri)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Değerlendirme ve Çıktılar](#değerlendirme-ve-çıktılar)
- [Sonuçlar](#sonuçlar)
- [Alıntılama](#alıntılama)
- [Lisans ve Notlar](#lisans-ve-notlar)

---

## Amaç ve Kapsam
Bu çalışmanın amacı, farklı kaynaklardan gelen görsel verilerdeki **renk/ışık farklılıklarını azaltarak** segmentasyon performansını iyileştirmek ve birleşik veri ile daha genellenebilir bir model elde etmektir. Bunun için:
- **Macenko normalizasyonu** ile boyama/aydınlatma değişkenliği azaltılır,
- YOLO segmentasyon etiketleri **görsel olarak doğrulanır**,
- Birleştirilmiş veri üzerinde **aşamalı eğitim** uygulanır.

---

## Yöntem
### 1) Macenko Normalizasyonu (Renk Standardizasyonu)
Görüntüler, Macenko yaklaşımıyla normalize edilerek veri setleri arası renk dağılımı farkları minimize edilir. Bu adım özellikle farklı cihaz/ortam koşullarında elde edilen görüntülerde modelin genellemesini artırmayı hedefler.

### 2) Etiket Doğrulama (Ground-Truth Poligon Overlay)
Segmentasyon etiketleri (poligonlar), görüntü üzerine bindirilerek (overlay) görsel kontrol yapılır. Amaç; hatalı/bozuk etiketleri eğitim öncesi tespit etmektir.

### 3) Birleştirme ve İki Aşamalı Eğitim
- **Stage-1:** Birleşik veri üzerinde başlangıç eğitimi (augment açık, sınırlı mosaic vb.)
- **Stage-2:** Stage-1’den gelen en iyi ağırlıklar ile **fine-tune** (daha konservatif augment; mosaic kapalı vb.)

---

## Veri Setleri
Bu depo, aşağıdaki veri setlerini kullanacak şekilde kurgulanmıştır:
- **MiDeSeC**
- **ICPR14**

> Veri setlerine ait lisans/erişim koşulları ilgili kaynaklara aittir. Bu depo doğrudan veri paylaşmaz; kullanıcı kendi erişimiyle veriyi yerleştirir.

---

## Proje Yapısı
Colab/Drive üzerinde örnek dizinler:
- `/content/drive/MyDrive/Colab Notebooks/MiDeSeC`
- `/content/drive/MyDrive/Colab Notebooks/ICPR14`
- `/content/drive/MyDrive/Colab Notebooks/MERGED_YOLO`
  - `images/train`, `images/val`, `images/test`
  - `labels/train`, `labels/val`, `labels/test`
  - `merged_midesec_icpr.yaml`

Eğitim çıktı klasörleri (Ultralytics):
- `/content/runs/segment/merged_stage1`
- `/content/runs/segment/merged_stage2_ft`

---

## Kurulum
### Ortam
- Python >= 3.8 (öneri: 3.10+)
- (Opsiyonel) GPU: CUDA destekli ortam

### Bağımlılıklar
Çalışmada kullanılan temel kütüphaneler:
- `ultralytics`
- `torch`
- `opencv-python (cv2)`
- `numpy`, `pandas`
- `matplotlib`
- `torchstain` (Macenko normalizasyonu)
- `tqdm`
- `Pillow`

Örnek kurulum:
```bash
pip install -U ultralytics torch torchvision torchaudio opencv-python numpy pandas matplotlib tqdm pillow torchstain

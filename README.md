# TrafficVisionAI
# 🚦 TrafficVisionAI – YOLOv8 Tabanlı Akıllı Trafik Işığı Sistemi

Bu proje, **YOLOv8 nesne tespiti** kullanarak gerçek zamanlı olarak trafikteki araç yoğunluğunu analiz eden ve yoğunluğa göre trafik ışıklarını **dinamik şekilde değiştiren** bir sistemdir.

---

## 🧠 Amaç
Klasik trafik ışıkları sabit sürelerle çalışır. Ancak yoğunluk değiştiğinde bu sistem verimsiz hale gelir.  
TrafficVisionAI, **kamera görüntülerinden araç sayısını tespit ederek** trafik akışına göre **ışık sürelerini optimize eder**.

---

## 🧩 Kullanılan Teknolojiler
- 🧮 **Python 3.x**
- 🧠 **YOLOv8 (ultralytics)**
- 🎥 **OpenCV (cv2)**
- 📊 **NumPy**
- ⏱️ **time (zamanlama işlemleri için)**

---

## ⚙️ Kurulum
1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install ultralytics opencv-python numpy

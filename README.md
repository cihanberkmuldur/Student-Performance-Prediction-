# 📊 Makine Öğrenmesi ile Öğrenci Akademik Başarı Tahmini

## 📝 Proje Özeti
Bu çalışmanın amacı, öğrencilerin demografik, sosyo-ekonomik ve akademik geçmiş verilerini kullanarak yıl sonu başarı notunu (G3) tahmin eden makine öğrenmesi modelleri geliştirmektir[cite: 2]. Problem temel olarak bir regresyon problemi olmakla birlikte, modeller aynı zamanda sınıflandırma perspektifinden de değerlendirilmiştir[cite: 2].

## 📂 Veri Seti
Analizlerde UCI Machine Learning Repository’de yer alan **Student Performance Data Set** kullanılmış olup, Matematik (`student-mat.csv`) ve Portekizce (`student-por.csv`) dersleri ayrı ayrı incelenmiştir[cite: 2]. 

Her iki veri seti de **33 adet öznitelik** içermektedir[cite: 2]. Öznitelik grupları şunlardır:
* **Demografik:** Yaş, cinsiyet, adres türü[cite: 2].
* **Sosyo-ekonomik:** Anne/baba eğitim durumu ve mesleği, aile desteği[cite: 2].
* **Akademik:** Çalışma süresi, devamsızlık, geçmiş başarısızlıklar, 1. Dönem Notu (G1) ve 2. Dönem Notu (G2)[cite: 2].
* **Hedef Değişken:** Yıl sonu notu (G3)[cite: 2].

**Veri Seti Kaynağı:** [Kaggle - Student Alcohol Consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)[cite: 2].

## ⚙️ Veri Ön İşleme
* Eksik veri kontrolü yapılmış ve veri setlerinin bütünlüğü doğrulanmıştır[cite: 2].
* Kategorik değişkenler `pd.get_dummies()` yöntemi ile sayısal forma dönüştürülmüştür[cite: 2].
* Veri seti %80 eğitim ve %20 test olacak şekilde ayrılmıştır[cite: 2].

## 🧠 Kullanılan Modeller
Tahminleme için aşağıdaki makine öğrenmesi algoritmaları kullanılmıştır:
* Lineer Regresyon[cite: 2]
* Decision Tree Regressor (Karar Ağacı)[cite: 2]
* Random Forest Regressor (Rastgele Orman)[cite: 2]

## 🧪 Analiz Senaryoları ve Metrikler
Modellerin performansı iki farklı senaryoda test edilmiştir[cite: 2]:
1. **Notlar Dahil Senaryosu:** Geçmiş sınav notlarının (G1 ve G2) modele dahil edildiği durum[cite: 2].
2. **Sadece Sosyal Senaryosu:** G1 ve G2 notlarının çıkarılarak sadece sosyo-ekonomik ve demografik verilerin kullanıldığı durum[cite: 2].

**Değerlendirme Metrikleri:**
* **Regresyon:** MSE, RMSE, MAE, R², AIC, BIC ve Entropi[cite: 2].
* **Sınıflandırma:** Accuracy (ACC), Precision, Sensitivity (Recall), F1-Score, Cross-Entropy ve G-Mean[cite: 2].

## 📈 Bulgular ve Sonuçlar
* **Notların Etkisi:** G1 ve G2 değişkenleri modele dahil edildiğinde tüm algoritmalar yüksek performans göstermiş olup, R² değerleri 0.72 – 0.85 aralığına ulaşmıştır[cite: 2]. Sadece sosyal değişkenlerin kullanıldığı senaryoda ise başarı ciddi şekilde düşmüş, R² değerleri 0.14 – 0.24 aralığına gerilemiştir[cite: 2].
* **Model Karşılaştırması:** **Random Forest**, en kararlı ve genellenebilir sonuçları üretmiştir[cite: 2]. **Decision Tree** modelinin ise aşırı öğrenmeye (overfitting) yatkın olduğu gözlemlenmiştir[cite: 2]. **Lineer Regresyon**, Portekizce dersinde beklentilerin üzerinde performans göstererek en yüksek R² (~0.85) değerine ulaşmıştır[cite: 2].
* **Ders Farklılıkları:** Portekizce dersi modelleri, Matematik dersine kıyasla daha düşük hata oranlarına (RMSE ≈ 1.21 vs 1.95) sahiptir[cite: 2].
* **Korelasyon:** G3 (Yıl sonu notu) ile en yüksek korelasyona sahip değişkenler G2 ve G1'dir[cite: 2]. Sosyal değişkenlerin doğrudan notu değil, dolaylı olarak başarı riskini etkilediği görülmüştür[cite: 2].

## 🚀 Sonuç ve Öneriler
Bu çalışma, akademik başarı tahmininde G1 ve G2 notlarının vazgeçilmez olduğunu kanıtlamaktadır[cite: 2]. Ancak regresyon başarısının düştüğü "sadece sosyal" senaryoda bile sınıflandırma algoritmaları %65 – %89 arası Accuracy üreterek öğrencinin başarısız olma riskini tahmin etmede başarılı olmuştur[cite: 2]. Bu sistemler, eğitim kurumları için erken uyarı sistemleri kurulmasına temel oluşturabilir[cite: 2].

```

# Makine Öğrenmesi Çalışmaları

Bu repo, çeşitli makine öğrenmesi algoritmalarının farklı veri setleri üzerinde uygulanmasını ve sonuçlarının analiz edilmesini içeren çalışmaları barındırmaktadır. Her bir klasör, belirli bir model veya konsept üzerine odaklanmıştır.

## İçindekiler

1.  [01- Çok Değişkenli Lineer Regresyon](#01--çok-değişkenli-lineer-regresyon)
2.  [02- Lojistik Regresyon](#02--lojistik-regresyon)
3.  [03- Yapay Sinir Ağları](#03--yapay-sinir-ağları-ysa)

---

## 01- Çok Değişkenli Lineer Regresyon

Bu çalışma, bir öğrencinin performansını çeşitli faktörlere dayanarak tahmin etmeyi amaçlayan çok değişkenli bir lineer regresyon modelinin sıfırdan implementasyonunu içermektedir.

### Veri Seti

- **Dosya:** `Student_Performance.csv`
- **Amaç:** Öğrenci performansını (`performance`) tahmin etmek.

### Kullanılan Model

- Gradient Descent algoritması kullanılarak optimize edilen, sıfırdan yazılmış bir Çok Değişkenli Lineer Regresyon modelidir.

### İncelenen Parametreler ve Sonuçlar

Modelin performansı, farklı normalizasyon teknikleri, iterasyon sayıları ve öğrenme oranları (alpha) denenerek analiz edilmiştir.

- **Normalizasyon Teknikleri:**

  - **Z-Score (Standard Scaler):** En düşük maliyet (`1.88`) ile en iyi sonucu vermiştir.
  - **Min-Max Scaler:** İkinci en iyi sonucu vermiştir (`Maliyet: 11.17`).
  - **L2 Normalizasyon:** En yüksek maliyeti vermiştir (`Maliyet: 76.12`).

- **Hiperparametreler (Z-Score Normalizasyon ile):**
  - **İterasyon Sayısı:** `[100, 250, 500, 1000, 1500]` değerleri test edilmiştir. En düşük maliyet **1500 iterasyonda** elde edilmiştir.
  - **Öğrenme Oranı (Alpha):** `[0.005, 0.001, 0.01]` değerleri test edilmiştir. En iyi sonuç **`alpha = 0.01`** ile alınmıştır.

### Elde Edilen En İyi Sonuç

Bu veri setinde en iyi tahmin performansı, veriye **Z-Score Normalizasyonu** uygulandıktan sonra, modelin **1500 iterasyon** ve **0.01 öğrenme oranı** ile eğitilmesiyle elde edilmiştir. Bu, doğru veri ön işleme ve hiperparametre ayarının model performansı üzerindeki kritik etkisini göstermektedir.

---

## 02- Lojistik Regresyon

Bu çalışma, banknotların fiziksel özelliklerine göre gerçek mi yoksa sahte mi olduğunu sınıflandıran bir lojistik regresyon modelini içermektedir. Hem doğrusal (linear) hem de doğrusal olmayan (non-linear) karar sınırları incelenmiştir.

### Veri Seti

- **Dosya:** `fake_bills.csv`
- **Amaç:** Banknotların sahte olup olmadığını (`Is Genuine`) sınıflandırmak.

### İncelenen Parametreler ve Sonuçlar

1.  **Doğrusal Karar Sınırı (Linear Decision Boundary):**

    - Veri büyük ölçüde doğrusal olarak ayrıştırılabildiği için, basit bir lojistik regresyon modeli bile yüksek başarı sağlamıştır.
    - **35,000 iterasyon** ve **0.003 öğrenme oranı** ile etkili bir sınıflandırma yapılmıştır.

2.  **Doğrusal Olmayan Karar Sınırı (Non-linear Decision Boundary) ve Regülerleştirme:**
    - Modelin daha karmaşık örüntüleri öğrenmesi için 6. dereceden polinomsal özellikler eklenmiştir.
    - Aşırı öğrenmeyi (overfitting) önlemek için **regülerleştirme (regularization)** parametresi olan **lambda (λ)** test edilmiştir.
    - **Lambda (λ) Değerleri:**
      - **λ = 0 (Regülerleştirme Yok):** Modelin, eğitim verisine aşırı uyum sağlayarak (overfitting) çok karmaşık ve genellenemez bir sınır çizdiği görülmüştür.
      - **λ = 30:** Modelin, hem veri setinin genel yapısını yakaladığı hem de aşırı öğrenmeden kaçındığı **en ideal karar sınırını** oluşturmuştur.
      - **λ = 3500 (Yüksek Değer):** Modelin yetersiz öğrendiği (underfitting) ve verinin karmaşıklığını yakalayamadığı basit bir sınır çizdiği görülmüştür.

### Elde Edilen En İyi Sonuç

Bu sınıflandırma probleminde en iyi ve en genellenebilir model, polinomsal özellikler eklendikten sonra **dengeli bir lambda değeri (örn: λ=30 civarı) ile regülerleştirilmiş** lojistik regresyon modelidir. Bu yaklaşım, modelin esnekliğini korurken aşırı öğrenmesini engellemiştir.

---

## 03- Yapay Sinir Ağları (YSA)

Bu çalışma, dermatolojik verileri kullanarak 6 farklı cilt hastalığını sınıflandırmak amacıyla PyTorch ile farklı mimarilerde Yapay Sinir Ağı (YSA) modellerinin oluşturulmasını ve karşılaştırılmasını içermektedir.

### Veri Seti

- **Dosya:** `dermatology_database_1.csv`
- **Amaç:** 34 özelliğe dayanarak 6 farklı cilt hastalığından birini teşhis etmek.

### İncelenen Parametreler ve Sonuçlar

Farklı derinliklerde (katman sayısı) ve farklı aktivasyon fonksiyonlarına sahip üç ana YSA modeli 100 epoch boyunca eğitilerek test edilmiştir. Tüm modellerde `Adam` optimizasyon algoritması (`lr=0.01`) kullanılmıştır.

- **Model Mimarisi Karşılaştırması (ReLU Aktivasyon Fonksiyonu ile):**

  - **Model 1 (1 Gizli Katman):** Test Doğruluğu: **%94.44**
  - **Model 2 (2 Gizli Katman):** Test Doğruluğu: **%97.22**
  - **Model 3 (3 Gizli Katman):** Test Doğruluğu: **%98.61**

- **Aktivasyon Fonksiyonu Karşılaştırması (En İyi Model Olan Model 3 Üzerinde):**
  - **ReLU:** En yüksek doğruluğu sağlamıştır. ( **%98.61** )
  - **Sigmoid:** ReLU'ya yakın bir performans göstermiştir. ( **%97.22** )
  - **GELU:** Bu problemde diğerlerine göre daha düşük bir doğruluk vermiştir. ( **%95.83** )

### Elde Edilen En İyi Sonuç

Bu çok sınıflı sınıflandırma probleminde en yüksek performansı, **3 gizli katmana sahip** ve aktivasyon fonksiyonu olarak **ReLU** kullanan YSA modeli elde etmiştir. Ulaşılan **%98.61**'lik test doğruluğu, problemin karmaşıklığını çözmek için daha derin bir sinir ağı mimarisinin daha etkili olduğunu ve doğru aktivasyon fonksiyonu seçiminin başarıda önemli bir rol oynadığını göstermektedir.

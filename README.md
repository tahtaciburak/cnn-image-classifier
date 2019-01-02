# Proje Açıklaması
Proje kapsamında bir derin öğrenme modeli olan CNN(Convolutional Neural Networks) kullanılarak 10 sınıflı görüntü tanıma işlemi yapılmıştır. 2700 örnek içeren veri seti %80 eğitim, %20 test olarak ayrıldıktan sonra veri çoğaltma(data augmentation) işlemiyle sağa ve sola döndürülmüş olan görüntüler de veri setine eklenmiştir. Yapay sinir ağı oluşturulurken 5 adet konvolüsyon+max_pooling+batch_normalizer katmanları kullanılmıştır. Bu katmanların çıktısı fully connected katmandan da geçtikten sonra softmax classifier katmanına iletilmiştir. Ağın gerçeklenmesi için python programlama dili ve Keras kütüphanesi kullanılmıştır. Testler sonucunda, %97.72 eğitim başarısı ve %97.3 test başarısı elde edilmiştir.

# Kaynak Kodların İndirilmesi
https://github.com/tahtaciburak/cnn-image-classifier/ adresinden projeye ait kodlar indirilebilir. Kodlarla birlikte Train/Test/Validation datasetleri de bu repo içerisinde mevcuttur. Herhangi bir düzenleme yapmaksızın kodların çalışırlığı bu repo üzerinden test edilebilir.

# Kodun Çalıştırılması
Proje, jupyter-notebook uygulaması kullanılarak geliştirmiştir. Kaynak kodlar hem jupyter-notebook ortamında hem de normal python dosyası şeklinde kaydedilmiştir. Kodu çalıştırabilmek için Keras ve Python3.6 yüklü olmalıdır. Ek olarak numpy, Pillow bağımlılıklarının da yüklü olması gerekmektedir.

## Jupyter-Notebook Kullanarak
`jupyter-notebook` komutuyla interaktif not defterinin core'u çalıştırılır. Ardından kaynak kodların bulunduğu dizine gidilir. Bu dizinde main.ipynb dosyası açıldıktan sonra parçalara bölünmüş olan kodlar adım adım çalıştırılabilir. `augmented-data-main.ipynb` dosyasında veri çoğaltma işlemi ile birlikte yapılan sonuçlar bulunmaktadır. İki farklı modelin sonuçları bu dosyalar kullanılarak karşılaştırılabilir.

## Python Kullanarak
`python3 main.py` komutu verilerek script çalıştırılabilir. Bağımlılıklar sorunsuz biçimde yüklenmişse main.py isimli script "Dataset/" dosyasını okuyarak işlemeye başlamaktadır. Eğitim ve test sonuçlarını STDOUT'a bastıktan sonra validasyon veri setini okuyup validasyon sonuçlarını göstermektedir.


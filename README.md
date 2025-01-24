# plant-disease-ai
NAI - Plant_health_classifier
Zuzanna Leszczyńska, Olga Bońdos s25337

Opis:
Nasz projekt obejmuje wytrenowanie i porównanie modeli rozpoznających czy roślina jest zdrowa lub chora, planujemy wykorzystać go w naszym projekcie inżynierskim SmartGreenhouse.

Dane:
Użyłyśmy datasetu pobranego z Kaggle. Składa się on z 87 tysięcy zdjęć roślin oznaczonych jako healthy lub na co są chore. Początkowo dataset składał się z folderów podzielonych na rośliny o różnych chorobach, jednak uprościliśmy go na rośliny chore i zdrowe, a następnie użyłyśmy labeli 1 lub 0 dla zdrowych i chorych roślin.

vipoooool/new-plant-diseases-dataset


Użyte modele:

Do klasyfikacji zdrowych i chorych roślin wybrałyśmy modele CNN, ResNet oraz RainForest, ponieważ świetnie nadają się do analizy obrazów. CNN potrafią skutecznie wychwytywać lokalne wzorce, takie jak tekstury czy krawędzie, które są istotne przy diagnozowaniu chorób roślin. ResNet umożliwia trenowanie głębszych sieci dzięki połączeniom rezydualnym i wykorzystaniu pre-trenowanych modeli, co pozwala na dokładniejsze rozróżnianie subtelnych różnic. RainForest łączy prostotę konstrukcji z efektywnością obliczeniową, dzięki czemu stanowi lekką, ale skuteczną alternatywę dla bardziej złożonych modeli.

ResNet-18 
Epoch [1/10], Loss: 0.0755, Accuracy: 97.87%
Validation Accuracy: 98.83%
Epoch [2/10], Loss: 0.0424, Accuracy: 98.90%
Validation Accuracy: 92.65%
Epoch [3/10], Loss: 0.0090, Accuracy: 99.67%
Validation Accuracy: 99.83%
Epoch [4/10], Loss: 0.0079, Accuracy: 99.84%
Validation Accuracy: 98.33%
Epoch [5/10], Loss: 0.0320, Accuracy: 99.31%
Validation Accuracy: 99.33%
Epoch [6/10], Loss: 0.0156, Accuracy: 99.59%
Validation Accuracy: 99.50%
Epoch [7/10], Loss: 0.0052, Accuracy: 99.88%
Validation Accuracy: 98.16%
Epoch [8/10], Loss: 0.0130, Accuracy: 99.75%
Validation Accuracy: 98.66%
Epoch [9/10], Loss: 0.0181, Accuracy: 99.51%
Validation Accuracy: 99.83%
Epoch [10/10], Loss: 0.0114, Accuracy: 99.67%
Validation Accuracy: 97.83%

CNN
Epoch [1/10], Loss: 0.1587, Accuracy: 93.99%
Validation Accuracy: 99.50%
Epoch [2/10], Loss: 0.0378, Accuracy: 99.14%
Validation Accuracy: 99.67%
Epoch [3/10], Loss: 0.0191, Accuracy: 99.71%
Validation Accuracy: 99.17%
Epoch [4/10], Loss: 0.0112, Accuracy: 99.71%
Validation Accuracy: 99.50%
Epoch [5/10], Loss: 0.0070, Accuracy: 99.84%
Validation Accuracy: 99.67%
Epoch [6/10], Loss: 0.0046, Accuracy: 99.84%
Validation Accuracy: 99.67%
Epoch [7/10], Loss: 0.0045, Accuracy: 99.88%
Validation Accuracy: 99.83%
Epoch [8/10], Loss: 0.0012, Accuracy: 99.92%
Validation Accuracy: 99.83%
Epoch [9/10], Loss: 0.0008, Accuracy: 99.96%
Validation Accuracy: 99.67%
Epoch [10/10], Loss: 0.0005, Accuracy: 99.96%
Validation Accuracy: 99.83%

RainForest
Epoch [1/10], Loss: 0.1411, Accuracy: 96.61%
Validation Accuracy: 98.33%
Epoch [2/10], Loss: 0.0760, Accuracy: 98.65%
Validation Accuracy: 99.67%
Epoch [3/10], Loss: 0.0287, Accuracy: 99.39%
Validation Accuracy: 98.83%
Epoch [4/10], Loss: 0.0230, Accuracy: 99.51%
Validation Accuracy: 99.50%
Epoch [5/10], Loss: 0.0164, Accuracy: 99.59%
Validation Accuracy: 99.67%
Epoch [6/10], Loss: 0.0204, Accuracy: 99.47%
Validation Accuracy: 99.67%
Epoch [7/10], Loss: 0.0221, Accuracy: 99.31%
Validation Accuracy: 99.50%
Epoch [8/10], Loss: 0.0105, Accuracy: 99.80%
Validation Accuracy: 99.83%
Epoch [9/10], Loss: 0.0073, Accuracy: 99.84%
Validation Accuracy: 99.50%
Epoch [10/10], Loss: 0.0047, Accuracy: 99.84%
Validation Accuracy: 99.83%



# Projekt Kompresji i Analizy Ruchu w Wideo

## Opis Projektu

Ten projekt demonstracyjny pokazuje, jak kompresować i dekompresować pojedyncze klatki wideo oraz analizować ruch w czasie rzeczywistym za pomocą optycznego przepływu. Projekt składa się z następujących kroków:

1. **Podział obrazu na bloki:** Obraz jest dzielony na mniejsze bloki o rozmiarze 8x8 pikseli.
2. **Kompresja:** Każdy blok jest poddawany dyskretnej transformacie kosinusowej (DCT), kwantyzowany, a następnie kodowany przy użyciu kodowania RLE (Run-Length Encoding).
3. **Dekompresja:** Skompresowane bloki są dekodowane, dekwantyzowane i poddawane odwrotnej transformacie kosinusowej (IDCT), aby zrekonstruować oryginalny obraz.
4. **Analiza ruchu:** Optyczny przepływ między kolejnymi klatkami wideo jest obliczany, co pozwala na wizualizację ruchu w czasie rzeczywistym.
# La transformada de Fourier en tiempo corto

> Para construir un espectrograma se usa la `transformada de Fourier en tiempo corto` (STFT).

> STFT consiste en calcular la fransformada de Fourier en frames sucesivos de la señal. 

> Por lo general, los frames se van tomando de manera `traslapada` y se les aplican funciones de `ventana`.

$$ X(k,\omega) = \sum_{n} x[n]\cdot w[n-k]\cdot e^{-j\,\omega\, n}$$ 

> donde $k$ y $\omega$ representan los índices de tiempo (frame) y de frecuencia, respectivamente.

<img src="../figuras/stft.png" width="500">

> Arquitectura genérica de extracción de características para análisis de audios:

<img src="../figuras/applied0.png" width="500">

<img src="../figuras/applied1.png" width="500">

<img src="../figuras/taxonomy.png" width="500">
# Entender la resolución de frecuencia en el sistema auditivo

> El estímulo mecánico se convierte en una código neuronal.

<img src="../figuras/hair_cells2.png" width="500">

> Los canales ionicos situados en las puntas de las células ciliares, se abren cuando se deflectan los cilios, permitiendo que los iones entren en
las celulas.

<img src="../figuras/hair_cells.png" width="500">

> A través del nervio auditivo se transmiten las respuestas neuronales  que llevan los features más importantes de los sonidos.

<img src="../figuras/Cariani.png" width="700">

<img src="../figuras/neural.png" width="400">

> Los intervalos de tiempo entre las descargas neuronales dan información de periodicidades en el sonido.

<img src="../figuras/espectrograma_neuronal.png" width="700">

> La respuesta neuronal de cada fibra en el nervio auditivo es dependiente específicamente de qué frecuencia sea el estímulo.

<img src="../figuras/stern.png" width="500">

> `Filtros auditivos`: más `angostos` para mejorar el problema de bajas frecuencias y más `anchos` para cubrir las altas frecuencias.

<img src="../figuras/tuning_curves.png" width="700">

<img src="../figuras/intensidad.png" width="700">

> El sistema auditivo es selectivo en frecuencias

<img src="../figuras/seneff0.png" width="700">

> El sincronismo se asocia a componentes de frecuencias bajas y a detección en presencia de niveles bajos de intensidades.

> El detector de envolvente se relaciona con la estructura fina de componentes de frecuencias altas.

<img src="../figuras/seneff.png" width="700">

> Tres distintas escalas de frecuencia se han usado para simular los anchos de banda variables de los filtros auditivos:

> Escala Bark (Heinrich Barkhausen, 1881-1956):

$$ Bark(f)=[26.8/(1 + (1960/f))] - 0.53$$

> donde $f$ es la frecuencia en Hz.

> Escala Mel (Stevens, Volkman, y Newman, 1937):

$$ Mel(f) = 2595\cdot log_{10}(1 + \frac{f}{700})$$

> Escala ERB (The Equivalent Rectangular Bandwidth), B.C.J. Moore and B. R. Glasberg, 1983:

$$ ERB(f) = 21.4\cdot log_{10}(1 + 4.37\cdot \frac{f}{1000}) $$
# TF-IDF en las conferencias matutinas de AMLO 2019-2023

## Análisis de Tópicos en las Conferencias Matutinas de AMLO (2019–2023)

Este repositorio contiene el análisis temático de las conferencias de prensa matutinas del presidente Andrés Manuel López Obrador entre 2019 y 2023, utilizando técnicas de modelado de tópicos basadas en descomposición matricial.

## Contenido del repositorio

- `RepositorioAMLO.py`: Clase para cargar y estructurar las conferencias en formato procesable.
- `preprocesar.py`: Funciones para limpiar, lematizar y normalizar el texto (acentos, puntuación, stopwords).
- `AnalisisTopicos.py`: Implementación del modelo TF-IDF y modelado de tópicos con NMF y SVD.
- `linea_del_tiempo.html`: Visualización interactiva del índice temático semanal por tópico.
- `wordclouds/`: Carpeta con las nubes de palabras por tópico.
- `resultados/`: Tablas, gráficas y textos interpretativos del análisis.

## Metodología

1. **Preprocesamiento**: Se normalizan y lematizan los textos, y se filtran stopwords ampliadas.
2. **Vectorización**: Se aplica TF-IDF con control de vocabulario y uso de bigramas.
3. **Modelado de tópicos**: Se exploran modelos con NMF y SVD. Se elige NMF con \( k=15 \) por su estabilidad e interpretabilidad.
4. **Índice temático semanal**: Se construye un indicador basado en frecuencia relativa, importancia acumulada y diversidad discursiva (índice de Shannon).

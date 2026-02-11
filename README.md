# Modelado de Tópicos en las Conferencias Matutinas de AMLO (2019-2023)

Proyecto integral de procesamiento de lenguaje natural aplicando técnicas de factorización matricial (SVD y NMF) para extraer y analizar patrones temáticos en transcripciones de conferencias presidenciales.

## Descripción General

Este proyecto implementa un pipeline completo de modelado de tópicos para analizar datos textuales de conferencias de prensa presidenciales diarias. El sistema extrae tópicos latentes, visualiza su evolución temporal y genera nubes de palabras interpretables utilizando técnicas avanzadas de reducción dimensional.

**Visualización Interactiva**: [Ver Resultados](https://franj0212.github.io/TF-IDF-en-las-conferencias-matutinas-de-AMLO-2019-2023-/)

## Estructura del Proyecto

### Módulos Principales

#### `AnalisisTopicos.py`
Módulo principal que implementa algoritmos de modelado de tópicos y herramientas de visualización.

**Clases:**
- **TF_IDF**: Configura y genera representación matricial TF-IDF
  - `configurar_vectorizador()`: Selección adaptativa de vocabulario con ajuste automático de min_df
  - Soporta n-gramas (1,2) y filtrado personalizable de stopwords

- **Top_SVD**: Implementa SVD Truncado con rotación Varimax para extracción de tópicos
  - `obtener_topicos_svd()`: Realiza descomposición SVD estándar
  - `fit_transform_rot()`: Implementación personalizada de SVD con rotación Varimax
  - `obtener_topicos_svd_con_varimax()`: Pipeline completo con rotación ortogonal
  - `generar_word_clouds()`: Visualiza tópicos como nubes de palabras ponderadas
  - `asignar_topicos_a_conferencias()`: Asigna tópico dominante a cada documento
  - `visualizar_pca/kpca/tsne()`: Múltiples visualizaciones de reducción dimensional
  - `actualizar_numero_topicos()`: Actualización dinámica del modelo

- **Top_NMF**: Factorización de Matrices No-Negativas para modelado interpretable de tópicos
  - `aplicar_nmf()`: Calcula matrices W y H (documento-tópico y tópico-término)
  - `calcular_palabras_topicos()`: Extrae palabras principales por tópico con pesos
  - `generar_word_clouds()`: Crea visualizaciones interpretables de nubes de palabras
  - `asignar_topicos_documentos()`: Asignación documento-tópico
  - Métodos de visualización: PCA, Kernel PCA, t-SNE, gráficos interactivos con Plotly

**Funciones Auxiliares:**
- `elegir_num_topicos_estabilidad()`: Determina número óptimo de tópicos usando métricas de estabilidad (Spearman, JSD, similitud coseno)
- Reducción dimensional: `aplicar_pca()`, `aplicar_kernel_pca()`, `aplicar_tsne()`
- Visualización interactiva con Plotly: `visualizar_reduccion_plotly()`

#### `preprocesar.py`
Utilidades de preprocesamiento de texto para procesamiento del idioma español.

**Clase: preprocesaTexto**
- Tokenización (basada en NLTK)
- Normalización de mayúsculas/minúsculas (preserva acentos españoles: á, é, í, ó, ú, ñ, ü)
- Remoción de acentos para normalización
- Eliminación de puntuación y caracteres especiales
- Filtrado de números
- Lematización (spaCy: es_core_news_sm)
- Stemming (Snowball stemmer)
- Eliminación de stopwords (español)
- Pipeline modular con método `preprocesa()`

#### `RepositorioAMLO.py`
Módulo de carga y estructuración de datos de transcripciones de conferencias.

**Clase: AMLOmañaneras**
- Carga datos de conferencia individual por fecha
- Integración automática de preprocesamiento
- Metadatos temporales: semana del año, contador de semana acumulativa
- `texto()`: Retorna texto preprocesado para una conferencia específica
- Objetos comparables con operadores basados en fecha

**Función: matriz_mañaneras**
- Genera matriz de documentos para rangos de fechas (2018-2023)
- Retorna: textos, etiquetas de fecha, etiquetas de semana acumulativa
- Filtra conferencias vacías/no disponibles
- Salida de estadísticas resumidas

### Notebook de Análisis

#### `Tarea3_FranciscoJavierHernandezVelasco_Ejercicio2.ipynb`
Pipeline completo de análisis demostrando:

1. **Carga de Datos y Preprocesamiento**
   - Lista combinada de stopwords (spaCy + NLTK + stopwordsiso)
   - Stopwords personalizadas específicas del dominio
   - 1,213 conferencias procesadas

2. **Representación TF-IDF**
   - Tamaño de vocabulario: 1,200 características
   - Ajuste adaptativo de min_df (iniciando en 0.01)
   - Soporte de bi-gramas (1,2)

3. **Extracción de Tópicos**
   - Selección óptima de tópicos: 15 tópicos (basado en estabilidad)
   - SVD con rotación Varimax
   - Descomposición NMF
   - Tópicos etiquetados: Corrupción, Salud, Energía, Infraestructura, COVID-19, Seguridad, Migración, Política, Programas Económicos, Petróleo/PEMEX, Educación, Aeropuerto, Electoral, Economía, Desastres Naturales

4. **Visualización y Análisis**
   - Nubes de palabras para cada tópico
   - Embeddings 2D: PCA, Kernel PCA, t-SNE
   - Visualizaciones interactivas con Plotly con metadatos temporales
   - Análisis de silueta para validación de clusters

5. **Análisis Temporal**
   - Índices de tópicos semanales con ponderación de entropía
   - Series de tiempo normalizadas (min-max, z-score)
   - Visualización de tendencias anuales y multi-anuales
   - Detección de eventos (marcadores verticales para años/meses)

### Productos Generados

- **index.html**: Dashboard web interactivo con dinámica temporal de tópicos
- **Reporte_E2.pdf**: Reporte de análisis comprensivo con hallazgos
- **E2Tarea3_FranciscoJavierHernandezVelasco_w.pdf**: Versión extendida del reporte

## Implementación Técnica

### Técnicas Clave
- **Vectorización TF-IDF**: Ponderación de importancia documento-término
- **SVD Truncado**: Aproximación de rango bajo con k=15 componentes
- **Rotación Varimax**: Rotación ortogonal para mejorar interpretabilidad
- **Factorización de Matrices No-Negativas**: Representación aditiva basada en partes
- **Reducción Dimensional**: PCA (lineal), Kernel PCA (RBF), t-SNE (manifold)
- **Estabilidad de Tópicos**: Evaluación de estabilidad multi-ejecución con procesamiento paralelo

### Bibliotecas y Dependencias
- scikit-learn: TfidfVectorizer, TruncatedSVD, NMF, métodos de descomposición
- spaCy: Lematización en español (es_core_news_sm)
- NLTK: Tokenización, stopwords
- factor-analyzer: Rotación Varimax
- WordCloud: Representación visual de tópicos
- Plotly: Visualizaciones interactivas
- pandas, numpy: Manipulación de datos
- matplotlib, seaborn: Visualizaciones estáticas

## Fuente de Datos

Transcripciones de conferencias obtenidas de: [NOSTRODATA - Conferencias Matutinas AMLO](https://github.com/NOSTRODATA/conferencias_matutinas_amlo)

El dataset contiene transcripciones diarias de conferencias de prensa presidenciales desde diciembre de 2018 hasta diciembre de 2023, organizadas por fecha con segmentación a nivel de participante.

## Resultados y Hallazgos

El análisis identifica exitosamente 15 tópicos temáticos distintos a lo largo de 5 años de conferencias:

- Patrones temporales claros en la prevalencia de tópicos
- Correlación entre tópicos y eventos del mundo real (aumento de COVID-19 en 2020-2021, discusión de proyectos de infraestructura)
- NMF proporciona distribuciones de tópicos más interpretables que SVD
- La rotación Varimax mejora significativamente la coherencia de tópicos SVD

## Comparación de Metodologías

**SVD con Varimax:**
- Ventajas: Captura estructura de varianza, computación rápida, rotación mejora interpretabilidad
- Desventajas: Permite valores negativos (requiere rotación), componentes menos intuitivos

**NMF:**
- Ventajas: Restricciones no-negativas aseguran interpretabilidad aditiva, natural para datos de frecuencia
- Desventajas: Optimización no-convexa, sensible a inicialización

El proyecto demuestra ambos enfoques con visualizaciones comparativas.

## Ejemplo de Uso

```python
from RepositorioAMLO import matriz_mañaneras
from AnalisisTopicos import TF_IDF, Top_NMF

# Cargar conferencias
textos, fechas, semanas = matriz_mañaneras('2019-01-01', '2023-12-31')

# Crear matriz TF-IDF
tfidf = TF_IDF()
tfidf.configurar_vectorizador(textos, stopwords_list, tamaño=1200)

# Aplicar NMF
model = Top_NMF(num_topicos=15)
model.aplicar_nmf(tfidf.tfidf_matrix)
model.calcular_palabras_topicos(tfidf.vectorizador)

# Visualizar
model.generar_word_clouds()
model.visualizar_tsne(perplexity=15)
```

## Licencia

Ver archivo [LICENSE](LICENSE) para detalles.

## Autor

Francisco Javier Hernández Velasco

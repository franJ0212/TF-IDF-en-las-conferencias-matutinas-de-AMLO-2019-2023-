
########################################################################################################################################
# Matriz TF-IDF
########################################################################################################################################

from sklearn.feature_extraction.text import TfidfVectorizer

class TF_IDF:
    def __init__(self):
        self.tfidf_matrix = None
        self.vectorizador = None
        self.vocabulario = None

    def configurar_vectorizador(self, textos, final_stopwords_list, min_features=1200, tamaño=1000, alpha=0.10):
        if tamaño is None:
            tamaño = len(str(textos))
            tamaño_propuesto = int(alpha * tamaño) + 1
            print(tamaño_propuesto)
        else:
            tamaño_propuesto = tamaño
        min_df = 0.01
        intentos = 0
        max_intentos = 100

        while True:
            try:
                self.vectorizador = TfidfVectorizer(
                    max_df=0.95,
                    min_df=min_df,
                    max_features=tamaño_propuesto,
                    use_idf=True,
                    strip_accents='unicode',
                    stop_words=final_stopwords_list,
                    ngram_range=(1, 2)
                )
                
                self.tfidf_matrix = self.vectorizador.fit_transform(textos)
                self.vocabulario = self.vectorizador.get_feature_names_out()  # Almacenando el vocabulario
                num_features = len(self.vocabulario)
                
                if num_features >= min_features:
                    print(f"Vectorización exitosa con {num_features} características (min_df={min_df}).")
                    break
                else:
                    print(f"Solo se encontraron {num_features} características, menos que el mínimo requerido de {min_features}.")
            
            except ValueError as e:
                print(f"Error: {e}. Reduciendo min_df y reintentando...")
            
            min_df /= (10 * (intentos + 1))
            intentos += 1
            if intentos > max_intentos:
                print(f"No se pudo alcanzar el mínimo de características después de {max_intentos} intentos.")
                break

        return self.tfidf_matrix, self.vectorizador
'''

tf_idf = TF_IDF()
textos = ["Este es un documento de ejemplo.", "Otro texto para probar el vectorizador."]
stopwords = ['de', 'para', 'el']
tfidf_matrix, vocabulario = tf_idf.configurar_vectorizador(textos, stopwords, min_features=1, tamaño=len(textos))
print("Vocabulario:", vocabulario.get_feature_names_out())
tfidf_matrix.toarray()

'''

########################################################################################################################################
# Analisis con SVD
########################################################################################################################################

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd, svd_flip, safe_sparse_dot
from sklearn.utils.validation import check_random_state
from factor_analyzer.rotator import Rotator
import scipy.sparse as sp
from sklearn.utils.sparsefuncs import mean_variance_axis
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class Top_SVD:
    """
    Clase para realizar análisis de tópicos mediante descomposición SVD y rotación Varimax.

    Atributos:
        svd_model (TruncatedSVD): Modelo SVD entrenado.
        components_orig (np.array): Componentes originales del SVD.
        document_topic_matrix (np.array): Datos transformados en el espacio de tópicos SVD.
        svd_model_rot (np.array): Componentes rotados tras rotación Varimax.
        document_topic_matrix_rot (np.array): Datos transformados en el espacio de componentes rotados.
        topicos_asignados (list): Índices de tópicos asignados a cada documento.
        palabras_topicos (list of tuples): Palabras clave y pesos de cada tópico.

    Métodos:
        obtener_topicos_svd(matriz_tfidf): Realiza SVD en la matriz TF-IDF para extraer tópicos.
        fit_transform_rot(X, n_components): Aplica SVD y rotación Varimax.
        obtener_topicos_svd_con_varimax(matriz_tfidf): Combina SVD y Varimax para un análisis de tópicos detallado.
        obtener_palabras_topicos(vectorizer): Extrae palabras y pesos para tópicos identificados.
        generar_word_clouds(): Muestra nubes de palabras para cada tópico.
        asignar_topico_conferencia(matriz_documento_topico, indice_conferencia): Asigna el tópico más relevante a un documento.
        asignar_topicos_a_conferencias(): Asigna tópicos a todos los documentos procesados.

    Uso:
        Útil para análisis de texto en grandes colecciones de documentos, como en análisis de sentimientos y clasificación.
    """

    def __init__(self, num_topicos, etiquetas_topicos=None):
        self.svd_model = None
        self.components_orig = None
        self.document_topic_matrix = None
        self.svd_model_rot = None
        self.document_topic_matrix_rot = None
        self.topicos_asignados = []
        self.palabras_topicos = []
        self.num_topicos = num_topicos
        if etiquetas_topicos is not None and len(etiquetas_topicos) == num_topicos:
            self.nombre_topicos = etiquetas_topicos
        else:
            self.nombre_topicos = [f"Topico {i + 1}" for i in range(num_topicos)]


    def obtener_topicos_svd(self, matriz_tfidf):
        """
        Realiza la descomposición SVD sobre la matriz TF-IDF para extraer self.num_topicos tópicos.

        Parámetros:
        - matriz_tfidf: matriz TF-IDF de los documentos.

        Retorna:
        - svd_model: modelo SVD entrenado.
        - document_topic_matrix: datos transformados en el espacio de tópicos.
        """
        self.svd_model = TruncatedSVD(n_components=self.num_topicos, random_state=12)
        self.document_topic_matrix = self.svd_model.fit_transform(matriz_tfidf)
        self.components_orig = self.svd_model.components_
        return self.svd_model, self.document_topic_matrix

    def fit_transform_rot(self, X, n_components, algorithm='randomized', n_iter=5, random_state=None, tol=0.0):
        """
        Realiza la descomposición SVD y una rotación Varimax sobre la matriz X para extraer componentes tópicos.

        Parámetros:
        - X: Matriz de datos, {array-like, sparse matrix} de forma (n_samples, n_features).
        - n_components: Número de componentes a extraer.
        - algorithm: 'randomized' para SVD randomizada o 'arpack' para la versión ARPACK.
        - n_iter: Número de iteraciones para SVD randomizada.
        - random_state: Semilla para la reproducibilidad.
        - tol: Tolerancia para el criterio de parada en ARPACK.

        Retorna:
        - X_transformed_rot: Datos transformados en el espacio de componentes rotados.
        - svd_model_rot: Componentes rotados tras la descomposición SVD y rotación Varimax.
        
        Nota: Usamos como base el metodo fit_transform de TruncatedSVD de:
        # Author: Lars Buitinck
        #         Olivier Grisel <olivier.grisel@ensta.org>
        #         Michael Becker <mike@beckerfuffle.com>
        # License: 3-clause BSD.
        
        """
        random_state = check_random_state(random_state)
        
        if algorithm == 'arpack':
            U, Sigma, VT = svds(X, k=n_components, tol=tol)
            Sigma = Sigma[::-1]
            U, VT = svd_flip(U[:, ::-1], VT[::-1])
        elif algorithm == 'randomized':
            U, Sigma, VT = randomized_svd(
                X,
                n_components=n_components,
                n_iter=n_iter,
                random_state=random_state
            )
        else:
            raise ValueError("El algoritmo debe ser 'randomized' o 'arpack'")
        
        rotator = Rotator(method='varimax')
        VT_rot = rotator.fit_transform(VT.T).T
        X_transformed_rot = safe_sparse_dot(X, VT_rot.T)
        self.svd_model_rot = VT_rot
        return X_transformed_rot, VT_rot

    def obtener_topicos_svd_con_varimax(self, matriz_tfidf):
        """
        Utiliza fit_transform_rot para realizar SVD y rotación Varimax sobre la matriz TF-IDF.

        Parámetros:
        - matriz_tfidf: matriz TF-IDF de los documentos.
        Retorna:
        - components_orig: Componentes originales de la SVD.
        - document_topic_matrix: Datos transformados en el espacio de tópicos.
        - svd_model_rot: Componentes rotados.
        - document_topic_matrix_rot: Datos transformados en el espacio de componentes rotados.
        """
        _, _ = self.obtener_topicos_svd(matriz_tfidf)
        self.document_topic_matrix_rot, self.svd_model_rot = self.fit_transform_rot(matriz_tfidf, self.num_topicos)
        return self.components_orig, self.document_topic_matrix, self.svd_model_rot, self.document_topic_matrix_rot

    def obtener_palabras_topicos(self, vectorizer, max_palabras=80):
        """
        Obtiene las palabras y sus pesos para cada tópico a partir de la matriz V(k) del modelo SVD.

        Parámetros:
        - vectorizer: vectorizador utilizado para crear la matriz TF-IDF.
        - max_palabras: número máximo de palabras a incluir por tópico.

        Retorna:
        - palabras_topicos: lista de listas de palabras y sus pesos para cada tópico.
        """
        palabras = vectorizer.get_feature_names_out()
        self.palabras_topicos = []

        for i in range(self.num_topicos):
            topico = self.components_orig[i]
            indices_top = topico.argsort()[::-1][:max_palabras]  # Limita el número de palabras por tópico
            palabras_top = [(palabras[j], topico[j]) for j in indices_top]
            self.palabras_topicos.append(palabras_top)

        return self.palabras_topicos

    def generar_word_clouds(self, vectorizador, use_rotated=True, max_palabras=80, save_images=False, width=800, height=400, show=True):
        """
        Genera y opcionalmente guarda word clouds para cada tópico utilizando las palabras y sus pesos,
        limitando el número de palabras usadas para cada nube y usando nombres personalizados para los títulos.

        Parámetros:
        - vectorizador: vectorizador de Tfidf utilizado para obtener nombres de palabras.
        - use_rotated: booleano que indica si usar los componentes rotados para la visualización.
        - max_palabras: número máximo de palabras a incluir en cada nube de palabras.
        - save_images: booleano que indica si guardar las imágenes generadas.
        - width: ancho de la imagen en píxeles.
        - height: altura de la imagen en píxeles.
        - show: booleano que indica si mostrar las imágenes generadas.
        """
        components = self.svd_model_rot if use_rotated else self.svd_model.components_

        for i in range(self.num_topicos):
            topic_weights = components[i]
            top_indices = topic_weights.argsort()[-max_palabras:]  # Limita el número de palabras por tópico
            palabras_top = {vectorizador.get_feature_names_out()[j]: topic_weights[j] for j in top_indices[::-1]}
            
            # Generar la WordCloud
            wordcloud = WordCloud(width=width, height=height, background_color='white').generate_from_frequencies(palabras_top)
            
            # Mostrar la WordCloud
            if show:
                plt.figure(figsize=(width / 100, height / 100))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(self.nombre_topicos[i] if i < len(self.nombre_topicos) else f'Tópico {i + 1} {"(Rotado)" if use_rotated else "(Original)"}')
                plt.show()

            # Guardar la WordCloud
            if save_images:
                image_filename = f"{self.nombre_topicos[i].replace(' ', '_') if i < len(self.nombre_topicos) else f'Topico_{i + 1}'}{'_Rotado' if use_rotated else '_Original'}.png"
                wordcloud.to_file(image_filename)
            
    def asignar_topicos_a_conferencias(self, rot=True):
        """
        Itera sobre cada documento en la matriz de tópicos seleccionada (rotada o no) y asigna un tópico usando `asignar_topico_conferencia`.
        Almacena los resultados en `topicos_asignados`.
        
        Parámetros:
        - rot: booleano que indica si usar la matriz de tópicos rotados para la asignación.
        """
        # Seleccionar la matriz de tópicos apropiada
        matrix_to_use = self.document_topic_matrix_rot if rot else self.document_topic_matrix
        
        # Asignar tópicos usando la matriz seleccionada
        self.topicos_asignados = [self.asignar_topico_conferencia(matrix_to_use, i) for i in range(matrix_to_use.shape[0])]

    def asignar_topico_conferencia(self, matriz_documento_topico, indice_conferencia):
        """
        Asigna el tópico correspondiente a una conferencia utilizando el valor máximo de cada renglón de la matriz documento-tópico.
        
        Parámetros:
        - matriz_documento_topico: matriz documento-tópico o documento-tópico rotada.
        - indice_conferencia: índice de la conferencia en la matriz documento-tópico.
        
        Retorna:
        - topico_asignado: tópico asignado a la conferencia.
        - valor_maximo: valor máximo del tópico asignado.
        """
        renglon_conferencia = matriz_documento_topico[indice_conferencia]
        topico_asignado = np.argmax(renglon_conferencia)
        valor_maximo = np.max(renglon_conferencia)
        
        return topico_asignado

    def establecer_nombres_topicos(self, nombres):
        """
        Establece o actualiza los nombres de los tópicos. Si no se proporcionan suficientes nombres,
        los nombres faltantes se generarán automáticamente como 'Topico n', 'Topico n+1', etc.
        
        Args:
        nombres (list of str): Lista de nombres para los tópicos.
        """
        num_topicos = len(set(self.topicos_asignados))
        if len(nombres) < num_topicos:
            nombres.extend([f"Topico {i + 1}" for i in range(len(nombres), num_topicos)])
        self.nombre_topicos = nombres

    def visualizar_pca(self, save=False, filename='pca_plot.png', dpi=300, figsize=(10, 6)):
        if self.document_topic_matrix is not None:
            visualizarPCA_mpl(self.document_topic_matrix, self.topicos_asignados, self.nombre_topicos, save, filename, dpi, figsize)
        else:
            print("No se ha inicializado la matriz de documentos-topicos.")

    def visualizar_kpca(self, save=False, filename='kpca_plot.png', dpi=300, figsize=(10, 6)):
        if self.document_topic_matrix_rot is not None:
            visualizarKPCA_mpl(self.document_topic_matrix_rot, self.topicos_asignados, self.nombre_topicos, save, filename, dpi, figsize)
        else:
            print("No se ha inicializado la matriz de documentos-topicos rotada.")

    def visualizar_tsne(self, perplexity=30, save=False, filename='tsne_plot.png', dpi=300, figsize=(10, 6)):
        if self.document_topic_matrix_rot is not None:
            visualizarTSNE_mpl(self.document_topic_matrix_rot, self.topicos_asignados, perplexity, self.nombre_topicos, save, filename, dpi, figsize)
        else:
            print("No se ha inicializado la matriz de documentos-topicos rotada.")

    def actualizar_numero_topicos(self, nuevo_num_topicos, tfidf_matrix, vectorizador, nombres=None):
        # Actualizar el número de tópicos
        self.num_topicos = nuevo_num_topicos
        
        # Aplicar SVD con Varimax utilizando la nueva matriz TF-IDF
        self.obtener_topicos_svd_con_varimax(tfidf_matrix)
        
        # Actualizar el vectorizador y obtener palabras tópicas
        self.obtener_palabras_topicos(vectorizador)
        
        # Asignar tópicos a conferencias usando la nueva matriz de tópicos rotada
        self.asignar_topicos_a_conferencias(rot=True)
        
        # Establecer o actualizar nombres de tópicos si se proporcionan
        if nombres:
            self.establecer_nombres_topicos(nombres)
        else:
            self.nombre_topicos = [f"Topico {i + 1}" for i in range(nuevo_num_topicos)]

    def visualizar_topicos(self, method_name='PCA', y_date=None, y_cum=None, etiquetas=None, perplexity=30, use_rotated=True):
        """
        Aplica reducción de dimensionalidad y visualiza los tópicos utilizando Plotly.
        
        Args:
        method_name (str): Método de reducción de dimensionalidad ('PCA', 'KernelPCA', 'TSNE').
        use_rotated (bool): Indica si se debe usar la matriz de tópicos rotada.
        y_date (list of datetime): Fechas para mostrar en los tooltips.
        y_cum (list): Datos acumulados para mostrar en los tooltips.
        etiquetas (list of str): Etiquetas para los tópicos.
        perplexity (int): Perplexidad para t-SNE, relevante solo si se utiliza t-SNE.
        """
        # Elegir la matriz adecuada
        matriz = self.document_topic_matrix_rot if use_rotated else self.document_topic_matrix

        # Aplicar la reducción de dimensionalidad seleccionada
        if method_name == 'PCA':
            embeddings = aplicar_pca(matriz)
        elif method_name == 'KernelPCA':
            embeddings = aplicar_kernel_pca(matriz)
        elif method_name == 'TSNE':
            embeddings = aplicar_tsne(matriz, perplexity)
        else:
            raise ValueError("Método de reducción de dimensionalidad no soportado.")

        # Asegurar que haya etiquetas para todos los tópicos
        if not etiquetas:
            etiquetas = self.nombre_topicos
        elif len(etiquetas) < len(set(self.topicos_asignados)):
            etiquetas.extend([f"Topico {i + 1}" for i in range(len(etiquetas), len(set(self.topicos_asignados)))])

        # Visualizar utilizando Plotly
        visualizar_reduccion_plotly(embeddings, self.topicos_asignados, y_date, y_cum, method_name, etiquetas)

########################################################################################################################################
# Analisis con NMF
########################################################################################################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Top_NMF:
    def __init__(self, num_topicos, etiquetas_topicos=None):
        # Modelos y matrices relacionadas con NMF
        self.nmf_model = None
        self.components = None
        self.document_topic_matrix = None
        
        # Asignaciones y meta información
        self.topicos_asignados = []
        self.palabras_topicos = []
        
        # Configuración de tópicos
        self.num_topicos = num_topicos
        
        # Inicializar nombres de tópicos
        if etiquetas_topicos is not None and len(etiquetas_topicos) == num_topicos:
            self.nombre_topicos = etiquetas_topicos
        else:
            self.nombre_topicos = [f"Topico {i + 1}" for i in range(num_topicos)]

    def aplicar_nmf(self, tf_idf_df):
        """
        Ajusta el modelo NMF al dataframe TF-IDF y actualiza los atributos con las matrices resultantes W y H.
        
        Args:
        tf_idf_df (DataFrame): DataFrame que contiene la matriz TF-IDF sobre la cual aplicar NMF.
        """
        self.nmf_model = NMF(n_components=self.num_topicos, random_state=12, max_iter=500)
        W = self.nmf_model.fit_transform(tf_idf_df)  # Matriz de características documento-tópico
        H = self.nmf_model.components_  # Matriz de tópicos-términos (componentes)

        # Actualizar los atributos de la clase
        self.document_topic_matrix = W
        self.components_orig = H

        return W, H

    def calcular_palabras_topicos(self, vectorizador):
        """
        Calcula las palabras más significativas y sus pesos para cada tópico a partir de los componentes NMF (matriz H).
        
        Args:
        vectorizador (TfidfVectorizer): El vectorizador que se usó para generar la matriz TF-IDF.
        """
        feature_names = vectorizador.get_feature_names_out()
        self.palabras_topicos = []
        for topic_idx, topic in enumerate(self.components_orig):
            # Crear un diccionario de palabras y sus pesos
            topic_dict = {feature_names[i]: topic[i] for i in topic.argsort()[:-100:-1]}  # Top 50 palabras en lugar de Top 20
            self.palabras_topicos.append(topic_dict)


    def generar_word_clouds(self, save_images=False, width=800, height=400, show=True):
        """
        Genera y opcionalmente guarda word clouds para cada tópico utilizando las palabras y sus pesos almacenados en `palabras_topicos`.
        
        Args:
        - save_images: booleano que indica si guardar las imágenes generadas.
        - width: ancho de la imagen en píxeles.
        - height: altura de la imagen en píxeles.
        - show: booleano que indica si mostrar las imágenes generadas.
        """
        for i, topic_dict in enumerate(self.palabras_topicos):
            # Generar la WordCloud
            wordcloud = WordCloud(width=width, height=height, background_color='white').generate_from_frequencies(topic_dict)
            
            # Mostrar la WordCloud
            if show:
                plt.figure(figsize=(width / 100, height / 100))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(self.nombre_topicos[i] if i < len(self.nombre_topicos) else f'Tópico {i + 1}')
                plt.show()

            # Guardar la WordCloud
            if save_images:
                image_filename = f"{self.nombre_topicos[i].replace(' ', '_') if i < len(self.nombre_topicos) else f'Topico_{i + 1}'}_WordCloud.png"
                wordcloud.to_file(image_filename)

    def asignar_topicos_documentos(self):
        """
        Asigna el tópico más representativo a cada documento basándose en la matriz documento-tópico.
        """
        # Asegúrate de que la matriz documento-tópico ha sido calculada
        if self.document_topic_matrix is None:
            raise ValueError("La matriz documento-tópico no ha sido inicializada.")
        
        # Asignar tópicos a cada documento
        self.topicos_asignados = []
        for i in range(self.document_topic_matrix.shape[0]):
            renglon_documento = self.document_topic_matrix[i]
            topico_asignado = np.argmax(renglon_documento)
            self.topicos_asignados.append(topico_asignado)

        return self.topicos_asignados

    def establecer_nombres_topicos(self, nombres):
        """
        Establece o actualiza los nombres de los tópicos. Si no se proporcionan suficientes nombres,
        los nombres faltantes se generarán automáticamente como 'Topico n', 'Topico n+1', etc.

        Args:
        nombres (list of str): Lista de nombres para los tópicos.
        """
        if len(nombres) < self.num_topicos:
            nombres.extend([f"Topico {i + 1}" for i in range(len(nombres), self.num_topicos)])
        self.nombre_topicos = nombres

    def visualizar_pca(self, save=False, filename='pca_plot.png', dpi=300, figsize=(10, 6)):
        """
        Visualiza la reducción de dimensionalidad usando PCA.
        """
        if self.document_topic_matrix is not None:
            visualizarPCA_mpl(self.document_topic_matrix, self.topicos_asignados, self.nombre_topicos, save, filename, dpi, figsize)
        else:
            print("No se ha inicializado la matriz de documentos-topicos.")

    def visualizar_kpca(self, save=False, filename='kpca_plot.png', dpi=300, figsize=(10, 6)):
        """
        Visualiza la reducción de dimensionalidad usando Kernel PCA.
        """
        if self.document_topic_matrix is not None:
            visualizarKPCA_mpl(self.document_topic_matrix, self.topicos_asignados, self.nombre_topicos, save, filename, dpi, figsize)
        else:
            print("No se ha inicializado la matriz de documentos-topicos.")

    def visualizar_tsne(self, perplexity=30, save=False, filename='tsne_plot.png', dpi=300, figsize=(10, 6)):
        """
        Visualiza la reducción de dimensionalidad usando t-SNE.
        """
        if self.document_topic_matrix is not None:
            visualizarTSNE_mpl(self.document_topic_matrix, self.topicos_asignados, perplexity, self.nombre_topicos, save, filename, dpi, figsize)
        else:
            print("No se ha inicializado la matriz de documentos-topicos.")

    def visualizar_topicos(self, method_name='PCA', y_date=None, y_cum=None, etiquetas=None, perplexity=30):
        """
        Aplica reducción de dimensionalidad y visualiza los tópicos utilizando Plotly.

        Args:
        method_name (str): Método de reducción de dimensionalidad ('PCA', 'KernelPCA', 'TSNE').
        y_date (list of datetime): Fechas para mostrar en los tooltips.
        y_cum (list): Datos acumulados para mostrar en los tooltips.
        etiquetas (list of str): Etiquetas para los tópicos.
        perplexity (int): Perplexidad para t-SNE, relevante solo si se utiliza t-SNE.
        """
        # Asegurarse de que la matriz documento-tópico está inicializada
        if self.document_topic_matrix is None:
            raise ValueError("La matriz de documentos-topicos no ha sido inicializada.")

        # Aplicar la reducción de dimensionalidad seleccionada
        if method_name == 'PCA':
            embeddings = aplicar_pca(self.document_topic_matrix)
        elif method_name == 'KernelPCA':
            embeddings = aplicar_kernel_pca(self.document_topic_matrix)
        elif method_name == 'TSNE':
            embeddings = aplicar_tsne(self.document_topic_matrix, perplexity)
        else:
            raise ValueError("Método de reducción de dimensionalidad no soportado.")

        # Asegurar que haya etiquetas para todos los tópicos
        if etiquetas is None:
            etiquetas = self.nombre_topicos

        # Visualizar utilizando Plotly
        visualizar_reduccion_plotly(embeddings, self.topicos_asignados, y_date, y_cum, method_name, etiquetas)

    def actualizar_numero_topicos(self, nuevo_num_topicos, tfidf_matrix, vectorizador, nombres=None):
        """
        Actualiza el número de tópicos y recalcula los componentes del modelo NMF con la nueva matriz TF-IDF.

        Args:
        nuevo_num_topicos (int): El nuevo número de tópicos.
        tfidf_matrix (sparse matrix): Nueva matriz TF-IDF.
        vectorizador (TfidfVectorizer): Vectorizador usado para generar la nueva matriz TF-IDF.
        nombres (list of str): Lista opcional de nombres para los tópicos. Si no se proporcionan, se generan automáticamente.
        """
        # Actualizar el número de tópicos
        self.num_topicos = nuevo_num_topicos
        
        # Aplicar NMF utilizando la nueva matriz TF-IDF
        self.aplicar_nmf(tfidf_matrix)
        
        # Actualizar palabras tópicas
        self.calcular_palabras_topicos(vectorizador)
        
        # Asignar tópicos a documentos
        self.asignar_topicos_documentos()
        
        # Establecer o actualizar nombres de tópicos si se proporcionan
        if nombres:
            self.establecer_nombres_topicos(nombres)
        else:
            self.nombre_topicos = [f"Topico {i + 1}" for i in range(nuevo_num_topicos)]


########################################################################################################################################
# Graficas
########################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from matplotlib import cm

def aplicar_pca(matriz_documento_topico):
    pca = PCA(n_components=2)
    return pca.fit_transform(matriz_documento_topico)

def aplicar_kernel_pca(matriz_documento_topico):
    kernel_pca = KernelPCA(n_components=2, kernel='rbf')
    return kernel_pca.fit_transform(matriz_documento_topico)

def aplicar_tsne(matriz_documento_topico, perplexity):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=12)
    return tsne.fit_transform(matriz_documento_topico)

def agregar_puntos_mpl(ax, embeddings, topicos_asignados, viridis, alpha=0.7):
    for topic, color in zip(set(topicos_asignados), viridis.colors):
        topic_mask = [t == topic for t in topicos_asignados]
        ax.scatter(embeddings[topic_mask, 0], embeddings[topic_mask, 1], c=[color], s=50, alpha=alpha)

def agregar_leyenda_mpl(ax, topicos_asignados, viridis, etiq=None):
    topicos_unicos = sorted(set(topicos_asignados))
    num_topicos = len(topicos_unicos)
    
    if etiq is None or len(etiq) < num_topicos:
        if etiq is None:
            etiq = []
        etiq.extend([f"Topic {i + 1}" for i in range(len(etiq), num_topicos)])

    for i, topic in enumerate(topicos_unicos):
        color_index = topicos_unicos.index(topic)
        label = etiq[color_index] if i < len(etiq) else f"Topic {topic + 1}"
        ax.scatter([], [], c=[viridis.colors[color_index]], label=label, s=100)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def visualizarPCA_mpl(matriz_documento_topico, topicos_asignados, etiq=None, save=False, filename='pca_plot.png', dpi=300, figsize=(10, 6)):
    viridis = cm.get_cmap('viridis', len(set(topicos_asignados)))
    pca_embeddings = aplicar_pca(matriz_documento_topico)
    
    fig, ax = plt.subplots(figsize=figsize)
    agregar_puntos_mpl(ax, pca_embeddings, topicos_asignados, viridis)
    agregar_leyenda_mpl(ax, topicos_asignados, viridis, etiq)
    ax.set_title("PCA")
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=dpi)
    plt.show()

def visualizarKPCA_mpl(matriz_documento_topico, topicos_asignados, etiq=None, save=False, filename='kpca_plot.png', dpi=300, figsize=(10, 6)):
    viridis = cm.get_cmap('viridis', len(set(topicos_asignados)))
    kernel_pca_embeddings = aplicar_kernel_pca(matriz_documento_topico)
    
    fig, ax = plt.subplots(figsize=figsize)
    agregar_puntos_mpl(ax, kernel_pca_embeddings, topicos_asignados, viridis)
    agregar_leyenda_mpl(ax, topicos_asignados, viridis, etiq)
    ax.set_title("Kernel PCA")
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=dpi)
    plt.show()

def visualizarTSNE_mpl(matriz_documento_topico, topicos_asignados, perplexity=30, etiq=None, save=False, filename='tsne_plot.png', dpi=300, figsize=(10, 6)):
    viridis = cm.get_cmap('viridis', len(set(topicos_asignados)))
    tsne_embeddings = aplicar_tsne(matriz_documento_topico, perplexity)
    
    fig, ax = plt.subplots(figsize=figsize)
    agregar_puntos_mpl(ax, tsne_embeddings, topicos_asignados, viridis)
    agregar_leyenda_mpl(ax, topicos_asignados, viridis, etiq)
    ax.set_title("t-SNE")
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=dpi)
    plt.show()
    
########################################################################################################################################
# Ploty
########################################################################################################################################

import pandas as pd
import plotly.graph_objects as go
from matplotlib import cm

def agregar_puntos_interactivos(df, fig, etiquetas):
    for topic in df['topic'].unique():
        df_topic = df[df['topic'] == topic]
        fig.add_trace(go.Scatter(
            x=df_topic['x'],
            y=df_topic['y'],
            mode='markers',
            marker=dict(color=df_topic['color'].iloc[0], size=10, opacity=0.7),
            name=etiquetas[topic],  # Usa etiquetas personalizadas
            text=df_topic['hover_info'],
            hoverinfo='text'
        ))

def visualizar_reduccion_plotly(embeddings, topicos_asignados, y_date, y_cum, method_name='Method', etiquetas=None):
    # Generar colores utilizando el mapa de colores 'viridis' de Matplotlib
    viridis = cm.get_cmap('viridis', len(set(topicos_asignados)))
    colores = [viridis(i) for i in range(viridis.N)]
    colores_hex = ["#{:02x}{:02x}{:02x}".format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colores]
    
    # Asegurar que haya etiquetas suficientes para todos los tópicos únicos
    num_topicos = len(set(topicos_asignados))
    if not etiquetas or len(etiquetas) < num_topicos:
        if not etiquetas:
            etiquetas = []
        etiquetas.extend([f"Tópico {i + 1}" for i in range(len(etiquetas), num_topicos)])
    
    # Preparar los datos para Plotly con información adicional para el tooltip
    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'topic': topicos_asignados,
        'color': [colores_hex[t % num_topicos] for t in topicos_asignados],
        'hover_info': [f'Año: {date.year}, Mes: {date.month}, Tópico: {etiquetas[topic]}, Semana acumulada: {cum}'
                       for date, topic, cum in zip(y_date, topicos_asignados, y_cum)]
    })
    
    # Iniciar la figura Plotly
    fig = go.Figure()
    
    # Agregar puntos al gráfico
    agregar_puntos_interactivos(df, fig, etiquetas)
    
    # Configuraciones adicionales de la figura
    fig.update_layout(
        title=f'Visualización de {method_name} con Plotly',
        xaxis_title="Componente Principal 1",
        yaxis_title="Componente Principal 2",
        legend_title="Tópicos",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Mostrar el gráfico
    fig.show()
    
    ##############################################################################################################################################################################
    
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import NMF
from joblib import Parallel, delayed

def calcular_estabilidad(H, num_topicos, medida):
    if medida == 'jsd':
        estabilidad = np.mean([jensenshannon(H[i], H[j]) for i in range(num_topicos) for j in range(i+1, num_topicos)])
    elif medida == 'coseno':
        estabilidad = np.mean([np.dot(H[i], H[j]) / (np.linalg.norm(H[i]) * np.linalg.norm(H[j])) for i in range(num_topicos) for j in range(i+1, num_topicos)])
    elif medida == 'spearman':
        estabilidad = np.mean([spearmanr(H[i], H[j])[0] for i in range(num_topicos) for j in range(i+1, num_topicos)])
    else:
        raise ValueError("Medida no válida. Debe ser 'jsd', 'coseno' o 'spearman'.")
    return estabilidad

def calcular_estabilidad_topicos(matriz_documentos, num_topicos, num_iteraciones, medida):
    estabilidad_topicos = []
    
    for _ in range(num_iteraciones):
        nmf = NMF(n_components=num_topicos, random_state=None)
        nmf.fit(matriz_documentos)
        H = nmf.components_
        
        estabilidad = calcular_estabilidad(H, num_topicos, medida)
        estabilidad_topicos.append(estabilidad)
    
    return np.mean(estabilidad_topicos)

def elegir_num_topicos_estabilidad(matriz_documentos, rango_topicos, num_iteraciones=10, medida='spearman', n_jobs=-1):
    """
    Elige el número óptimo de tópicos basado en la estabilidad utilizando diferentes medidas de similitud.
    
    Parámetros:
    - matriz_documentos: matriz de frecuencia de términos en los documentos.
    - rango_topicos: rango de números de tópicos a evaluar (ejemplo: range(2, 21)).
    - num_iteraciones: número de iteraciones para calcular la estabilidad (por defecto: 10).
    - medida: medida de similitud a utilizar ('jsd', 'coseno' o 'spearman'; por defecto: 'jsd').
    - n_jobs: número de trabajos paralelos a utilizar (por defecto: -1, todos los núcleos disponibles).
    
    Retorna:
    - num_topicos_optimo: número óptimo de tópicos basado en la estabilidad.
    """
    estabilidades = Parallel(n_jobs=n_jobs)(
        delayed(calcular_estabilidad_topicos)(matriz_documentos, num_topicos, num_iteraciones, medida)
        for num_topicos in rango_topicos
    )
    
    num_topicos_optimo = rango_topicos[np.argmax(estabilidades)]
    
    return num_topicos_optimo
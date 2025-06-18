import pandas as pd
from datetime import timedelta, datetime
import locale
import os
from preprocesar import *

'''
El repositorio con los datos son de: @nostrodata
https://github.com/NOSTRODATA/conferencias_matutinas_amlo?tab=readme-ov-file

preprocesar es la funcion que vimos en la ayudantia

'''

# Configurando el locale para español
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')  # Ajusta de idioma
# Corregimos los primeros archivo que tenian un - antes de del nombre.

class AMLOmañaneras:
    def __init__(self, año, mes, dia):
        self.año = año
        self.mes = mes
        self.dia = dia
        self.fecha = pd.Timestamp(f'{año}-{mes}-{dia}')  # Fecha específica
        self.semana_año = self.fecha.isocalendar().week  # Número de semana en el año
        self.semana_cum = self.calcular_semana_cumulativa(self.fecha)  # Semanas acumuladas desde el inicio
        self.ruta_base = 'C:\\Users\\Javier CIMAT\\Documents\\AMLO'  # Ruta base donde se encuentran los datos
        self.datos = self.cargar_dia()  # Cargar los datos del día específico

    def cargar_dia(self):
        """Carga los datos de la conferencia mañanera para el día especificado."""
        ruta_archivo = self.generar_ruta_archivo()
        try:
            datos_dia = pd.read_csv(ruta_archivo).drop(columns=['Sentimiento', 'Participante'])
            return datos_dia
        except FileNotFoundError:
            #print(f"Archivo no encontrado: {ruta_archivo}")
            return pd.DataFrame()  # Devuelve un DataFrame vacío si no se encuentra el archivo

    def generar_ruta_archivo(self):
        """Genera la ruta completa del archivo CSV para el día especificado."""
        nombre_mes = self.fecha.strftime('%B').lower()
        ruta_archivo = f"{self.ruta_base}\\{self.fecha.year}\\{self.fecha.month}-{self.fecha.year}\\{nombre_mes} {self.fecha.day}, {self.fecha.year}\\csv_por_participante\\PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv"
        return ruta_archivo

    def calcular_semana_cumulativa(self, fecha):
        """Calcula el número de semanas acumuladas desde la fecha inicial del documento."""
        fecha_inicial = pd.Timestamp('2018-12-03')  # Fecha de inicio del primer documento
        diferencia = fecha - fecha_inicial
        return diferencia.days // 7 + 1

    def __str__(self):
        """Devuelve una representación en string del objeto, mostrando la fecha y los detalles de los datos cargados."""
        if self.datos.empty:
            info_datos = "No hay datos cargados."
        else:
            info_datos = f"Datos cargados para el {self.fecha.strftime('%Y-%m-%d')}."
        return f"AMLOmañaneras del día {self.fecha.strftime('%Y-%m-%d')} - Semana del año: {self.semana_año}, Semana acumulada: {self.semana_cum}\n{info_datos}"

    def texto(self, preprocesar=True):
        """
        Devuelve el texto de las conferencias del día especificado.
        """
        if self.datos.empty:
            return "No hay texto disponible."

        texto_resultante = self.datos['Texto'].astype(str)

        if preprocesar:
            procesador = preprocesaTexto(idioma='es',
                                         _tokeniza=False,
                                         _muestraCambios=False,
                                         _quitarAcentos=True,
                                         _remueveStop=True,
                                         _stemming=False,
                                         _lematiza=True,
                                         _removerPuntuacion=True)
            texto_resultante = procesador.preprocesa(' '.join(texto_resultante))

        return texto_resultante
    
    def __eq__(self, otro):
        return self.fecha == otro.fecha

    def __lt__(self, otro):
        return self.fecha < otro.fecha

    def __le__(self, otro):
        return self.fecha <= otro.fecha

    def __gt__(self, otro):
        return self.fecha > otro.fecha

    def __ge__(self, otro):
        return self.fecha >= otro.fecha
    
    def obtener_ruta(self):
        """Devuelve la ruta del archivo para el día específico."""
        nombre_mes = self.fecha.strftime('%B').lower()
        ruta_archivo = f"{self.ruta_base}\\{self.fecha.year}\\{self.fecha.month}-{self.fecha.year}\\{nombre_mes} {self.fecha.day}, {self.fecha.year}\\csv_por_participante\\PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv"
        return ruta_archivo
    
'''

# Crear una instancia para el día específico, por ejemplo, 15 de marzo de 2021
mañaneras_dia_especifico = AMLOmañaneras(2021, 3, 15)

# Imprimir información básica y el texto de las mañaneras de ese día
print(mañaneras_dia_especifico)
print("Texto de las Mañaneras del día:")
print(mañaneras_dia_especifico.texto())

# Crear dos instancias de la clase para diferentes días
mañaneras_dia_15_marzo = AMLOmañaneras(2021, 3, 15)
mañaneras_dia_16_marzo = AMLOmañaneras(2021, 3, 16)

# Comparar las dos instancias
print("Son iguales:", mañaneras_dia_15_marzo == mañaneras_dia_16_marzo)  # False
print("15 de marzo es anterior al 16 de marzo:", mañaneras_dia_15_marzo < mañaneras_dia_16_marzo)  # True

'''

#############################################################################################################################################################################################################################

def matriz_mañaneras(fecha_inicio, fecha_fin, all_dates=False):
    """
    Genera una matriz de textos de las conferencias mañaneras de AMLO dentro de un rango de fechas especificado.
    La matriz solo incluye entradas para los días en los cuales se pudo cargar y procesar texto efectivamente.

    Parámetros:
    fecha_inicio (str o pd.Timestamp): Fecha de inicio del rango para cargar mañaneras. Si all_dates es True, se ignora.
    fecha_fin (str o pd.Timestamp): Fecha final del rango para cargar mañaneras. Si all_dates es True, se ignora.
    all_dates (bool): Si es True, ignora las fechas de inicio y fin proporcionadas y utiliza el rango completo desde el 
    inicio de las mañaneras hasta el 31 de diciembre de 2023.

    Retorna:
    tuple: Tres listas:
        - textos (list): Los textos de las mañaneras que se han podido cargar.
        - etiquetas (list): Las fechas correspondientes a cada texto cargado.
        - etiquetas_cum (list): El número de semana acumulada desde la fecha de inicio para cada texto cargado.

    Notas:
    - La función imprime un mensaje final que indica las fechas de la primera y última mañanera efectivamente cargada y el total de mañaneras cargadas.
    - Los días para los cuales no se encuentra texto ("No hay texto disponible.") son omitidos en las listas de retorno.
    """
    if all_dates:
        fecha_inicio = pd.Timestamp('2018-12-03')  # Fecha de inicio de las mañaneras
        fecha_fin = pd.Timestamp('2023-12-31')  # Máximo definido

    else:
        fecha_inicio = pd.Timestamp(fecha_inicio)
        fecha_fin = pd.Timestamp(fecha_fin)
    
    # Genera todas las fechas en el rango
    rango_fechas = pd.date_range(fecha_inicio, fecha_fin, freq='D')

    textos = []
    etiquetas = []
    etiquetas_cum = []

    fecha_primera = None  # Para guardar la fecha de la primera mañanera cargada
    fecha_ultima = None   # Para guardar la fecha de la última mañanera cargada

    for fecha in rango_fechas:
        mañanera = AMLOmañaneras(fecha.year, fecha.month, fecha.day)
        texto = mañanera.texto()
        
        if texto != "No hay texto disponible.":
            if fecha_primera is None:
                fecha_primera = fecha  # Establece la primera fecha efectiva
            fecha_ultima = fecha  # Siempre actualiza a la última fecha efectiva encontrada
            
            textos.append(texto)
            etiquetas.append(fecha)
            etiquetas_cum.append(mañanera.semana_cum)

    if fecha_primera and fecha_ultima:
        # Impresión del resumen al final de la ejecución
        print(f"\nMañaneras del {fecha_primera.strftime('%Y-%m-%d')} al {fecha_ultima.strftime('%Y-%m-%d')}. \nTotal mañaneras cargadas: {len(textos)}\n")
    else:
        print("\n No se encontraron mañaneras en el rango de fechas especificado!\n")

    return textos, etiquetas, etiquetas_cum

'''

# Ejemplo de uso:
textos, fechas, semanas_cum = matriz_mañaneras('2018-12-01', '2019-01-15')
textos

'''

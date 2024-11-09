from joblib import load
import pandas as pd
import numpy as np
import faulthandler

faulthandler.enable()

class FaultDetectionModel:
    REQUIRED_COLUMNS = ['vp1', 'vp2', 'vp3']
    
    def __init__(self, model_path, data_source):
        """
        Inicializa el modelo de detección de fallas. Carga el modelo y los datos, y valida 
        que las columnas necesarias estén presentes en los datos.
        
        :param model_path: Ruta al archivo del modelo preentrenado.
        :param data_source: Puede ser una ruta de archivo CSV, un DataFrame de pandas,
                            un array de datos o un diccionario.
        """
        self.model = load(model_path)
        self.data = self.load_data(data_source)
        self.validate_columns()

    def load_data(self, data_source):
        """
        Carga los datos desde un archivo CSV, un DataFrame de pandas, un array o un diccionario.
        
        :param data_source: Fuente de datos, que puede ser una ruta de archivo CSV, un DataFrame, un array o un diccionario.
        :return: Un DataFrame de pandas con los datos cargados.
        """
        if isinstance(data_source, str):
            # Cargar datos desde un archivo CSV
            data = pd.read_csv(data_source, usecols=self.REQUIRED_COLUMNS)
        elif isinstance(data_source, pd.DataFrame):
            data = data_source
        elif isinstance(data_source, dict):
            data = pd.DataFrame(data_source)
        elif isinstance(data_source, (list, np.ndarray)):
            data = pd.DataFrame(data_source, columns=self.REQUIRED_COLUMNS)
        else:
            raise ValueError("El tipo de fuente de datos no es compatible.")
        
        # Remover valores nulos
        data.dropna(inplace=True)
        return data

    def validate_columns(self):
        """
        Valida que las columnas 'vp1', 'vp2', y 'vp3' estén presentes en los datos.
        """
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Las columnas necesarias están ausentes: {', '.join(missing_columns)}")

    def dynamic_define(self):
        """
        Define los límites de valores para detectar fallas basándose en la media de 'vp1'.
        """
        vp1_mean = self.data['vp1'].mean()
        av1 = int(vp1_mean * 1.09)  # 9% por encima de la media
        av2 = int(vp1_mean * 0.91)  # 9% por debajo de la media
        return av1, av2

    def assign_new_labels(self, label):
        """
        Asigna una etiqueta de falla (1) o normal (0) dependiendo de los límites calculados.
        """
        label1, label2 = self.dynamic_define()
        return 1 if label >= label1 or label <= label2 else 0

    @staticmethod
    def convert_float_to_string(arr):
        """
        Convierte un arreglo de floats a strings sin decimales, rellenándolos con ceros para
        obtener un formato uniforme.
        """
        arr_str = np.char.mod('%d', arr)
        max_len = max(len(s) for s in arr_str)
        arr_str = np.char.zfill(arr_str, max_len)
        return arr_str

    def predict_labels(self):
        """
        Predice las etiquetas combinadas para cada columna (vp1, vp2, vp3) y aplica el modelo.
        """
        y_test = (self.data['vp1'].apply(self.assign_new_labels).astype('str') +
                  self.data['vp2'].apply(self.assign_new_labels).astype('str') +
                  self.data['vp3'].apply(self.assign_new_labels).astype('str'))
        
        # Realizar predicción usando el modelo preentrenado
        y_pred = self.model.predict(np.asarray(self.data))
        
        # Convertir las predicciones a un formato de string uniforme
        y_pred = self.convert_float_to_string(y_pred)
        
        return y_test, y_pred
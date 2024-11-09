# FaultDetectionModel

Esta es una implementación de `FaultDetectionModel`, una clase diseñada para la detección de fallas eléctricas a partir de datos de mediciones. Este modelo permite cargar datos desde un archivo CSV o directamente desde una variable, y utiliza un modelo previamente entrenado para predecir etiquetas de fallas.

## Requisitos previos

- Python 3.x
- Librerías necesarias: `joblib`, `pandas`, `numpy`

## Instalación

Instala las dependencias necesarias utilizando `pip`:

```bash
pip install joblib pandas numpy
```

## Uso

### Crear una instancia de `FaultDetectionModel` usando un archivo CSV

```python
fault_detector_csv = FaultDetectionModel(
    model_path='model.joblib',
    data_source='oficinas.csv'
)
```

### Crear una instancia de `FaultDetectionModel` usando un DataFrame o diccionario de datos

```python
data_dict = {
    'vp1': [1.2, 1.3, 1.5],
    'vp2': [2.3, 2.4, 2.6],
    'vp3': [3.4, 3.5, 3.7]
}
fault_detector_var = FaultDetectionModel(
    model_path='model.joblib',
    data_source=data_dict
)
```

### Ejecutar las predicciones

```python
Y_test_csv, Y_pred_csv = fault_detector_csv.predict_labels()
Y_test_var, Y_pred_var = fault_detector_var.predict_labels()
```

## Descripción de los métodos

- **`predict_labels()`**: Este método procesa los datos de entrada para identificar fallas, asignando etiquetas basadas en los valores de las columnas `vp1`, `vp2` y `vp3`, y utilizando el modelo entrenado para obtener predicciones.

## Manejo de errores

La clase `FaultDetectionModel` verifica que los datos de entrada contengan las columnas necesarias (`vp1`, `vp2`, `vp3`). Si alguna de estas columnas falta, se generará un error indicando al usuario la ausencia de las columnas requeridas.

## Autor

Desarrollado por Audacia.

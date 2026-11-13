# Práctica 6 · AHPSort II (Sistemas de Ayuda a la Decisión)

Clasificación de provincias españolas en tres niveles de seguridad (Alto, Medio, Bajo) mediante
**AHPSort II**, a partir de cinco tipos de hechos delictivos (Crímenes, Robos, Vehículos,
Orden público y Drogas), y representación de la clasificación en mapas.

## Contenido

| Ruta | Descripción |
|---|---|
| `Practica6_AHPSort.ipynb` | Notebook que resuelve los 4 ejercicios |
| `ahpsort.py` | Implementación de AHPSort II (pesos AHP, funciones de prioridad, clasificación, mapas) |
| `datos/` | Datos de criminalidad (Andalucía 2017, España 2017 y 2023) y población |
| `mapas/` | Shapefiles de provincias de Andalucía y de España |

## Ejecución

**Google Colab (recomendado):** abrir el notebook en
https://colab.research.google.com/github/mbvela2002/AHPSort_Practica6/blob/main/Practica6_AHPSort.ipynb
y pulsar
`Entorno de ejecución → Ejecutar todo`. La primera celda clona este repositorio e instala las dependencias.

**En local:**
```bash
pip install -r requirements.txt
jupyter notebook Practica6_AHPSort.ipynb
```

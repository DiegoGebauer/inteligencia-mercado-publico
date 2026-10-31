# Comparación del piloto con los modelos alineados

Se aplican los modelos guardados de 199 y 299 ejemplos. No se reentrena.

El piloto original tenía 1.000 licitaciones de enero-julio. De ellas, 100 entraron al entrenamiento de 299 y siete tienen textos muy parecidos a ejemplos conocidos. Se comparan los 893 casos restantes, con idénticos textos y casos para ambos modelos. De los archivos de validación y test solo se consultan códigos y textos para excluir coincidencias; no se leen sus etiquetas ni se evalúa septiembre.

Los embeddings de los 893 casos se generan una vez y se comparten entre ambos clasificadores. Cada modelo conserva sus propias probabilidades y ordena por afinidad esperada 0–3. Los puntajes entre modelos no se tratan como probabilidades calibradas comparables. El piloto sigue siendo una muestra enriquecida: sus porcentajes no estiman todo el mercado y no tiene etiquetas completas para calcular accuracy. La exclusión de similitud por tokens no garantiza eliminar toda similitud semántica.

## Instalación y ejecución

Extraer el ZIP en la raíz del proyecto conservando las carpetas src, data y docs. El paquete añade un script y un contrato de selección. Conserva el piloto original y los modelos anteriores.

```powershell
cd "D:\Python\Proyectos-Github\inteligencia-mercado-publico"
.\.venv\Scripts\python.exe -m src.nlp.compare_aligned_pilot
```

El contrato verifica los modelos, los datasets y el código usado para representar y ordenar. Si los archivos cambiaron, se requiere preparar una selección actualizada en vez de saltarse las comprobaciones.

## Resultados

En data/processed se generan:

- affinity_aligned_199_pilot_predictions.csv: ranking del modelo de 199.
- affinity_aligned_299_pilot_predictions.csv: ranking del modelo de 299.
- affinity_aligned_pilot_comparison.csv: posiciones y afinidades de ambos sobre los mismos casos.
- affinity_aligned_pilot_review.csv: hasta 40 casos para revisar. Combina los primeros diez de cada modelo, diez desacuerdos y diez contrastes aleatorios sin duplicar códigos. AfinidadRevision queda vacía para revisión humana; no entra al entrenamiento automáticamente.
- affinity_aligned_pilot_manifest.json: selección, exclusiones, hashes y distribución de predicciones.

Al terminar muestra las primeras diez oportunidades de cada modelo. La revisión ayudará a decidir qué ranking es más útil. La evaluación de septiembre corresponde al paso posterior, cuando se cierre la versión elegida.

Se verificaron la selección y el funcionamiento con clasificadores simulados. La clasificación real de los 893 casos queda pendiente de ejecutar el comando.

# Alcance del producto

## Nombre

**Inteligencia de Oportunidades en Mercado Público**

## Problema

La búsqueda manual de licitaciones puede requerir revisar numerosas
oportunidades que no corresponden a los servicios de una empresa.

Este proyecto busca reducir ese trabajo mediante la extracción periódica de
licitaciones desde la API de Mercado Público y la construcción de un ranking
explicable de relevancia.

## Usuario inicial

El usuario inicial será una consultora chilena de datos que ofrece servicios de:

- Inteligencia de negocios y dashboards.
- Power BI y Looker Studio.
- SQL, PostgreSQL y BigQuery.
- Procesos ETL e integración de datos.
- Automatización con Python.
- Analítica predictiva y machine learning.
- Consultoría y capacitación en datos.

## Objetivo del producto

Identificar y priorizar licitaciones cuyos requerimientos sean compatibles con
el perfil, los servicios y la cobertura territorial de la empresa.

Cada oportunidad recibirá un puntaje de relevancia acompañado de una
explicación, por ejemplo:

- Términos relevantes encontrados.
- Servicio empresarial relacionado.
- Categoría del producto o servicio.
- Región del organismo comprador.
- Fecha de cierre.
- Presencia de términos excluidos.

## Entradas

El sistema utilizará inicialmente:

1. Datos públicos obtenidos desde la API de Mercado Público.
2. Perfil empresarial definido en `config/company_profile.yaml`.
3. Nombre y descripción general de cada licitación.
4. Categorías, productos y descripciones de sus ítems.
5. Información del organismo comprador.
6. Fechas y estado de la licitación.

## Salidas esperadas

El sistema deberá generar una tabla ordenada con, al menos:

- Código de la licitación.
- Nombre.
- Organismo comprador.
- Región.
- Fecha de publicación.
- Fecha de cierre.
- Puntaje de relevancia.
- Términos coincidentes.
- Servicio empresarial relacionado.
- Señal de prioridad territorial.
- Razón resumida del puntaje.

## Alcance inicial del MVP

El MVP incluirá:

- Consumo seguro de la API.
- Extracción periódica de licitaciones.
- Consulta del detalle de cada oportunidad.
- Persistencia de datos históricos.
- Limpieza y normalización de textos.
- Perfil empresarial configurable.
- Ranking inicial basado en términos.
- Modelo posterior de similitud semántica.
- Explicaciones para cada puntaje.
- Actualización semanal de oportunidades.
- Evaluación mensual del ranking.

## Fuera del alcance inicial

El MVP no realizará:

- Predicción de la probabilidad de ganar una licitación.
- Preparación o envío automático de ofertas.
- Recomendaciones legales.
- Evaluación completa de bases administrativas.
- Estimación automática de costos de una propuesta.
- Compra automática de documentos o servicios.
- Contacto automático con organismos públicos.
- Reemplazo de la evaluación humana de una oportunidad.

## Cobertura territorial

La cobertura será nacional.

Las oportunidades de las regiones de Los Ríos y Los Lagos recibirán una señal
adicional de prioridad, pero no se excluirán licitaciones provenientes de otras
regiones.

## Criterio de relevancia

La relevancia se estimará combinando progresivamente:

1. Coincidencia con servicios y palabras clave.
2. Similitud entre el texto de la licitación y el perfil empresarial.
3. Ausencia o presencia de servicios excluidos.
4. Prioridad territorial.
5. Vigencia y cercanía de la fecha de cierre.

El resultado será un ranking de compatibilidad comercial, no una estimación de
adjudicación.

## Frecuencia planificada

- Extracción y actualización de oportunidades: semanal.
- Evaluación del funcionamiento del ranking: mensual.
- Reentrenamiento o ajuste: solamente cuando la evaluación lo justifique.

## Criterio inicial de éxito

El MVP se considerará útil si permite:

- Reducir la cantidad de licitaciones que requieren revisión manual.
- Posicionar oportunidades relevantes entre los primeros lugares.
- Explicar de manera comprensible cada recomendación.
- Evitar que oportunidades claramente ajenas dominen el ranking.
- Mantener trazabilidad desde el resultado hasta la información original.
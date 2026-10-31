-- Plantilla usada por load_history.py; referencias de tablas validadas por el cargador.
ASSERT NOT EXISTS (SELECT 1 FROM {lotes} GROUP BY LoteId HAVING COUNT(*) > 1) AS 'Identidad duplicada en lotes';

ASSERT NOT EXISTS (SELECT 1 FROM {capturas} GROUP BY CapturaId HAVING COUNT(*) > 1) AS 'Identidad duplicada en capturas';

ASSERT NOT EXISTS (SELECT 1 FROM {lote_captura} GROUP BY LoteId, CodigoExterno HAVING COUNT(*) > 1) AS 'Identidad duplicada en lote_captura';

ASSERT NOT EXISTS (SELECT 1 FROM {predicciones} GROUP BY PrediccionId HAVING COUNT(*) > 1) AS 'Identidad duplicada en predicciones';

ASSERT NOT EXISTS (SELECT 1 FROM {revisiones} GROUP BY RevisionHistorialId HAVING COUNT(*) > 1) AS 'Identidad duplicada en revisiones';

ASSERT NOT EXISTS (SELECT 1 FROM {predicciones} p LEFT JOIN {capturas} c ON c.CapturaId=p.CapturaId WHERE c.CapturaId IS NULL OR c.CodigoExterno IS DISTINCT FROM p.CodigoExterno) AS 'Prediccion sin captura coincidente';

ASSERT NOT EXISTS (SELECT 1 FROM {lote_captura} lc LEFT JOIN {lotes} l ON l.LoteId=lc.LoteId LEFT JOIN {capturas} c ON c.CapturaId=lc.CapturaId WHERE l.LoteId IS NULL OR c.CapturaId IS NULL OR lc.CodigoExterno IS DISTINCT FROM c.CodigoExterno) AS 'Relacion lote-captura invalida';

ASSERT NOT EXISTS (SELECT 1 FROM {predicciones} p LEFT JOIN {lote_captura} lc ON lc.LoteId=p.LoteId AND lc.CodigoExterno=p.CodigoExterno AND lc.CapturaId=p.CapturaId WHERE lc.CapturaId IS NULL) AS 'Prediccion sin relacion de lote';

ASSERT NOT EXISTS (SELECT 1 FROM {revisiones} r LEFT JOIN {predicciones} p ON p.PrediccionId=r.PrediccionId WHERE p.PrediccionId IS NULL OR r.CapturaId IS DISTINCT FROM p.CapturaId OR r.CodigoExterno IS DISTINCT FROM p.CodigoExterno OR r.LoteId IS DISTINCT FROM p.LoteId OR r.AfinidadPredicha IS DISTINCT FROM p.AfinidadPredicha) AS 'Revision sin prediccion coincidente';

ASSERT NOT EXISTS (SELECT 1 FROM {predicciones} WHERE AfinidadPredicha NOT BETWEEN 0 AND 3 OR AfinidadEsperada NOT BETWEEN 0 AND 3 OR IS_NAN(AfinidadEsperada) OR IS_INF(AfinidadEsperada) OR ABS(ProbAfinidad0+ProbAfinidad1+ProbAfinidad2+ProbAfinidad3-1)>0.00001 OR ABS(AfinidadEsperada-(ProbAfinidad1+2*ProbAfinidad2+3*ProbAfinidad3))>0.00001 OR ProbAfinidad0 NOT BETWEEN 0 AND 1 OR IS_NAN(ProbAfinidad0) OR IS_INF(ProbAfinidad0) OR ProbAfinidad1 NOT BETWEEN 0 AND 1 OR IS_NAN(ProbAfinidad1) OR IS_INF(ProbAfinidad1) OR ProbAfinidad2 NOT BETWEEN 0 AND 1 OR IS_NAN(ProbAfinidad2) OR IS_INF(ProbAfinidad2) OR ProbAfinidad3 NOT BETWEEN 0 AND 1 OR IS_NAN(ProbAfinidad3) OR IS_INF(ProbAfinidad3)) AS 'Clases o probabilidades invalidas';

ASSERT NOT EXISTS (SELECT 1 FROM {revisiones} WHERE AfinidadHumana IS NOT NULL AND AfinidadHumana NOT BETWEEN 0 AND 3) AS 'Afinidad humana invalida';

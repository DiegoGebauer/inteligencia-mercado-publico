"""Coordina captura, prediccion, historial y publicacion opcional en BigQuery."""
import argparse
import json
from pathlib import Path

from src.ingestion.recent_batch import capturar_lote, plan_lote, predecir_lote
from src.reporting.build_batch_history import build_history
from src.warehouse.load_history import apply_history, prepare_history

ROOT = Path(__file__).resolve().parents[1]


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def resolver_lote(lote_id):
    base = (ROOT / "data/raw/batches").resolve()
    carpeta = (base / lote_id).resolve()
    if carpeta.parent != base:
        raise ValueError("El lote debe estar directamente en data/raw/batches.")
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe el lote: {carpeta}")
    manifiesto = leer_json(carpeta / "captura.json")
    if manifiesto["lote_id"] != carpeta.name:
        raise ValueError("La identidad del lote no coincide con su carpeta.")
    return carpeta, manifiesto


def prediccion_existente(lote_id):
    base = ROOT / "data/processed/batches" / lote_id
    resultados = []
    for ruta in base.glob("*/registro_ejecucion.json"):
        registro = leer_json(ruta)
        if registro.get("estado") == "completo":
            csv = ruta.parent / "predicciones.csv"
            if not csv.is_file():
                raise FileNotFoundError(f"Falta el resultado registrado: {csv}")
            resultados.append(csv)
    if len(resultados) > 1:
        raise ValueError("Hay varias predicciones completas para este lote.")
    return resultados[0] if resultados else None


def publicar_historial(historial, args):
    """Reutiliza el cargador transaccional existente; no reintenta una carga."""
    print("\nValidando el historial para BigQuery...")
    plan = prepare_history(historial, root=ROOT)
    print(f"Destino: {args.project}.{args.dataset} | Region: {args.location}")
    informe = apply_history(
        plan,
        args.project,
        args.dataset,
        location=args.location,
        max_bytes=args.max_bytes,
    )
    print("\nPublicacion BigQuery confirmada:")
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return informe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    entrada = parser.add_mutually_exclusive_group(required=True)
    entrada.add_argument("--fecha", help="Fecha de consulta en formato ddmmaaaa.")
    entrada.add_argument("--lote", help="Identificador de un lote local existente.")
    parser.add_argument("--limite", type=int, default=5)
    parser.add_argument(
        "--capturar", action="store_true",
        help="Habilita consultas a Mercado Publico para --fecha.",
    )
    parser.add_argument(
        "--publicar-bigquery", action="store_true",
        help="Publica el historial acumulativo en BigQuery al terminar.",
    )
    parser.add_argument("--project", default="inteligencia-mercado-publico")
    parser.add_argument("--dataset", default="mercado_publico")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--max-bytes", type=int, default=1_073_741_824)
    args = parser.parse_args(argv)

    if args.capturar and not args.fecha:
        parser.error("--capturar requiere --fecha.")
    if args.max_bytes <= 0:
        parser.error("--max-bytes debe ser positivo.")

    if args.fecha:
        plan = plan_lote(args.fecha, args.limite)
        plan["publicar_bigquery_al_ejecutar"] = args.publicar_bigquery
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        if not args.capturar:
            print("\nSolo plan: agrega --capturar para ejecutar las etapas.")
            return 0
    else:
        carpeta, manifiesto = resolver_lote(args.lote)
        if manifiesto["estado"] not in {"completo", "vacio"}:
            raise ValueError(
                "El lote esta incompleto o fallido. Debe recuperarse "
                "antes de continuar con la prediccion."
            )

    # Este control detiene entradas historicas invalidas antes de consultar la API.
    print("\nVerificando las entradas del historial...")
    historial = build_history(root=ROOT)
    hubo_cambios = False

    if args.fecha:
        carpeta = capturar_lote(args.fecha, args.limite, raiz=ROOT)
        carpeta, manifiesto = resolver_lote(carpeta.name)
        hubo_cambios = True

    if manifiesto["estado"] == "vacio":
        print("\nEl lote esta vacio; se conserva en el historial.")
    else:
        resultado = prediccion_existente(carpeta.name)
        if resultado is not None:
            print("\nReutilizando prediccion completa:", resultado)
        else:
            print("\nCalculando la prediccion del lote...")
            resultado = predecir_lote(carpeta, raiz=ROOT)
            hubo_cambios = True
            print("Prediccion guardada:", resultado)

    if hubo_cambios:
        print("\nActualizando el historial...")
        historial = build_history(root=ROOT)
    print("Historial verificado:", historial)

    # Tanto las predicciones nuevas como las reutilizadas llegan a esta etapa.
    if args.publicar_bigquery:
        try:
            publicar_historial(historial, args)
        except Exception:
            print(
                "\nLa publicacion no quedo confirmada al cliente. Conserva el "
                "ID del trabajo y revisa su estado en BigQuery antes de repetir."
            )
            print(
                "Las capturas y predicciones locales se conservan. "
                f"El lote para continuar es: {carpeta.name}"
            )
            raise

    print("\nPipeline terminado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Fusiona los zips de custom_dataset_flow/input_zips/ en custom_dataset_flow/output_unified/

#!/usr/bin/env python3
import json
import math
import sys
import pandas as pd
import os
import io
import zipfile
import argparse
from pathlib import Path

# --- Configuración desde your_config.json (ya provista por ti) ---
with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

zips_files_path = config_datos['custom_dataset_flow_path'] + 'input_zips/'
output_files_path = config_datos['custom_dataset_flow_path'] + 'output_unified/'

# -----------------------------------------------------------------

def unique_name(name, existing, zip_tag):
  """
  Genera un nombre único si 'name' ya existe en el set existing.
  Añade sufijo con zip_tag y contador: base__ziptag__1.ext
  """
  base, ext = os.path.splitext(name)
  candidate = name
  i = 1
  while candidate in existing:
    candidate = f"{base}__{zip_tag}__{i}{ext}"
    i += 1
  existing.add(candidate)
  return candidate

def try_read_csv_bytes(bytes_io):
  """
  Intenta leer un CSV desde un bytes buffer con diferentes encodings.
  Retorna dataframe o lanza excepción.
  """
  for enc in ("utf-8", "latin-1"):
    try:
      bytes_io.seek(0)
      return pd.read_csv(io.TextIOWrapper(bytes_io, encoding=enc))
    except Exception:
      continue
  # si falla, intenta con pandas autodetectando (último recurso)
  bytes_io.seek(0)
  return pd.read_csv(io.BytesIO(bytes_io.read()))

def build_replacement_variants(orig_basename):
  """
  Devuelve una lista de variantes que pueden aparecer en el CSV para referirse al audio.
  Ej: 'a.wav', 'clips/a.wav', './clips/a.wav', con separadores backslash y slash.
  """
  variants = set()
  variants.add(orig_basename)
  variants.add(f"clips/{orig_basename}")
  variants.add(f"./clips/{orig_basename}")
  variants.add(f"clips\\{orig_basename}")
  variants.add(f".\\clips\\{orig_basename}")
  return sorted(variants, key=len, reverse=True)  # ordenar por longitud para reemplazos seguros

def update_dataframe_names(df, mapping):
  """
  Reemplaza en todas las celdas string las apariciones de nombres originales por los nuevos.
  mapping: dict{orig_basename: new_basename}
  """
  if df is None or not mapping:
    return df

  # Precrear un mapa de reemplazo con variantes
  full_map = {}
  for orig, new in mapping.items():
    for var in build_replacement_variants(orig):
      # mapea cada variante a 'clips/<new>' si la variante contiene 'clips/' o a new si solo es basename
      if var.startswith("clips/") or var.startswith("./clips/") or var.startswith("clips\\") or var.startswith(".\\clips\\"):
        full_map[var] = f"clips/{new}"
      else:
        full_map[var] = new

  # Reemplazo en todas las celdas string
  def replace_cell(x):
    if not isinstance(x, str):
      return x
    s = x
    # reemplazar por mayor a menor longitud de clave (para evitar problemas con substrings)
    for k in sorted(full_map.keys(), key=len, reverse=True):
      if k in s:
        s = s.replace(k, full_map[k])
    return s

  return df.applymap(replace_cell)

def read_train_csv_from_zip(z, train_name):
  """
  Lee train.csv desde zipfile.ZipFile z en la ruta train_name.
  Devuelve DataFrame o None si no pudo.
  """
  try:
    with z.open(train_name) as f:
      # leemos bytes y usamos helper para robustez de encoding
      raw = io.BytesIO(f.read())
      raw.seek(0)
      # intentar con pandas detectando encoding
      try:
        return try_read_csv_bytes(raw)
      except Exception as e:
        print(f"    Error leyendo {train_name}: {e}")
        return None
  except Exception as e:
    print(f"    Error abriendo {train_name} en zip: {e}")
    return None

def main():
    input_dir = Path(zips_files_path)
    output_dir = Path(output_files_path)
    clips_out = output_dir / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)

    # existing = nombres de archivos ya presentes en clips_out
    existing = set(p.name for p in clips_out.iterdir() if p.is_file())

    dfs = []
    zip_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".zip"])
    if not zip_files:
        print("No se encontraron archivos .zip en", input_dir)
        return

    for zpath in zip_files:
        zip_tag = zpath.stem.replace(" ", "_")
        print(f"Procesando {zpath.name} ...")
        mapping = {}  # orig_basename -> new_basename para este zip
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                names = z.namelist()
                # localizar train.csv (caso-insensible)
                train_candidates = [n for n in names if n.lower().endswith("train.csv")]
                train_df = None
                if train_candidates:
                    train_name = train_candidates[0]
                    train_df = read_train_csv_from_zip(z, train_name)
                else:
                    print(f"  Aviso: {zpath.name} no tiene train.csv; se saltará el CSV de este zip.")

                # extraer archivos dentro de clips/
                for member in names:
                    # normalizar separadores a '/'
                    member_norm = member.replace("\\", "/")
                    if member_norm.startswith("clips/") and not member_norm.endswith("/"):
                        orig_basename = os.path.basename(member_norm)
                        new_basename = unique_name(orig_basename, existing, zip_tag)
                        mapping[orig_basename] = new_basename
                        # extraer y escribir
                        try:
                            with z.open(member) as src:
                                target_path = clips_out / new_basename
                                with open(target_path, "wb") as out_f:
                                    out_f.write(src.read())
                        except Exception as e:
                            print(f"    Error extrayendo {member_norm} de {zpath.name}: {e}")

                # actualizar train_df si existe
                if train_df is not None:
                    train_df = update_dataframe_names(train_df, mapping)
                    dfs.append(train_df)
                else:
                    # nada que añadir
                    pass
        except zipfile.BadZipFile:
            print(f"  ERROR: {zpath.name} no es un zip válido o está corrupto. Se salta.")
        except Exception as e:
            print(f"  ERROR procesando {zpath.name}: {e}")

    # concatenar DataFrames (mantener sólo una cabecera)
    if dfs:
        try:
            df_all = pd.concat(dfs, ignore_index=True, sort=False)
            out_csv = output_dir / "train.csv"
            output_dir.mkdir(parents=True, exist_ok=True)
            df_all.to_csv(out_csv, index=False, encoding="utf-8")
            print("Hecho. Archivo concatenado escrito en:", out_csv)
            print("Clips extraídos en:", clips_out)
        except Exception as e:
            print("Error concatenando/escribiendo CSV final:", e)
    else:
        print("No se concatenaron CSVs (no se encontraron train.csv válidos).")

if __name__ == "__main__":
    main()
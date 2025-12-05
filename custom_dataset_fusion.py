# Ejecutar después de custom_dataset_flow.py
# Fusiona custom_dataset_flow/output_unified/ con custom_dataset/

#!/usr/bin/env python3
import json
import io
import os
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime

# Cargar configuración
with open('your_config.json', 'r') as f:
  config_datos = json.load(f)

flow_output_dir = Path(config_datos['custom_dataset_flow_path']) / 'output_unified'
dataset_dir = Path(config_datos['custom_dataset_path'])
clips_flow_dir = flow_output_dir / 'clips'
clips_dataset_dir = dataset_dir / 'clips'

def unique_name(name, existing, src_tag):
  base, ext = os.path.splitext(name)
  candidate = name
  i = 1
  while candidate in existing:
    candidate = f"{base}__{src_tag}__{i}{ext}"
    i += 1
  existing.add(candidate)
  return candidate

def build_replacement_variants(orig_basename):
  variants = set()
  variants.add(orig_basename)
  variants.add(f"clips/{orig_basename}")
  variants.add(f"./clips/{orig_basename}")
  variants.add(f"clips\\{orig_basename}")
  variants.add(f".\\clips\\{orig_basename}")
  return sorted(variants, key=len, reverse=True)

def update_dataframe_names(df, mapping):
  if df is None or not mapping:
    return df
  full_map = {}
  for orig, new in mapping.items():
    for var in build_replacement_variants(orig):
      if var.startswith("clips/") or var.startswith("./clips/") or var.startswith("clips\\") or var.startswith(".\\clips\\"):
        full_map[var] = f"clips/{new}"
      else:
        full_map[var] = new
  def replace_cell(x):
    if not isinstance(x, str):
      return x
    s = x
    for k in sorted(full_map.keys(), key=len, reverse=True):
      if k in s:
        s = s.replace(k, full_map[k])
    return s
  return df.applymap(replace_cell)

def try_read_csv_file(path):
  for enc in ("utf-8", "latin-1"):
    try:
      return pd.read_csv(path, encoding=enc)
    except Exception:
      continue
  # último recurso
  return pd.read_csv(path)

def find_train_csv(dir_path):
  if not dir_path.exists():
    return None
  candidates = [p for p in dir_path.iterdir() if p.is_file() and p.name.lower() == 'train.csv']
  if candidates:
    return candidates[0]
  alt = list(dir_path.rglob('*train.csv'))
  return alt[0] if alt else None

def main():
  # Validaciones básicas
  if not flow_output_dir.exists():
    print("Error: no existe", flow_output_dir)
    return
  dataset_dir.mkdir(parents=True, exist_ok=True)
  clips_dataset_dir.mkdir(parents=True, exist_ok=True)

  # Conjunto de nombres ya existentes en dataset/clips
  existing = set(p.name for p in clips_dataset_dir.iterdir() if p.is_file())

  # 1) Leer train.csv existente en dataset (si existe) y hacer backup
  dataset_train_path = find_train_csv(dataset_dir)
  df_dataset = None
  if dataset_train_path:
    try:
      df_dataset = try_read_csv_file(dataset_train_path)
      # backup
      ts = datetime.now().strftime('%Y%m%d_%H%M%S')
      backup_path = dataset_dir / f"train.csv.bak.{ts}"
      shutil.copy2(dataset_train_path, backup_path)
      print(f"Backup creado de train.csv original en: {backup_path}")
    except Exception as e:
      print("Aviso: no se pudo leer o respaldar train.csv del dataset:", e)
      df_dataset = None
  else:
    print("No se encontró train.csv en el dataset destino (se creará uno nuevo).")

  # 2) Procesar clips del flujo: copiar a dataset/clips y construir mapping
  mapping = {}  # orig_basename -> new_basename
  if clips_flow_dir.exists() and clips_flow_dir.is_dir():
    for root, _, files in os.walk(clips_flow_dir):
      for fname in files:
        orig_basename = os.path.basename(fname)
        new_basename = unique_name(orig_basename, existing, 'flow')
        mapping[orig_basename] = new_basename
        src_path = Path(root) / fname
        dst_path = clips_dataset_dir / new_basename
        try:
          shutil.copy2(src_path, dst_path)
        except Exception as e:
          print(f"  Error copiando {src_path} -> {dst_path}: {e}")
    print(f"Copiados clips del flujo a {clips_dataset_dir} (colisiones resueltas).")
  else:
    print(f"No se encontró carpeta clips/ en {flow_output_dir}; no hay archivos de audio para copiar.")

  # 3) Leer train.csv del flujo (si existe), actualizar referencias según mapping
  flow_train_path = find_train_csv(flow_output_dir)
  df_flow = None
  if flow_train_path:
    try:
      df_flow = try_read_csv_file(flow_train_path)
      df_flow = update_dataframe_names(df_flow, mapping)
      print("Leído y actualizado train.csv del flujo:", flow_train_path)
    except Exception as e:
      print("Aviso: no se pudo leer/actualizar train.csv del flujo:", e)
      df_flow = None
  else:
    print("No se encontró train.csv en output_unified; no hay filas nuevas desde el flujo.")

  # 4) Concatenar dataframes: primero dataset existente (si hay), luego flow
  dfs = []
  if df_dataset is not None:
    dfs.append(df_dataset)
  if df_flow is not None:
    dfs.append(df_flow)

  if dfs:
    try:
      df_all = pd.concat(dfs, ignore_index=True, sort=False)
      out_csv = dataset_dir / 'train.csv'
      df_all.to_csv(out_csv, index=False, encoding='utf-8')
      print("CSV final escrito en:", out_csv)
      print("Fusión completada correctamente.")
    except Exception as e:
      print("Error concatenando o escribiendo el CSV final:", e)
  else:
    print("No hay CSVs válidos para concatenar. No se escribió train.csv.")

if __name__ == '__main__':
  main()
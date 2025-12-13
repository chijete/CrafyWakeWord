# Antes de ejecutarlo todos los clips deben estar en custom_dataset/train.csv, ninguno en test y dev.csv
# Mezcla algunos clips de custom_dataset/train.csv y los pasa a custom_dataset/test.csv y custom_dataset/dev.csv

#!/usr/bin/env python3
import json
import io
import os
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime

def asegurar_vacio_o_crear(ruta):
  p = Path(ruta)
  # crear directorio padre si hace falta
  p.parent.mkdir(parents=True, exist_ok=True)

  if not p.exists():
    # no existe -> crear archivo vacío
    p.touch()
    return

  # existe -> comprobar tamaño
  if p.stat().st_size == 0:
    # existe y está vacío -> continuar
    return

  # existe y tiene contenido -> error
  raise FileExistsError(f"El archivo '{p}' ya existe y no está vacío.")

asegurar_vacio_o_crear("custom_dataset/test.csv")
asegurar_vacio_o_crear("custom_dataset/dev.csv")

train_df = pd.read_csv("custom_dataset/train.csv")

total = len(train_df)
n10 = int(round(total * 0.10))
n5  = int(round(total * 0.05))

df_10 = train_df.sample(n=n10)
resto = train_df.drop(df_10.index)
df_5 = resto.sample(n=n5)
df_restante = train_df.drop(df_10.index).drop(df_5.index)

df_10 = df_10.reset_index(drop=True)
df_5 = df_5.reset_index(drop=True)
df_restante = df_restante.reset_index(drop=True)

df_restante.to_csv("custom_dataset/train.csv", index=False)
df_10.to_csv("custom_dataset/dev.csv", index=False)
df_5.to_csv("custom_dataset/test.csv", index=False)
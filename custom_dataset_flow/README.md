Folders:
input_zips/
  zip files
output_unified/
  result

Cuando hay nuevos zips con audios custom:
1. Borrar los zips de custom_dataset_flow/input_zips/ y custom_dataset_flow/output_unified/
2. Poner los nuevos zips en custom_dataset_flow/input_zips/
3. Ejecutar python custom_dataset_flow.py
4. Ejecutar python custom_dataset_fusion.py
5. Listo, los nuevos audios custom ya están en custom_dataset/
6. Ejecutar python custom_dataset_mix.py
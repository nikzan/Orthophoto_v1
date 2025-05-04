import rasterio
import numpy as np

def read_dem_data(file_path):
    """Читает данные DEM из GeoTIFF файла."""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Читаем первый банд (обычно высота)
            transform = src.transform
            crs = src.crs
            profile = src.profile #Метаданные GeoTiff
            return data, transform, crs, profile
    except rasterio.RasterioIOError as e:
        print(f"Ошибка при чтении DEM: {e}")
        return None, None, None, None

# Пример использования
file_path = "DEMColorado.tif"  # Замените на путь к вашему файлу
data, transform, crs, profile = read_dem_data(file_path)

if data is not None:
    print(f"Данные DEM прочитаны.  Размер: {data.shape}, CRS: {crs}")
    # Здесь можно выполнять дальнейшую обработку данных
else:
    print("Не удалось прочитать данные DEM.")

#Для записи обработанных данных:
def write_modified_dem(output_path, data, transform, crs, profile):
    """
    Записывает обработанные данные DEM в новый GeoTIFF файл.

    Args:
        output_path: Путь для сохранения нового GeoTIFF файла.
        data: Массив NumPy с данными DEM.
        transform: Объект Affine transform из rasterio.
        crs: Объект CRS из rasterio.
        profile: Профиль GeoTIFF (метаданные) из rasterio.
    """
    try:
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            **profile
        ) as dst:
            dst.write(data, 1)
        print(f"Файл успешно записан: {output_path}")
    except rasterio.RasterioIOError as e:
        print(f"Ошибка при записи файла: {e}")

# Пример использования функции write_modified_dem
#Предположим, что вы выполнили какие-то операции с данными (например, обрезали область)
#output_path = "output.tif"
#write_modified_dem(output_path, data, transform, crs, profile)

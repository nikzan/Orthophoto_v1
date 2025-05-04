import exifread

def extract_gps_coordinates(image_path):
    """
    Извлекает широту и долготу из EXIF-метаданных JPEG-изображения.
    
    Параметры:
        image_path (str): Путь к файлу изображения
        
    Возвращает:
        tuple: (широта, долгота) в десятичных градусах или None, если данные отсутствуют
    """
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    
    def _convert_to_decimal(gps_value, ref):
        """Конвертирует значение из формата DMS в десятичные градусы"""
        degrees = float(gps_value.values[0].num) / gps_value.values[0].den
        minutes = float(gps_value.values[1].num) / gps_value.values[1].den
        seconds = float(gps_value.values[2].num) / gps_value.values[2].den
        decimal = degrees + (minutes / 60) + (seconds / 3600)
        return -decimal if ref in ['S', 'W'] else decimal

    try:
        # Извлечение широты
        gps_latitude = tags.get('GPS GPSLatitude')
        gps_latitude_ref = tags.get('GPS GPSLatitudeRef').values
        latitude = _convert_to_decimal(gps_latitude, gps_latitude_ref)

        # Извлечение долготы
        gps_longitude = tags.get('GPS GPSLongitude')
        gps_longitude_ref = tags.get('GPS GPSLongitudeRef').values
        longitude = _convert_to_decimal(gps_longitude, gps_longitude_ref)

        return (latitude, longitude)
    
    except (AttributeError, KeyError):
        return None

# Пример использования
coordinates = extract_gps_coordinates("DJI_0005.jpg")
if coordinates:
    print(f"Широта: {coordinates[0]}, Долгота: {coordinates[1]}")
else:
    print("GPS-данные не найдены")
# Техническая инструкция по запуску проекта ЭЭГ

## 1. Требования

* Python 3.9 или выше
* Установленные зависимости:

```bash
pip install -r requirements.txt
```

Рекомендуется использовать виртуальное окружение.

---

## 2. Структура проекта

```
project_root/
├── raw_data/
│   ├── <ID>_RAW DATA.csv
│   ├── <ID>_intervalMarker.csv
│   └── <ID>_Восприятие_инфоповодов_basic_pre_processing.csv  # для пилота
├── erp_results/               # ERP-графики PNG
├── emotions_results/          # Графики эмоциональных метрик PNG
├── Preprocessing_Results.xlsx # Таблица предобработки
├── task_1.py                  # ERP-анализ
├── task_2.py                  # Эмоциональные метрики
├── config.py                  # Настройки ERP
├── config_emotions.py         # Настройки эмоций
├── data_loader.py             # Чтение CSV, синхронизация
└── handlers.py                # Извлечение сигналов и маркеров
```

---

## 3. Запуск ERP-анализа (Задача 1)

```bash
python task_1.py
```

* Создаёт 24 ERP-графика по каждому стимулу
* Сохраняет результаты в `erp_results/`
* Создаёт таблицу предобработки `Preprocessing_Results.xlsx`, содержащую:

  * показатели качества (чистоты) каналов EEG;
  * количество отбракованных эпох;
  * количество эпизодов (эпох) для каждого предъявленного стимула;
  * дополнительные метрики синхронизации и корректности данных.

---

## 4. Запуск анализа эмоциональных метрик (Задача 2)

```bash
python task_2.py
```

* Генерирует графики метрик: stress, interest, engagement, excitement, focus, relaxation
* Сохраняет в `emotions_results/`
* При наличии информации о поле строит отдельные графики по мужскому/женскому полу

---

## 5. Особенности работы

* Скрипты автоматически проверяют целостность CSV-файлов и корректность колонок (`timestamp`, `marker_label__desc`, EEG-каналы)
* Если число доступных эпох меньше 10, графики не строятся, информация записывается в Excel


* GitHub репозиторий с кодом: [https://github.com/nadyashanarova/eeg-analysis.git](https://github.com/nadyashanarova/eeg-analysis.git)
* XLSX с предобработкой и статистикой находится в корне проекта (`Preprocessing_Results.xlsx`)
* Графики ERP и эмоциональных метрик находятся в папках `erp_results/` и `emotions_results/`

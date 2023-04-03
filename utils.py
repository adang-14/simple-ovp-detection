def get_category_frequencies(diagnosis_codes):
    category_counts = {}
    for code in diagnosis_codes:
        category = code[0]
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
    return category_counts

def feature_engineering(row):
    diagnosis_codes = row['diagnosis_codes']
    category_frequencies = get_category_frequencies(diagnosis_codes)

    row['total_codes'] = len(diagnosis_codes)
    row['unique_categories'] = len(category_frequencies)

    # Add category frequencies as new columns
    for category, count in category_frequencies.items():
        column_name = f'category_{category}_frequency'
        row[column_name] = count

    return row
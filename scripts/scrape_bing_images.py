import os
from bing_image_downloader import downloader

queries = {
    "car": "car street daytime",
    "bicycle": "bicycle city road",
    "scooter": "electric kick scooter street",
    "bus": "bus city traffic"
}

for label, query in queries.items():
    downloader.download(
        query=query,
        limit=100,
        output_dir='dataset',
        adult_filter_off=True,
        timeout=60,
        verbose=True
    )

    src = os.path.join('dataset', query)
    dst = os.path.join('dataset', label)
    os.makedirs(dst, exist_ok=True)

    for fname in os.listdir(src):
        os.replace(os.path.join(src, fname),
                   os.path.join(dst, fname))

    os.rmdir(src)
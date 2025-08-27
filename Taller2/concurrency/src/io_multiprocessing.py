import requests
import multiprocessing
import os

session = None


def set_global_session():
    global session
    if not session:
        session = requests.Session()


def download_site(args):

    number, url = args

    os.makedirs("data", exist_ok=True)

    with session.get(url) as response:
        name = multiprocessing.current_process().name

        file_name = os.path.join("data", f"image{number}.jpg")    
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=100):
                f.write(chunk)
        print(f"{name}: download {len(response.content)} from {url} in {file_name}")


def multiprocessing_method(sites):

    sites_number = [(idx, url)  for idx, url  in enumerate (sites, start = 1)]

    with multiprocessing.Pool(initializer=set_global_session) as pool:
        pool.map(download_site, sites_number)


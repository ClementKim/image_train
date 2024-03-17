import time
import requests
import urllib.request

from os import path, mkdir, listdir

from openpyxl import load_workbook

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

def internet_connection():
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def image_check(home_dir):
    download_or_not = True

    file_path = path.join(home_dir, "image to download.xlsx")
    wb = load_workbook(file_path, read_only=True)
    ws = wb['Sheet1']

    query = []
    for row in ws.rows:
        for cell in row:
            query.append(cell.value)

    query = list(set(query))

    if not "img" in listdir(home_dir):
        mkdir("img")

    img_dir_list = listdir(path.join(home_dir, "img"))

    target = []
    for target_name in img_dir_list:
        if target_name == ".DS_Store":
            continue

        target.append(target_name[:-7])

    target = list(set(target))

    if all([True if query_name in target else False for query_name in query]):
        download_or_not = False

    if download_or_not:
        download_target = [query_name for query_name in query if not query_name in target]
        print("Downloading start")
        image_download(home_dir, download_target)

    else:
        print("Nothing to download")

def image_download(home_dir, download_target):
    if not internet_connection():
        print("The internet is not connected")
        exit(1)

    img_dir = path.join(home_dir, "img")

    driver_dir = path.join(home_dir, "chrome driver")

    for target in download_target:
        driver = webdriver.Chrome()
        chrome_options = webdriver.ChromeOptions()
        chrome_options.binary = driver_dir

        driver.get("https://www.google.com/imghp")
        search_bar = driver.find_element(By.NAME, "q")

        search_bar.send_keys(target)
        search_bar.submit()

        pause_time = 1

        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            driver.execute_script("window.scrollBy(0, 5000)")
            time.sleep(pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")

            if new_height - last_height > 0:
                last_height = new_height
                continue
            else:
                break

        img_elements = driver.find_elements(By.CLASS_NAME, 'rg_i.Q4LuWd')

        imgs = []
        for idx, img in enumerate(img_elements):
            print(f'{target}{idx + 1} : {round((idx + 1) / len(img_elements) * 100, 2)}%')
            try:
                img_src = img.get_attribute('src')
                img_alt = img.get_attribute('alt')
                imgs.append({
                    'alt': img_alt,
                    'src': img_src
                })
            except NoSuchElementException:
                pass

            except Exception as e:
                print(f'error in {idx + 1}')
                print(e)
        driver.close()

        for idx, one in enumerate(imgs):
            src = one['src']
            alt = one['alt']

            if not src or str(src).startswith("https://encrypted"):
                continue

            try:
                time.sleep(pause_time)
                if idx < 10:
                    urllib.request.urlretrieve(src, path.join(img_dir, f"{target}_0{idx}.png"))
                else:
                    urllib.request.urlretrieve(src, path.join(img_dir, f"{target}_{idx}.png"))
                print(idx, alt)
            except Exception as e:
                print(f"Error downloading image {idx + 1}: {e}")

    print("Image downloading is done")

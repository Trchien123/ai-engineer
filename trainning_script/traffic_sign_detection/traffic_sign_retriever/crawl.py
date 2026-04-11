import requests
from bs4 import BeautifulSoup
import json
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import io

BASE_URL = "https://www.transport.nsw.gov.au"
START_URL = "https://www.transport.nsw.gov.au/operations/roads-and-waterways/traffic-signs"
CATEGORIES = {
    "Bus stop": f"{START_URL}?f%5B0%5D=road_traffic_sign_type%3A2061",
    "Delineation": f"{START_URL}?f%5B0%5D=road_traffic_sign_type%3A2062",
    "Express": f"{START_URL}?f%5B0%5D=road_traffic_sign_type%3A2063",
    "Guide": f"{START_URL}?f%5B0%5D=road_traffic_sign_type%3A2064",
    "Regulatory": f"{START_URL}?f%5B0%5D=road_traffic_sign_type%3A2065",
    "Temporary": f"{START_URL}?f%5B0%5D=road_traffic_sign_type%3A2066",
    "Warning": f"{START_URL}?f%5B0%5D=road_traffic_sign_type%3A2067",
}
IMAGE_ROOT = "downloaded_images"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

def download_image(url, sign_no, category_name):
    try:
        cat_folder = os.path.join(IMAGE_ROOT, category_name.replace(" ", "_"))
        if not os.path.exists(cat_folder):
            os.makedirs(cat_folder)

        # Clean the sign name for the filename
        clean_name = "".join([c for c in sign_no if c.isalnum() or c in (' ', '-', '_')]).strip().replace(" ", "_")
        filename = f"{clean_name}.jpg"
        filepath = os.path.join(cat_folder, filename)

        # 2. Get the image data
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Load the image from the response content
            img = Image.open(io.BytesIO(response.content))
            
            # 3. Convert to RGB 
            # (GIFs are 'P' mode; JPEG requires 'RGB'. This also removes transparency)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(filepath, "JPEG", quality=95)
            return filepath 
            
    except Exception as e:
        print(f"      Error converting/downloading image {sign_no}: {e}")
    
    return url

def scrape_sign_details(detail_url, category_name):
    response = requests.get(detail_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    print(f"Status Code: {response.status_code}") # Should be 200    
    data = {"category": category_name}
    
    # 1. Scrape the Table
    h2_details = soup.find('h2', id='Sign_Details')
    table = h2_details.find_next('table') if h2_details else soup.find('table')

    if table:
        for row in table.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 2:
                # Clean up the key (remove ':', bold tags, and extra spaces)
                key = cols[0].get_text(strip=True).replace(':', '')
                # Get text from the second column, joining multiple paragraphs if they exist
                value = cols[1].get_text(" ", strip=True) 
                data[key] = value
    else:
        print(f"Warning: No table found at {detail_url}")
    
    # 2. Scrape the Sign Image
    # We look for the image inside the 'picture' tag or just the main content img
    img_tag = soup.find('img', loading='eager') or soup.select_one('picture img')
    if img_tag and 'src' in img_tag.attrs: 
        web_url = img_tag['src'] if img_tag['src'].startswith('http') else BASE_URL + img_tag['src']
        
        # We use the 'Sign No' or 'name' for the filename
        filename_base = data.get("Sign No", "unknown_sign")
        data['local_image_path'] = download_image(web_url, filename_base, category_name)
        data['original_url'] = web_url
    
    return data

def main():
    all_signs = []
    for cat_name, cat_url in CATEGORIES.items():
        print(f"\n--- Starting Category: {cat_name} ---")
        page = 0
        while True:
            url = f"{cat_url}&page={page}"
            print(f"Scraping {cat_name} - Page {page}...")

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to load page {page}. Status: {response.status_code}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            sign_links = soup.select('.featured__item h3 a')
            
            if not sign_links:
                print("No more results found.")
                break
                
            for link in sign_links:
                full_link = BASE_URL + link['href']
                name = link.get_text(strip=True)
                print(f"  - Extracting: {name}")
                try:
                    details = scrape_sign_details(full_link, cat_name)
                    details["name"] = name
                    all_signs.append(details)
                    # Polite delay to avoid overwhelming the server
                    time.sleep(0.7) 
                except Exception as e:
                    print(f"Error scraping {name}: {e}")
            page += 1
            # if page > 0: break 

    # Save to JSON
    print(f"\nScraping complete. Total signs collected: {len(all_signs)}")
    with open('nsw_traffic_signs.json', 'w', encoding='utf-8') as f:
        json.dump(all_signs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
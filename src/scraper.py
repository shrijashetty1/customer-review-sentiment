
import time
import pandas as pd
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class GoogleMapsScraper:
    def __init__(self, chromedriver_path):
        self.chromedriver_path = chromedriver_path
        self.driver = self._init_driver()
        
    def _init_driver(self):
        service = Service(self.chromedriver_path)
        chrome_options = Options()
        chrome_options.add_argument("--disable-images")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver

    def open_page(self, url):
        self.driver.get(url)
    
    def accept_cookies(self):
        try:
            accept_button = self.driver.find_element(By.XPATH, "//button[@aria-label='Aceptar todo']")
            accept_button.click()
            print("Clicked 'Aceptar todo'")
            time.sleep(5)
        except Exception:
            print("No 'Aceptar todo' button found")
            pass

    def extract_reviews_summary(self):
        wait = WebDriverWait(self.driver, 30)
        reviews_container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde')))
        raw_html_reviews = reviews_container.find_elements(By.CSS_SELECTOR, 'tr.BHOKXe')
        
        ratings = []
        reviews_counts = []
        
        for star in raw_html_reviews:
            rating_text = star.get_attribute('aria-label')
            rating_parts = rating_text.split(',')
            stars = rating_parts[0].split()[0]
            num_reviews = rating_parts[1].strip().split()[0].replace('.', '')
            ratings.append(int(stars))
            reviews_counts.append(int(num_reviews))
        print('OK! -> summary extracted')
        return pd.DataFrame({'stars': ratings, 'reviews': reviews_counts}), reviews_container
    
    def extract_visible_reviews(self, reviews_container, scroll_pause_time=120):
        reviewers = []
        ratings = []
        review_texts = []
        review_dates = []
        local_guides = []
        text_backups = []
        processed_review_ids = set()

        last_height = self.driver.execute_script("return arguments[0].scrollHeight", reviews_container)

        while True:
            raw_html_reviews = self.driver.find_elements(By.CSS_SELECTOR, 'div.jftiEf.fontBodyMedium')
            
            for review in raw_html_reviews:
                try:
                    review_id = review.get_attribute('data-review-id')
                    
                    if review_id in processed_review_ids:
                        continue  # Skip already processed reviews
                    
                    processed_review_ids.add(review_id)

                    # Expand the review text if a 'More' button exists
                    try:
                        more_buttons = review.find_elements(By.CSS_SELECTOR, 'button.w8nwRe')
                        for button in more_buttons:
                            if button.is_displayed():
                                button.click()
                                time.sleep(1)
                    except Exception:
                        pass

                    # Extract the details of the review
                    reviewer = review.find_element(By.CSS_SELECTOR, 'div.d4r55').text if review.find_element(By.CSS_SELECTOR, 'div.d4r55') else ''
                    review_text = review.find_element(By.CSS_SELECTOR, 'span.wiI7pd').text if review.find_element(By.CSS_SELECTOR, 'span.wiI7pd') else ''
                    rating = review.find_element(By.CSS_SELECTOR, 'span.kvMYJc').get_attribute('aria-label') if review.find_element(By.CSS_SELECTOR, 'span.kvMYJc') else ''
                    review_date = review.find_element(By.CSS_SELECTOR, 'span.rsqaWe').text if review.find_element(By.CSS_SELECTOR, 'span.rsqaWe') else ''
                    local_guide = review.find_element(By.CLASS_NAME, "RfnDt").text if review.find_element(By.CLASS_NAME, "RfnDt") else ''

                    # Append the data
                    reviewers.append(reviewer)
                    review_texts.append(review_text)
                    ratings.append(rating)
                    review_dates.append(review_date)
                    local_guides.append(local_guide)
                    text_backups.append(review.text)
                    print(rating, '-', review_text)
                except Exception as e:
                    print(f"Error extracting review: {str(e)}")

            # Scroll down the page
            self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", reviews_container)
            time.sleep(10)

            new_height = self.driver.execute_script("return arguments[0].scrollHeight", reviews_container)
            
            if new_height == last_height:
                print('No more reviews detected, forcing additional scroll')
                time.sleep(scroll_pause_time)
                new_raw_html_reviews = self.driver.find_elements(By.CSS_SELECTOR, 'div.jftiEf.fontBodyMedium')
                if len(new_raw_html_reviews) == len(raw_html_reviews):
                    print('No more reviews loaded, exiting scroll')
                    break
            else:
                last_height = new_height

        collected_data = pd.DataFrame({
            'author': reviewers,
            'local_guide_info': local_guides,
            'rating': ratings,
            'review': review_texts,
            'date_text': review_dates,
            'text_backup': text_backups
        })

        return collected_data
    
    def close(self):
        self.driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Google Maps reviews and save data to CSV")
    parser.add_argument('url', type=str, help='URL of the Google Maps page to scrape')
    parser.add_argument('csv_name', type=str, help='Name for the output CSV file')
    args = parser.parse_args()

    chromedriver_path = '../chromedriver'
    scraper = GoogleMapsScraper(chromedriver_path)
    
    print(args.url)
    scraper.open_page(args.url)
    scraper.accept_cookies()
    
    stars_summary, reviews_container = scraper.extract_reviews_summary()
    collected_reviews = scraper.extract_visible_reviews(reviews_container)
    
    # Save the extracted data
    stars_summary.to_csv(f'../data/raw/resumme_{args.csv_name}.csv', index=False)
    collected_reviews.to_csv(f'../data/raw/collected_reviews_{args.csv_name}.csv', index=False)
    
    scraper.close()

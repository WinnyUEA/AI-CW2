#Libaries for the web scraper
from urllib.parse import urlencode
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

#Builds the URL itself with arrival, depature and all of the othe input but for single joreny only
def build_thameslink_url(origin_crs, dest_crs, depart_date, depart_time, depart_type,
                         adults="1", children="0", time_preference="at"):
    # Choose prefix
    ts_prefix = "Arrive_at" if time_preference == "by" else "Depart_at"
    timestamp = f"{ts_prefix}_{depart_date}T{depart_time}"
    path = f"{origin_crs}/{dest_crs}/{timestamp}//{adults}/{children}/"

    params = {
        "departNow": "no",
        "realTime": "no",
        "searchPreferences": "",
        "showAdditionalRoutes": "no",
        "showCheapest": "no",
        "tocSpecific": "no"
    }

    base_url = "https://ticket.thameslinkrailway.com/journeys-grid/"
    full_url = base_url + path + "?" + urlencode(params)
    hour, minute = depart_time.split(":")
    return full_url, int(hour), int(minute)

#Builds the urls for return joreny with both outbound and return included
def build_thameslink_return_url(origin_crs, dest_crs, depart_date, depart_time, return_date, return_time,
                                depart_type, return_type, adults="1", children="0"):
    if depart_type == "arriving":
        depart_ts = f"Arrive_at_{depart_date}T{depart_time}"
    else:
        depart_ts = f"{depart_date}T{depart_time}"

    if return_type == "arriving":
        return_ts = f"Arrive_at_{return_date}T{return_time}"
    else:
        return_ts = f"{return_date}T{return_time}"

    path = f"{origin_crs}/{dest_crs}/{depart_ts}/{return_ts}/{adults}/{children}/"

    params = {
        "departNow": "no",
        "realTime": "no",
        "searchPreferences": "",
        "showAdditionalRoutes": "no",
        "showCheapest": "no",
        "tocSpecific": "no"
    }

    base_url = "https://ticket.thameslinkrailway.com/journeys-grid/"
    full_url = base_url + path + "?" + urlencode(params)

    dep_hour, dep_minute = depart_time.split(":")
    ret_hour, ret_minute = return_time.split(":")

    return full_url, int(dep_hour), int(dep_minute), int(ret_hour), int(ret_minute)

#Cookies handeled for the website itself
def accept_cookies_if_present(driver):
    try:
        wait = WebDriverWait(driver, 10)
        cookie_btn = wait.until(EC.presence_of_element_located((
            By.XPATH, "//a[contains(text(), 'Allow All Cookies') or @title='Allow All Cookies']"
        )))
        driver.execute_script("arguments[0].click();", cookie_btn)
        print("🍪 Clicked 'Allow All Cookies'")
    except:
        print("✅ No cookie popup or already handled.")

#The fuction finds the cheapest one way ticket for one way jorney
def extract_cheapest_single_ticket(driver, target_hour, target_minute, match_on_arrival=False):
    wait = WebDriverWait(driver, 15)
    time.sleep(10)

    user_minutes = target_hour * 60 + target_minute
    best_price = float("inf")
    best_dep_time = None
    best_arr_time = None
    best_tile = None

    fare_tiles = driver.find_elements(By.CSS_SELECTOR, "div.fare-list-v2__tile")

    for tile in fare_tiles:
        try:
            #Skips the title itself
            if "fare-list-v2__tile--unavailable" in tile.get_attribute("class"):
                continue

            #This extract the screen which has all the ticket infos
            sr_elem = tile.find_element(By.CSS_SELECTOR, ".action .sr-only")
            sr_text = sr_elem.text.strip()
            if not sr_text:
                continue
            #Extracts the departure and arrival times
            match = re.search(r"valid on the (\d{2}:\d{2}) from .*? at (\d{2}:\d{2})", sr_text)
            if not match:
                continue

            dep_time_str = match.group(1)
            arr_time_str = match.group(2)

            #This see what tiem to comapre
            compare_time = arr_time_str if match_on_arrival else dep_time_str
            compare_dt = datetime.strptime(compare_time, "%H:%M")
            compare_minutes = compare_dt.hour * 60 + compare_dt.minute

            max_diff = 60 if match_on_arrival else 30
            if abs(compare_minutes - user_minutes) > max_diff:
                continue

            #Extact the chapest price
            price_match = re.search(r"£(\d+\.\d{2})", sr_text)
            if not price_match:
                continue

            price = float(price_match.group(1))

            if price < best_price:
                best_price = price
                best_dep_time = dep_time_str
                best_arr_time = arr_time_str
                best_tile = tile

        except Exception as e:
            print(f"⚠️ Error processing tile: {e}")
            continue

    #Clixks the cheapest ticket in the whole website it found
    if best_tile:
        try:
            driver.execute_script("arguments[0].click();", best_tile)
            print(f"✅ Selected ticket: {best_dep_time} → {best_arr_time} for £{best_price:.2f}")
        except Exception as e:
            print(f"❌ Failed to click tile: {e}")
        return best_dep_time, best_arr_time, best_price
    else:
        return None, None, None

#This fucntion scraper the chapest ticket of return joreny, return only
def extract_cheapest_return_journey(driver, target_hour, target_minute, match_on_arrival=False):
    wait = WebDriverWait(driver, 15)
    time.sleep(10)

    best_price = float("inf")
    best_dep_time = None
    best_arr_time = None
    best_tile = None
    user_minutes = target_hour * 60 + target_minute

    try:
        #See where the right colum is since in the website the return ticket is inthe right column
        return_column = driver.find_element(By.CSS_SELECTOR,
            "div.service-grid-v2__content__row__column--right")
        fare_tiles = return_column.find_elements(By.CSS_SELECTOR, "div.fare-list-v2__tile")
    except Exception as e:
        print("❌ Could not locate return fare tiles:", e)
        return None, None, None

    #Goes throught all the retun ticket price and date
    for tile in fare_tiles:
        try:
            sr_text = tile.find_element(By.CSS_SELECTOR, ".action .sr-only").text.strip()
            #Uses regex to extact what time it is in the html
            time_match = re.search(r"valid on the (\d{2}:\d{2}) from .*? at (\d{2}:\d{2})", sr_text)
            if not time_match:
                continue

            dep_time_str = time_match.group(1)
            arr_time_str = time_match.group(2)

            #Here it which time to use with the user input compared
            compare_time = arr_time_str if match_on_arrival else dep_time_str
            compare_dt = datetime.strptime(compare_time, "%H:%M")
            compare_minutes = compare_dt.hour * 60 + compare_dt.minute

            #time difference
            max_diff = 60 if match_on_arrival else 30
            if abs(compare_minutes - user_minutes) > max_diff:
                continue

            #Price extract
            price_match = re.search(r"£(\d+\.\d{2})", sr_text)
            if not price_match:
                continue

            price = float(price_match.group(1))

            if price < best_price:
                best_price = price
                best_dep_time = dep_time_str
                best_arr_time = arr_time_str
                best_tile = tile

        except Exception as e:
            print(f"⚠️ Error in return tile: {e}")
            continue
    #Clicks on the best cheap ticket
    if best_tile:
        try:
            driver.execute_script("arguments[0].click();", best_tile)
            print(f"✅ Selected return: {best_dep_time} → {best_arr_time} for £{best_price:.2f}")
        except Exception as e:
            print(f"❌ Failed to click return tile: {e}")
        return best_dep_time, best_arr_time, best_price

    print("❌ No suitable return journey found.")
    return None, None, None

#This functsion extracts the cheapets ticket of outbound in the two way joreny
def click_cheapest_outbound_tile(driver, target_hour, target_minute, match_on_arrival=False):
    time.sleep(10)
    #See wher ethe left column is in html since outbound is in the left side
    left_column = driver.find_element(By.CSS_SELECTOR,
        "div.service-grid-v2__content__row__column--left")
    fare_tiles = left_column.find_elements(By.CSS_SELECTOR, "div.fare-list-v2__tile")
    #Variable for to track price and time
    best_tile = None
    best_price = float("inf")
    best_dep_time = best_arr_time = None
    user_minutes = target_hour * 60 + target_minute
    #This loops throught all the tcikste to fin the on the user input match
    for tile in fare_tiles:
        try:
            sr_text = tile.find_element(By.CSS_SELECTOR, ".action .sr-only").text.strip()
            match = re.search(r"valid on the (\d{2}:\d{2}) from .*? at (\d{2}:\d{2})", sr_text)
            if not match:
                continue

            dep_time_str, arr_time_str = match.groups()
            #See if it is arrivel time or for departure time
            compare_time = arr_time_str if match_on_arrival else dep_time_str
            compare_minutes = datetime.strptime(compare_time, "%H:%M").hour * 60 + datetime.strptime(compare_time, "%H:%M").minute
            #Time difference
            if abs(compare_minutes - user_minutes) > (60 if match_on_arrival else 30):
                continue
            #Price Extract
            price_match = re.search(r"£(\d+\.\d{2})", sr_text)
            if not price_match:
                continue

            price = float(price_match.group(1))
            if price < best_price:
                best_price = price
                best_dep_time = dep_time_str
                best_arr_time = arr_time_str
                best_tile = tile

        except Exception as e:
            print(f"⚠️ Outbound error: {e}")
            continue
    #Clicks on the one that is the best tcikets
    if best_tile:
        try:
            driver.execute_script("arguments[0].click();", best_tile)
            print(f"✅ Clicked outbound tile at {best_dep_time} → {best_arr_time} for £{best_price:.2f}")
            return best_dep_time, best_arr_time, best_price
        except Exception as e:
            print(f"❌ Failed to click outbound tile: {e}")
    else:
        print("❌ No suitable outbound ticket found.")
    return None, None, None

#This functions runs the webscraper itself
def run_thames_scraper(
    origin, destination,
    depart_date, depart_time, depart_type,
    time_preference="at",
    is_return=False, return_date=None, return_time=None, return_type="departing",
    adults="1", children="0"
):
    #Mkaes the adate into the YYYY-MM-DD format
    try:
        depart_date = datetime.strptime(depart_date, "%d/%m/%Y").strftime("%Y-%m-%d")
        if return_date:
            return_date = datetime.strptime(return_date, "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        pass

    #Build the url for the searchh of webscraper
    if is_return:
        url, dep_hour, dep_minute, ret_hour, ret_minute = build_thameslink_return_url(
            origin, destination,
            depart_date, depart_time,
            return_date, return_time,
            depart_type, return_type,
            adults, children
        )
    #Builds the one way webscraper
    else:
        url, dep_hour, dep_minute = build_thameslink_url(
            origin, destination,
            depart_date, depart_time,
            depart_type,
            adults, children,
            time_preference=time_preference
        )
    #Lauch
    driver = webdriver.Chrome()
    driver.get(url)
    accept_cookies_if_present(driver)

    out_dep = out_arr = return_dep = return_arr = None
    total_price = None
    #Runs the two way joreny with return and outbouund
    if is_return:
        out_dep, out_arr, _ = extract_cheapest_single_ticket(
            driver, dep_hour, dep_minute,
            match_on_arrival=(depart_type == "arriving")
        )
        return_dep, return_arr, _ = extract_cheapest_return_journey(
            driver, ret_hour, ret_minute,
            match_on_arrival=(return_type == "arriving")
        )
        # scrape basket total...
        try:
            time.sleep(5)
            total_elem = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div.grid-basket-summary-v2__ticket__body__total .sr-text")
                )
            )
            total_price = float(total_elem.text.strip().replace("£", ""))
        except:
            total_price = None
    #For the one way joreny handel
    else:
        out_dep, out_arr, total_price = extract_cheapest_single_ticket(
            driver, dep_hour, dep_minute,
            match_on_arrival=(depart_type == "arriving")
        )

    #Retruns the data itself
    driver.quit()
    return {
        "origin": origin,
        "destination": destination,
        "out_dep": out_dep,
        "out_arr": out_arr,
        "return_dep": return_dep,
        "return_arr": return_arr,
        "total_price": total_price,
        "url": url
    }

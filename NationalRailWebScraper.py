#libareis Used for this web scraper
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlencode
from datetime import datetime, timedelta
import time


#This code below rounds the minute down to 15 minuts interval. This was doen sdue to the website only allowing 15 min period
def round_down_to_quarter(minute):
    return str((int(minute) // 15) * 15).zfill(2)

#The function below will build the url for the website for the scraper to acces it
def build_url(origin, destination, date_str, time_str, adults, children,
              return_date_str=None, return_time_str=None,
              outbound_type="departing", return_type="departing",
              time_preference="at"):
    #This will exprt all the date and time from the user input
    day, month, year = date_str.split("/")
    hour, minute = time_str.split(":")
    minute = round_down_to_quarter(minute)
    leaving_date = f"{day}{month}{year[2:]}"
    leaving_hour = hour.zfill(2)

    #This will see what the date is either like depature or arrive by time
    if time_preference == "by":
        leaving_type = "arriving"
    elif time_preference == "after":
        leaving_type = "departing"
    else:
        leaving_type = outbound_type

    #This now build the URL
    params = {
        "origin": origin,
        "destination": destination,
        "leavingType": leaving_type,
        "leavingDate": leaving_date,
        "leavingHour": leaving_hour,
        "leavingMin": minute,
        "adults": adults,
        "children": children,
        "extraTime": "0"
    }

    #sO over here it returen the trip that user decided on
    if return_date_str and return_time_str:
        r_day, r_month, r_year = return_date_str.split("/")
        r_hour, r_minute = return_time_str.split(":")
        r_minute = round_down_to_quarter(r_minute)
        return_date = f"{r_day}{r_month}{r_year[2:]}"
        return_hour = r_hour.zfill(2)
        if time_preference == "by":
            return_type_final = "arriving"
        elif time_preference == "after":
            return_type_final = "departing"
        else:
            return_type_final = return_type

        params.update({
            "type": "return",
            "returnType": return_type_final,
            "returnDate": return_date,
            "returnHour": return_hour,
            "returnMin": r_minute
        })
    else:
        params["type"] = "single"
    url = "https://www.nationalrail.co.uk/journey-planner/?" + urlencode(params)
    return url, int(hour), int(minute)

#This is bascially to accept the cookies of the website itself
def accept_cookies_if_present(driver):
    try:
        wait = WebDriverWait(driver, 10)
        reject_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Reject All')]")))
        reject_btn.click()
        print("🍪 Cookie popup rejected.")
    except:
        print("✅ No cookie popup appeared or already handled.")

#The function below will extract the cheap tikcet for outward jorney only
def extract_cheapest_ticket(driver, target_hour, target_minute, match_on_arrival=False):
    time.sleep(10)
    cards = driver.find_elements(By.CSS_SELECTOR, "section[id^='outward-']")
    cheapest_price = float("inf")
    best_match = None

    for card in cards:
        try:
            times = card.find_elements(By.CSS_SELECTOR, 'time[class*="bAcVR"]')

            if len(times) < 2:
                continue

            #SO matches what kind of time user chose
            dep_time_str = times[0].text.strip()
            arr_time_str = times[1].text.strip()
            #This compare time difference and accept the give time diff
            match_time = arr_time_str if match_on_arrival else dep_time_str
            match_dt = datetime.strptime(match_time, "%H:%M")
            match_minutes = match_dt.hour * 60 + match_dt.minute
            target_minutes = target_hour * 60 + target_minute
            diff = abs(match_minutes - target_minutes)
            max_diff = 60 if match_on_arrival else 30
            if diff > max_diff:
                continue
            #Extract and keep the cheapest ticket
            price_elem = card.find_element(By.XPATH, ".//div[contains(@id,'result-card-price-outward')]/div/span[2]")
            price_text = price_elem.text.strip().replace("£", "")
            price = float(price_text)
            if price < cheapest_price:
                cheapest_price = price
                best_match = {
                    "dep_time": dep_time_str,
                    "arr_time": arr_time_str,
                    "price": price
                }

        except Exception:
            continue

    return best_match

#This fuction extact cheap ticket for outward jorney but for only 2 way ticket
def select_cheapest_outward_journey_within_time(driver, target_hour, target_minute, match_on_arrival=False):
    time.sleep(10)
    cards = driver.find_elements(By.CSS_SELECTOR, "section[id^='outward-']")
    best_card = None
    lowest_price = float("inf")
    best_time = None
    best_arrival = None

    for card in cards:
        try:
            times = card.find_elements(By.CSS_SELECTOR, 'time[class*="bAcVR"]')
            if len(times) < 2:
                continue
            #SO matches what kind of time user chose
            dep_time_str = times[0].text.strip()
            arr_time_str = times[1].text.strip()

            # This compare time difference and accept the give time diff
            match_time = arr_time_str if match_on_arrival else dep_time_str
            match_dt = datetime.strptime(match_time, "%H:%M")
            match_minutes = match_dt.hour * 60 + match_dt.minute
            target_minutes = target_hour * 60 + target_minute
            max_diff = 60 if match_on_arrival else 30
            if abs(match_minutes - target_minutes) > max_diff:
                continue
            #Extract and keep the cheapest ticket
            price_elem = card.find_element(By.XPATH, ".//div[contains(@id,'result-card-price-outward')]/div/span[2]")
            price_text = price_elem.text.strip().replace("£", "")
            price = float(price_text)

            if price < lowest_price:
                lowest_price = price
                best_card = card
                best_time = dep_time_str
                best_arrival = arr_time_str

        except Exception:
            continue
    #Here it clciks the best price to move forward in webb page
    if best_card:
        try:
            button = best_card.find_element(By.XPATH, ".//input[@type='button']")
            driver.execute_script("arguments[0].click();", button)
            print(f"✅ Clicked outward journey at {best_time} → {best_arrival} for £{lowest_price:.2f}")
            return best_time, best_arrival, lowest_price
        except Exception as e:
            print(f"❌ Failed to click selected ticket: {e}")
    else:
        print("❌ No ticket found within ±60 minutes." if match_on_arrival else "❌ No ticket found within ±30 minutes.")

    return None, None, None

#This fucstion will excatert the return ticket fornthe return jorney of the 2 way tciket
def select_return_journey(driver, return_time_str, match_on_arrival=False):
    user_return_time = datetime.strptime(return_time_str, "%H:%M")
    #This see how up and down the time can be for the selcted user time so for arrival is 60 mintues up and down and 30 minutes up and down for departure
    time_window = timedelta(minutes=60 if match_on_arrival else 30)
    time.sleep(10)
    journey_sections = driver.find_elements(By.CSS_SELECTOR, 'section[id^="inward-"]')

    #Does tracking variable for best match
    best_option = None
    best_price = float('inf')
    closest_time_diff = timedelta.max
    best_dep_time = None
    best_arr_time = None

    #Goes through return joreny of website and see which condition macthes the user input
    for section in journey_sections:
        try:
            times = section.find_elements(By.CSS_SELECTOR, 'time[datetime]')
            if len(times) < 2:
                continue

            #time and decides if it is arrival or departure
            dep_time_str = times[0].text.strip()
            arr_time_str = times[1].text.strip()
            compare_time_str = arr_time_str if match_on_arrival else dep_time_str
            compare_time = datetime.strptime(compare_time_str, "%H:%M")

            #calculkates the time difference and see which matches
            time_diff = abs(compare_time - user_return_time)
            if time_diff > time_window:
                continue

            price_elem = section.find_element(By.XPATH, ".//div[contains(@id,'price')]/div/span[2]")
            price = float(price_elem.text.replace("£", "").strip())

            #selecte the card tat matches the user input
            if price < best_price or (price == best_price and time_diff < closest_time_diff):
                best_price = price
                best_dep_time = dep_time_str
                best_arr_time = arr_time_str
                closest_time_diff = time_diff
                best_option = section
        except Exception:
            continue

    #Clicks on the card that matches the user inoput
    if best_option:
        try:
            select_btn = best_option.find_element(By.XPATH, './/input[@type="button"]')
            driver.execute_script("arguments[0].click();", select_btn)
            print(f"✅ Return Journey: {best_dep_time} → {best_arr_time}, Price = £{best_price:.2f}")
            return best_dep_time, best_arr_time, best_price
        except Exception as e:
            print(f"❌ Failed to click return journey: {e}")
    else:
        print("❌ No suitable return journey found within ±60 minutes." if match_on_arrival else "❌ No suitable return journey found within ±30 minutes.")
    return None, None, None

#This is the main function that runs the whole web scraper once the user inputs
def run_national_scraper(
    origin, destination,
    depart_date, depart_time, depart_type,
    time_preference="at",
    is_return=False, return_date=None, return_time=None, return_type="departing",
    adults="1", children="0"
):
    #This builds the url using the user input
    url, dep_hour, dep_minute = build_url(
        origin, destination,
        depart_date, depart_time,
        adults, children,
        return_date, return_time,
        outbound_type=depart_type,
        return_type=return_type,
        time_preference=time_preference
    )

    #usese selenium to opne jorney and handels cookie
    driver = webdriver.Chrome()
    driver.get(url)
    accept_cookies_if_present(driver)

    #this installs jorney date
    out_dep = out_arr = return_dep = return_arr = None
    total_price = None

    #this handels the retruen jorney
    if is_return:
        #handels the outward joreny on 2 way and clisk it
        out_dep, out_arr, out_price = select_cheapest_outward_journey_within_time(
            driver, dep_hour, dep_minute,
            match_on_arrival=(depart_type == "arriving")
        )
        #handels the return joreny on 2 way and clicks ir
        return_dep, return_arr, return_price = select_return_journey(
            driver, return_time,
            match_on_arrival=(return_type == "arriving")
        )
        #calculates the total of 2 way joreny
        if out_price and return_price:
            total_price = out_price + return_price
    else:
        #handesl single joreny
        result = extract_cheapest_ticket(
            driver, dep_hour, dep_minute,
            match_on_arrival=(depart_type == "arriving")
        )
        if result:
            out_dep = result["dep_time"]
            out_arr = result["arr_time"]
            total_price = result["price"]

    driver.quit()

    #retrurns all the joreny after web scraping
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

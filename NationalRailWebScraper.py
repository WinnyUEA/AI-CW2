from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlencode
from datetime import datetime, timedelta
import time

def round_down_to_quarter(minute):
    return str((int(minute) // 15) * 15).zfill(2)

def build_url(origin, destination, date_str, time_str, adults, children,
              return_date_str=None, return_time_str=None,
              outbound_type="departing", return_type="departing",
              time_preference="at"):
    """
    Build National Rail URL with time preference.
    time_preference: 'at' (exact), 'by' (arrive by), or 'after' (depart after).
    """
    day, month, year = date_str.split("/")
    hour, minute = time_str.split(":")
    minute = round_down_to_quarter(minute)
    leaving_date = f"{day}{month}{year[2:]}"
    leaving_hour = hour.zfill(2)

    # Map preference to leavingType
    if time_preference == "by":
        leaving_type = "arriving"
    elif time_preference == "after":
        leaving_type = "departing"
    else:
        leaving_type = outbound_type

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

    if return_date_str and return_time_str:
        r_day, r_month, r_year = return_date_str.split("/")
        r_hour, r_minute = return_time_str.split(":")
        r_minute = round_down_to_quarter(r_minute)
        return_date = f"{r_day}{r_month}{r_year[2:]}"
        return_hour = r_hour.zfill(2)

        # Map preference to returnType
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

def accept_cookies_if_present(driver):
    try:
        wait = WebDriverWait(driver, 10)
        reject_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Reject All')]")))
        reject_btn.click()
        print("🍪 Cookie popup rejected.")
    except:
        print("✅ No cookie popup appeared or already handled.")

def extract_cheapest_ticket(driver, target_hour, target_minute, match_on_arrival=False):
    time.sleep(10)
    cards = driver.find_elements(By.CSS_SELECTOR, "section[data-testid^='result-card-section-outward']")
    cheapest_price = float("inf")
    best_match = None

    for card in cards:
        try:
            times = card.find_elements(By.CSS_SELECTOR, 'time[datetime]')
            if len(times) < 2:
                continue

            dep_time_str = times[0].text.strip()
            arr_time_str = times[1].text.strip()

            match_time = arr_time_str if match_on_arrival else dep_time_str
            match_dt = datetime.strptime(match_time, "%H:%M")
            match_minutes = match_dt.hour * 60 + match_dt.minute
            target_minutes = target_hour * 60 + target_minute
            diff = abs(match_minutes - target_minutes)

            max_diff = 60 if match_on_arrival else 30
            if diff > max_diff:
                continue

            price_elem = card.find_element(By.XPATH, ".//span[contains(text(), '£')]")
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

def select_cheapest_outward_journey_within_time(driver, target_hour, target_minute, match_on_arrival=False):
    time.sleep(10)
    cards = driver.find_elements(By.CSS_SELECTOR, "section[data-testid^='result-card-section-outward']")
    best_card = None
    lowest_price = float("inf")
    best_time = None
    best_arrival = None

    for card in cards:
        try:
            times = card.find_elements(By.CSS_SELECTOR, 'time[datetime]')
            if len(times) < 2:
                continue

            dep_time_str = times[0].text.strip()
            arr_time_str = times[1].text.strip()

            match_time = arr_time_str if match_on_arrival else dep_time_str
            match_dt = datetime.strptime(match_time, "%H:%M")
            match_minutes = match_dt.hour * 60 + match_dt.minute
            target_minutes = target_hour * 60 + target_minute

            max_diff = 60 if match_on_arrival else 30
            if abs(match_minutes - target_minutes) > max_diff:
                continue

            price_elem = card.find_element(By.XPATH, ".//span[contains(text(), '£')]")
            price_text = price_elem.text.strip().replace("£", "")
            price = float(price_text)

            if price < lowest_price:
                lowest_price = price
                best_card = card
                best_time = dep_time_str
                best_arrival = arr_time_str

        except Exception:
            continue

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

def select_return_journey(driver, return_time_str, match_on_arrival=False):
    user_return_time = datetime.strptime(return_time_str, "%H:%M")
    time_window = timedelta(minutes=60 if match_on_arrival else 30)
    time.sleep(10)
    journey_sections = driver.find_elements(By.CSS_SELECTOR, 'section[id^="inward-"]')

    best_option = None
    best_price = float('inf')
    closest_time_diff = timedelta.max
    best_dep_time = None
    best_arr_time = None

    for section in journey_sections:
        try:
            times = section.find_elements(By.CSS_SELECTOR, 'time[datetime]')
            if len(times) < 2:
                continue

            dep_time_str = times[0].text.strip()
            arr_time_str = times[1].text.strip()
            compare_time_str = arr_time_str if match_on_arrival else dep_time_str
            compare_time = datetime.strptime(compare_time_str, "%H:%M")

            time_diff = abs(compare_time - user_return_time)
            if time_diff > time_window:
                continue

            price_elem = section.find_element(By.XPATH, ".//div[contains(@id,'price')]/div/span[2]")
            price = float(price_elem.text.replace("£", "").strip())

            if price < best_price or (price == best_price and time_diff < closest_time_diff):
                best_price = price
                best_dep_time = dep_time_str
                best_arr_time = arr_time_str
                closest_time_diff = time_diff
                best_option = section

        except Exception:
            continue

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

def run_national_scraper(
    origin, destination,
    depart_date, depart_time, depart_type,
    time_preference="at",
    is_return=False, return_date=None, return_time=None, return_type="departing",
    adults="1", children="0"
):
    # Build the URL using the new time_preference
    url, dep_hour, dep_minute = build_url(
        origin, destination,
        depart_date, depart_time,
        adults, children,
        return_date, return_time,
        outbound_type=depart_type,
        return_type=return_type,
        time_preference=time_preference
    )

    driver = webdriver.Chrome()
    driver.get(url)
    accept_cookies_if_present(driver)

    out_dep = out_arr = return_dep = return_arr = None
    total_price = None

    if is_return:
        out_dep, out_arr, out_price = select_cheapest_outward_journey_within_time(
            driver, dep_hour, dep_minute,
            match_on_arrival=(depart_type == "arriving")
        )
        return_dep, return_arr, return_price = select_return_journey(
            driver, return_time,
            match_on_arrival=(return_type == "arriving")
        )
        if out_price and return_price:
            total_price = out_price + return_price
    else:
        result = extract_cheapest_ticket(
            driver, dep_hour, dep_minute,
            match_on_arrival=(depart_type == "arriving")
        )
        if result:
            out_dep = result["dep_time"]
            out_arr = result["arr_time"]
            total_price = result["price"]

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


# def main():
#     print("=== National Rail Cheapest Ticket Finder (with 'Arrive By') ===")
#     origin = input("Enter Origin Station Code (e.g., PAD): ").strip().upper()
#     destination = input("Enter Destination Station Code (e.g., NRW): ").strip().upper()
#     date_str = input("Enter Travel Date (DD/MM/YYYY): ").strip()
#     time_str = input("Enter Time (HH:MM, 24-hour format): ").strip()
#     time_type = input("Is this time for 'depart by' or 'arrive by'? ").strip().lower()
#     adults = input("Number of Adults (default 1): ").strip() or "1"
#     children = input("Number of Children (default 0): ").strip() or "0"
#
#     is_return = input("Is this a return journey? (yes/no): ").strip().lower()
#     return_date_str = None
#     return_time_str = None
#     return_time_type = "departing"
#
#     outbound_type = "arriving" if time_type == "arrive by" else "departing"
#
#     if is_return == "yes":
#         return_date_str = input("Enter Return Date (DD/MM/YYYY): ").strip()
#         return_time_str = input("Enter Return Time (HH:MM, 24-hour format): ").strip()
#         return_time_type_input = input("Is return time for 'depart by' or 'arrive by'? ").strip().lower()
#         return_time_type = "arriving" if return_time_type_input == "arrive by" else "departing"
#
#     journey_url, target_hour, target_minute = build_url(
#         origin, destination, date_str, time_str, adults, children,
#         return_date_str, return_time_str, outbound_type, return_time_type
#     )
#
#     print(f"🌐 Opening browser to:\n{journey_url}")
#     driver = webdriver.Chrome()
#     driver.get(journey_url)
#     accept_cookies_if_present(driver)
#
#     if is_return == "yes":
#         print("🔍 Looking for outward journey...")
#         outward_time, outward_arrival, outward_price = select_cheapest_outward_journey_within_time(
#             driver, target_hour, target_minute, match_on_arrival=(outbound_type == "arriving")
#         )
#
#         print("🔁 Looking for return journey...")
#         return_time, return_arrival, return_price = select_return_journey(
#             driver, return_time_str, match_on_arrival=(return_time_type == "arriving")
#         )
#
#         driver.quit()
#         print("\n🧾 Round Trip Summary:")
#         print("------------------------------")
#         if outward_time:
#             print(f"🚆 Outward: {outward_time} → {outward_arrival} | £{outward_price:.2f}")
#         else:
#             print("❌ No outward journey found")
#
#         if return_time:
#             print(f"🔁 Return: {return_time} → {return_arrival} | £{return_price:.2f}")
#         else:
#             print("❌ No return journey found")
#
#         if outward_price and return_price:
#             print(f"💷 Total: £{return_price:.2f}")
#         print(f"🔗 URL: {journey_url}")
#
#     else:
#         print("🔍 Searching for single journey...")
#         best_match = extract_cheapest_ticket(driver, target_hour, target_minute, match_on_arrival=(outbound_type == "arriving"))
#         driver.quit()
#
#         print("\n🧾 Single Journey Summary:")
#         print("------------------------------")
#         if best_match:
#             print(f"🕓 {best_match['dep_time']} → {best_match['arr_time']}")
#             print(f"💷 Price: £{best_match['price']:.2f}")
#         else:
#             print("❌ No suitable ticket found")
#         print(f"🔗 URL: {journey_url}")
#
# if __name__ == "__main__":
#     main()

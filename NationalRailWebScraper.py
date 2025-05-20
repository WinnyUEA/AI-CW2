from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlencode
from datetime import datetime, timedelta
import time

def round_down_to_quarter(minute):
    return str((int(minute) // 15) * 15).zfill(2)

def build_url(origin, destination, date_str, time_str, adults, children, return_date_str=None, return_time_str=None):
    day, month, year = date_str.split("/")
    hour, minute = time_str.split(":")
    minute = round_down_to_quarter(minute)
    leaving_date = f"{day}{month}{year[2:]}"
    leaving_hour = hour.zfill(2)

    params = {
        "origin": origin,
        "destination": destination,
        "leavingType": "departing",
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

        params.update({
            "type": "return",
            "returnType": "departing",
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

def extract_cheapest_ticket(driver, target_hour, target_minute):
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

            dep_time = datetime.strptime(dep_time_str, "%H:%M")
            dep_minutes = dep_time.hour * 60 + dep_time.minute
            target_minutes = target_hour * 60 + target_minute
            diff = abs(dep_minutes - target_minutes)

            if diff > 30:
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


def select_cheapest_outward_journey_within_time(driver, target_hour, target_minute):
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

            dep_time = datetime.strptime(dep_time_str, "%H:%M")
            dep_minutes = dep_time.hour * 60 + dep_time.minute
            target_minutes = target_hour * 60 + target_minute
            diff = abs(dep_minutes - target_minutes)

            if diff > 30:
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
        print("❌ No ticket found within ±60 minutes.")

    return None, None, None


def select_return_journey(driver, return_time_str):
    user_return_time = datetime.strptime(return_time_str, "%H:%M")
    time_window = timedelta(minutes=30)
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
            dep_time = datetime.strptime(dep_time_str, "%H:%M")

            time_diff = abs(dep_time - user_return_time)
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
        print("❌ No suitable return journey found within ±60 minutes.")

    return None, None, None


def main():
    print("=== National Rail Cheapest Ticket Finder ===")
    origin = input("Enter Origin Station Code (e.g., PAD): ").strip().upper()
    destination = input("Enter Destination Station Code (e.g., NRW): ").strip().upper()
    date_str = input("Enter Travel Date (DD/MM/YYYY): ").strip()
    time_str = input("Enter Departure Time (HH:MM, 24-hour format): ").strip()
    adults = input("Number of Adults (default 1): ").strip() or "1"
    children = input("Number of Children (default 0): ").strip() or "0"

    is_return = input("Is this a return journey? (yes/no): ").strip().lower()
    return_date_str = None
    return_time_str = None

    if is_return == "yes":
        return_date_str = input("Enter Return Date (DD/MM/YYYY): ").strip()
        return_time_str = input("Enter Return Time (HH:MM, 24-hour format): ").strip()

    journey_url, target_hour, target_minute = build_url(
        origin, destination, date_str, time_str, adults, children, return_date_str, return_time_str
    )

    print(f"🌐 Opening browser to:\n{journey_url}")
    driver = webdriver.Chrome()
    driver.get(journey_url)

    accept_cookies_if_present(driver)

    if is_return == "yes":
        print("🔍 Looking for the best outward journey within ±60 mins...")
        outward_time, outward_arrival, outward_price = select_cheapest_outward_journey_within_time(driver, target_hour, target_minute)

        print("🔄 Now extracting best return journey within ±60 mins of return time...")
        return_time, return_arrival, return_price = select_return_journey(driver, return_time_str)

        total_price = outward_price

        print("\n🧾 Summary of Selected Journey:")
        print("----------------------------------")
        print(f"🚆 Outward: {outward_time} → {outward_arrival} | £{outward_price:.2f}" if outward_time else "❌ No outward journey found")
        print(f"🔁 Return : {return_time} → {return_arrival} | £{return_price:.2f}" if return_time else "❌ No return journey found")
        print(f"💷 Outward Price Only: £{total_price:.2f}" if total_price else "💷 Outward Price: N/A")
        print(f"🌐 Final Booking URL: {journey_url}")
        print("----------------------------------")

        input("\n👉 Press ENTER to close browser.")
        driver.quit()
    else:
        print("🔍 Scanning results for cheapest ticket near your time...")
        best_match = extract_cheapest_ticket(driver, target_hour, target_minute)
        driver.quit()

        if best_match:
            print(f"\n💸 Cheapest ticket near {time_str}:")
            print(f"🕓 {best_match['dep_time']} → {best_match['arr_time']}")
            print(f"💷 Price: £{best_match['price']:.2f}")
            print(f"🔗 URL: {journey_url}")
        else:
            print("\n❌ No suitable tickets found within ±30 minutes of your time.")

def run_national_scraper(origin, destination, date_str, time_str, adults, children, return_date_str=None, return_time_str=None):
    journey_url, target_hour, target_minute = build_url(
        origin, destination, date_str, time_str, adults, children, return_date_str, return_time_str
    )
    driver = webdriver.Chrome()
    driver.get(journey_url)
    accept_cookies_if_present(driver)

    if return_date_str and return_time_str:
        out_dep, out_arr, out_price = select_cheapest_outward_journey_within_time(driver, target_hour, target_minute)
        ret_dep, ret_arr, ret_price = select_return_journey(driver, return_time_str)
        driver.quit()

        if out_price and ret_price:
            return {
                "provider": "National Rail",
                "total_price": ret_price,
                "out_dep": out_dep, "out_arr": out_arr, "out_price": out_price,
                "ret_dep": ret_dep, "ret_arr": ret_arr, "ret_price": ret_price,
                "url": journey_url
            }

    else:
        best_match = extract_cheapest_ticket(driver, target_hour, target_minute)
        driver.quit()

        if best_match:
            return {
                "provider": "National Rail",
                "total_price": best_match["price"],
                "out_dep": best_match["dep_time"],
                "out_arr": best_match["arr_time"],
                "out_price": best_match["price"],
                "url": journey_url
            }

    return None




# if __name__ == "__main__":
#     main()
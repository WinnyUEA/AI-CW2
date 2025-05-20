from urllib.parse import urlencode
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

def build_thameslink_url(origin_crs, dest_crs, depart_date, depart_time,
                         adults="1", children="0"):
    depart_ts = f"{depart_date}T{depart_time}"
    path = f"{origin_crs}/{dest_crs}/{depart_ts}//{adults}/{children}/"

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

def build_thameslink_return_url(origin_crs, dest_crs, depart_date, depart_time,
                                return_date, return_time, adults="1", children="0"):
    depart_ts = f"{depart_date}T{depart_time}"
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

def extract_cheapest_single_ticket(driver, target_hour, target_minute):
    wait = WebDriverWait(driver, 15)
    time.sleep(10)

    user_minutes = target_hour * 60 + target_minute
    best_price = float("inf")
    best_dep_time = None
    best_arr_time = None

    # Find all fare tiles in visible fare panel
    fare_tiles = driver.find_elements(By.CSS_SELECTOR, "div.fare-list-v2__tile")

    for tile in fare_tiles:
        try:
            # Skip unavailable fares
            if "fare-list-v2__tile--unavailable" in tile.get_attribute("class"):
                continue

            # Extract sr-only text
            sr_elem = tile.find_element(By.CSS_SELECTOR, ".action .sr-only")
            sr_text = sr_elem.text.strip()
            if not sr_text:
                continue

            # Extract time info from sr-only text
            match = re.search(r"valid on the (\d{2}:\d{2}) from .*? at (\d{2}:\d{2})", sr_text)
            if not match:
                continue

            dep_time_str = match.group(1)
            arr_time_str = match.group(2)

            dep_time = datetime.strptime(dep_time_str, "%H:%M")
            dep_minutes = dep_time.hour * 60 + dep_time.minute

            # Filter by ±30 mins
            if abs(dep_minutes - user_minutes) > 30:
                print(f"⏱️ Skipping {dep_time_str} – outside ±30 mins")
                continue

            # Extract price from aria-hidden (not sr-text)
            price_span = tile.find_element(By.CSS_SELECTOR, ".price span[aria-hidden='true']")
            price_text = price_span.text.strip().replace("£", "")
            price = float(price_text)

            print(f"✅ Valid fare: £{price:.2f} at {dep_time_str} → {arr_time_str}")

            if price < best_price:
                best_price = price
                best_dep_time = dep_time_str
                best_arr_time = arr_time_str

        except Exception as e:
            print(f"⚠️ Error processing fare tile: {e}")
            continue

    if best_price < float("inf"):
        return best_dep_time, best_arr_time, best_price
    else:
        return None, None, None




def extract_cheapest_return_journey(driver, target_return_hour, target_return_minute):
    wait = WebDriverWait(driver, 15)
    time.sleep(10)

    best_price = float("inf")
    best_dep_time = None
    best_arr_time = None
    best_tile = None

    # Find return panel container
    try:
        return_section = driver.find_element(By.CSS_SELECTOR,
            "div.service-grid-v2__content__row__column--right")
        fare_tiles = return_section.find_elements(By.CSS_SELECTOR, "div.fare-list-v2__tile")
    except Exception as e:
        print("❌ Could not locate return fare tiles:", e)
        return None, None, None

    for tile in fare_tiles:
        try:
            # Extract sr-only fare description
            sr_text = tile.find_element(By.CSS_SELECTOR, ".action .sr-only").text.strip()

            # Extract departure and arrival times
            time_match = re.search(r"valid on the (\d{2}:\d{2}) from .*? at (\d{2}:\d{2})", sr_text)
            if not time_match:
                continue

            ret_dep_time_str = time_match.group(1)
            ret_arr_time_str = time_match.group(2)

            dep_time = datetime.strptime(ret_dep_time_str, "%H:%M")
            dep_minutes = dep_time.hour * 60 + dep_time.minute
            user_minutes = target_return_hour * 60 + target_return_minute

            if abs(dep_minutes - user_minutes) >= 30:
                continue

            # Extract price
            price_match = re.search(r"£(\d+\.\d{2})", sr_text)
            if not price_match:
                continue

            price = float(price_match.group(1))

            if price < best_price:
                best_price = price
                best_dep_time = ret_dep_time_str
                best_arr_time = ret_arr_time_str
                best_tile = tile

        except Exception as e:
            print(f"⚠️ Error in return tile: {e}")
            continue

    if best_tile:
        try:
            driver.execute_script("arguments[0].click();", best_tile)
            print(f"✅ Selected return at {best_dep_time} → {best_arr_time} for £{best_price:.2f}")
            return best_dep_time, best_arr_time, best_price
        except Exception as e:
            print(f"❌ Failed to click return tile: {e}")

    print("❌ No suitable return journey found.")
    return None, None, None



def main():
    print("=== Thameslink Cheapest Ticket Finder ===")

    origin = input("Enter Origin CRS (e.g., NRW): ").strip().upper()
    dest = input("Enter Destination CRS (e.g., PAD): ").strip().upper()
    date_str = input("Enter Travel Date (DD/MM/YYYY): ").strip()
    time_str = input("Enter Departure Time (HH:MM, 24-hour): ").strip()
    adults = input("Number of Adults (default 1): ").strip() or "1"
    children = input("Number of Children (default 0): ").strip() or "0"

    is_return = input("Is this a return journey? (yes/no): ").strip().lower()
    return_date_str = None
    return_time_str = None

    if is_return == "yes":
        return_date_str = input("Enter Return Date (DD/MM/YYYY): ").strip()
        return_time_str = input("Enter Return Time (HH:MM, 24-hour): ").strip()

        # Format dates for URL
        depart_date = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        return_date = datetime.strptime(return_date_str, "%d/%m/%Y").strftime("%Y-%m-%d")

        # Build return journey URL
        url, hour, minute, return_hour, return_minute = build_thameslink_return_url(
            origin, dest, depart_date, time_str, return_date, return_time_str, adults, children
        )
    else:
        # Format date for URL
        formatted_date = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")

        # Build single journey URL
        url, hour, minute = build_thameslink_url(origin, dest, formatted_date, time_str, adults, children)

    print(f"🌐 Opening browser to: {url}")
    driver = webdriver.Chrome()
    driver.get(url)

    accept_cookies_if_present(driver)

    if is_return == "yes":
        print("🔍 Looking for outbound ticket...")
        out_dep, out_arr, out_price = extract_cheapest_single_ticket(driver, hour, minute)

        print("🔄 Looking for return ticket...")
        ret_dep, ret_arr, ret_price = extract_cheapest_return_journey(driver, return_hour, return_minute)

        total = out_price + ret_price if out_price and ret_price else None

        print("\n🧾 Round Trip Summary:")
        print("----------------------------")
        print(f"🚆 Outbound : {out_dep} → {out_arr} | £{out_price:.2f}" if out_dep else "❌ Outward not found")
        print(f"🔁 Return   : {ret_dep} → {ret_arr} | £{ret_price:.2f}" if ret_dep else "❌ Return not found")
        print(f"💷 Total    : £{total:.2f}" if total else "💷 Total: N/A")
        print(f"🔗 URL      : {url}")
        driver.quit()
    else:
        out_dep, out_arr, out_price = extract_cheapest_single_ticket(driver, hour, minute)
        driver.quit()

        if out_dep:
            print("\n🧾 Cheapest Ticket Found:")
            print("----------------------------")
            print(f"🕓 Journey : {out_dep} → {out_arr}")
            print(f"💷 Price   : £{out_price:.2f}")
            print(f"🔗 URL     : {url}")
        else:
            print("\n❌ No tickets found within ±30 minutes.")


def run_thames_scraper(origin, destination, date_str, time_str, adults, children, return_date_str=None, return_time_str=None):
    is_return = return_date_str and return_time_str

    if is_return:
        depart_date = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        return_date = datetime.strptime(return_date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        url, dep_hour, dep_min, ret_hour, ret_min = build_thameslink_return_url(
            origin, destination, depart_date, time_str, return_date, return_time_str, adults, children)
    else:
        formatted_date = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        url, dep_hour, dep_min = build_thameslink_url(
            origin, destination, formatted_date, time_str, adults, children)

    driver = webdriver.Chrome()
    driver.get(url)
    accept_cookies_if_present(driver)

    if is_return:
        out_dep, out_arr, out_price = extract_cheapest_single_ticket(driver, dep_hour, dep_min)
        ret_dep, ret_arr, ret_price = extract_cheapest_return_journey(driver, ret_hour, ret_min)
        driver.quit()

        if out_price and ret_price:
            return {
                "provider": "Thameslink",
                "total_price": out_price + ret_price,
                "out_dep": out_dep, "out_arr": out_arr, "out_price": out_price,
                "ret_dep": ret_dep, "ret_arr": ret_arr, "ret_price": ret_price,
                "url": url
            }

    else:
        out_dep, out_arr, out_price = extract_cheapest_single_ticket(driver, dep_hour, dep_min)
        driver.quit()

        if out_dep:
            return {
                "provider": "Thameslink",
                "total_price": out_price,
                "out_dep": out_dep,
                "out_arr": out_arr,
                "out_price": out_price,
                "url": url
            }

    return None




# if __name__ == "__main__":
#     main()
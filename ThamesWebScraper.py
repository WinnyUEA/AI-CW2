from urllib.parse import urlencode
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

def build_thameslink_url(origin_crs, dest_crs, depart_date, depart_time, depart_type,
                         adults="1", children="0"):
    if depart_type == "arriving":
        depart_ts = f"Arrive_at_{depart_date}T{depart_time}"
    else:
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
            if "fare-list-v2__tile--unavailable" in tile.get_attribute("class"):
                continue

            sr_elem = tile.find_element(By.CSS_SELECTOR, ".action .sr-only")
            sr_text = sr_elem.text.strip()
            if not sr_text:
                continue

            match = re.search(r"valid on the (\d{2}:\d{2}) from .*? at (\d{2}:\d{2})", sr_text)
            if not match:
                continue

            dep_time_str = match.group(1)
            arr_time_str = match.group(2)

            compare_time = arr_time_str if match_on_arrival else dep_time_str
            compare_dt = datetime.strptime(compare_time, "%H:%M")
            compare_minutes = compare_dt.hour * 60 + compare_dt.minute

            max_diff = 60 if match_on_arrival else 30
            if abs(compare_minutes - user_minutes) > max_diff:
                continue

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

    if best_tile:
        try:
            driver.execute_script("arguments[0].click();", best_tile)
            print(f"✅ Selected ticket: {best_dep_time} → {best_arr_time} for £{best_price:.2f}")
        except Exception as e:
            print(f"❌ Failed to click tile: {e}")
        return best_dep_time, best_arr_time, best_price
    else:
        return None, None, None


def extract_cheapest_return_journey(driver, target_hour, target_minute, match_on_arrival=False):
    wait = WebDriverWait(driver, 15)
    time.sleep(10)

    best_price = float("inf")
    best_dep_time = None
    best_arr_time = None
    best_tile = None
    user_minutes = target_hour * 60 + target_minute

    try:
        return_column = driver.find_element(By.CSS_SELECTOR,
            "div.service-grid-v2__content__row__column--right")
        fare_tiles = return_column.find_elements(By.CSS_SELECTOR, "div.fare-list-v2__tile")
    except Exception as e:
        print("❌ Could not locate return fare tiles:", e)
        return None, None, None

    for tile in fare_tiles:
        try:
            sr_text = tile.find_element(By.CSS_SELECTOR, ".action .sr-only").text.strip()
            time_match = re.search(r"valid on the (\d{2}:\d{2}) from .*? at (\d{2}:\d{2})", sr_text)
            if not time_match:
                continue

            dep_time_str = time_match.group(1)
            arr_time_str = time_match.group(2)

            compare_time = arr_time_str if match_on_arrival else dep_time_str
            compare_dt = datetime.strptime(compare_time, "%H:%M")
            compare_minutes = compare_dt.hour * 60 + compare_dt.minute

            max_diff = 60 if match_on_arrival else 30
            if abs(compare_minutes - user_minutes) > max_diff:
                continue

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

    if best_tile:
        try:
            driver.execute_script("arguments[0].click();", best_tile)
            print(f"✅ Selected return: {best_dep_time} → {best_arr_time} for £{best_price:.2f}")
        except Exception as e:
            print(f"❌ Failed to click return tile: {e}")
        return best_dep_time, best_arr_time, best_price

    print("❌ No suitable return journey found.")
    return None, None, None


def click_cheapest_outbound_tile(driver, target_hour, target_minute, match_on_arrival=False):
    time.sleep(10)
    left_column = driver.find_element(By.CSS_SELECTOR,
        "div.service-grid-v2__content__row__column--left")
    fare_tiles = left_column.find_elements(By.CSS_SELECTOR, "div.fare-list-v2__tile")

    best_tile = None
    best_price = float("inf")
    best_dep_time = best_arr_time = None
    user_minutes = target_hour * 60 + target_minute

    for tile in fare_tiles:
        try:
            sr_text = tile.find_element(By.CSS_SELECTOR, ".action .sr-only").text.strip()
            match = re.search(r"valid on the (\d{2}:\d{2}) from .*? at (\d{2}:\d{2})", sr_text)
            if not match:
                continue

            dep_time_str, arr_time_str = match.groups()
            compare_time = arr_time_str if match_on_arrival else dep_time_str
            compare_minutes = datetime.strptime(compare_time, "%H:%M").hour * 60 + datetime.strptime(compare_time, "%H:%M").minute
            if abs(compare_minutes - user_minutes) > (60 if match_on_arrival else 30):
                continue

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

def run_thames_scraper(
    origin, destination,
    depart_date, depart_time, depart_type,
    is_return=False, return_date=None, return_time=None, return_type="departing",
    adults="1", children="0"
):
    # ←——— ADD THIS BLOCK:
    from datetime import datetime
    try:
        depart_date = datetime.strptime(depart_date, "%d/%m/%Y").strftime("%Y-%m-%d")
        if return_date:
            return_date = datetime.strptime(return_date, "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        pass

    if is_return:
        url, dep_hour, dep_minute, ret_hour, ret_minute = build_thameslink_return_url(
            origin, destination, depart_date, depart_time,
            return_date, return_time,
            depart_type, return_type, adults, children
        )
    else:
        url, dep_hour, dep_minute = build_thameslink_url(
            origin, destination, depart_date, depart_time,
            depart_type, adults, children
        )
    driver = webdriver.Chrome()
    driver.get(url)
    accept_cookies_if_present(driver)

    out_dep = out_arr = return_dep = return_arr = None
    total_price = None

    if is_return:
        out_dep, out_arr, _ = extract_cheapest_single_ticket(
            driver, dep_hour, dep_minute, match_on_arrival=(depart_type == "arriving")
        )

        return_dep, return_arr, _ = extract_cheapest_return_journey(
            driver, ret_hour, ret_minute, match_on_arrival=(return_type == "arriving")
        )

        # Get basket price
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import time

            time.sleep(5)
            total_elem = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.grid-basket-summary-v2__ticket__body__total .sr-text"))
            )
            price_text = total_elem.text.strip()
            total_price = float(price_text.replace("£", "").strip())
        except:
            total_price = None
    else:
        out_dep, out_arr, total_price = extract_cheapest_single_ticket(
            driver, dep_hour, dep_minute, match_on_arrival=(depart_type == "arriving")
        )

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
#     print("=== Thameslink Cheapest Ticket Finder (with 'Arrive By' + Return Click Total) ===")
#
#     origin = input("Enter Origin CRS (e.g., NRW): ").strip().upper()
#     dest = input("Enter Destination CRS (e.g., PAD): ").strip().upper()
#     date_str = input("Enter Travel Date (DD/MM/YYYY): ").strip()
#     time_str = input("Enter Departure Time (HH:MM, 24-hour): ").strip()
#     time_type = input("Is this time for 'depart by' or 'arrive by'? ").strip().lower()
#     adults = input("Number of Adults (default 1): ").strip() or "1"
#     children = input("Number of Children (default 0): ").strip() or "0"
#
#     is_return = input("Is this a return journey? (yes/no): ").strip().lower()
#     return_date_str = None
#     return_time_str = None
#     return_time_type = "departing"
#
#     depart_type = "arriving" if time_type == "arrive by" else "departing"
#
#     if is_return == "yes":
#         return_date_str = input("Enter Return Date (DD/MM/YYYY): ").strip()
#         return_time_str = input("Enter Return Time (HH:MM, 24-hour): ").strip()
#         return_time_type_input = input("Is return time for 'depart by' or 'arrive by'? ").strip().lower()
#         return_time_type = "arriving" if return_time_type_input == "arrive by" else "departing"
#
#         depart_date = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
#         return_date = datetime.strptime(return_date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
#
#         url, hour, minute, return_hour, return_minute = build_thameslink_return_url(
#             origin, dest, depart_date, time_str, return_date, return_time_str,
#             depart_type, return_time_type, adults, children
#         )
#     else:
#         formatted_date = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
#
#         url, hour, minute = build_thameslink_url(
#             origin, dest, formatted_date, time_str, depart_type, adults, children
#         )
#
#     print(f"🌐 Opening browser to: {url}")
#     driver = webdriver.Chrome()
#     driver.get(url)
#
#     accept_cookies_if_present(driver)
#
#     if is_return == "yes":
#         print("🔍 Looking for outbound ticket...")
#         out_dep, out_arr, _ = extract_cheapest_single_ticket(driver, hour, minute, match_on_arrival=(depart_type == "arriving"))
#
#         print("🔄 Looking for return ticket...")
#         ret_dep, ret_arr, _ = extract_cheapest_return_journey(driver, return_hour, return_minute, match_on_arrival=(return_time_type == "arriving"))
#
#         # Wait for basket total
#         time.sleep(5)
#         try:
#             total_elem = WebDriverWait(driver, 10).until(
#                 EC.presence_of_element_located((By.CSS_SELECTOR, "div.grid-basket-summary-v2__ticket__body__total .sr-text"))
#             )
#             total_text = total_elem.text.strip()
#         except:
#             total_text = "N/A"
#
#         print("\n🧾 Round Trip Summary:")
#         print("----------------------------")
#         print(f"🚆 Outbound : {out_dep} → {out_arr}" if out_dep else "❌ Outbound not found")
#         print(f"🔁 Return   : {ret_dep} → {ret_arr}" if ret_dep else "❌ Return not found")
#         print(f"💷 Total from Site: {total_text}")
#         print(f"🔗 URL      : {url}")
#         driver.quit()
#
#     else:
#         out_dep, out_arr, out_price = extract_cheapest_single_ticket(driver, hour, minute, match_on_arrival=(depart_type == "arriving"))
#         driver.quit()
#
#         if out_dep:
#             print("\n🧾 Cheapest Ticket Found:")
#             print("----------------------------")
#             print(f"🕓 Journey : {out_dep} → {out_arr}")
#             print(f"💷 Price   : £{out_price:.2f}")
#             print(f"🔗 URL     : {url}")
#         else:
#             print("\n❌ No tickets found within ±60 minutes." if depart_type == "arriving" else "\n❌ No tickets found within ±30 minutes.")
#
# if __name__ == "__main__":
#     main()

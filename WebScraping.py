from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import csv

def load_station_codes_from_csv(csv_path):
    station_map = {}
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row['name'].strip().lower()
            code = row['tiploc'].strip().upper()
            aliases = row['longname.name_alias'].strip().lower().split(',') if row['longname.name_alias'] else []

            # Add main name
            station_map[name] = code

            # Add each alias
            for alias in aliases:
                alias = alias.strip()
                if alias and alias not in station_map:
                    station_map[alias] = code
    return station_map




csv_path = "/Users/prasid/Masters/AI/Adssignemnet 2/Station data/7028B-CW2-Task3-Specification-v2.csv"
station_code_map = load_station_codes_from_csv(csv_path)





def get_cheapest_ticket_selenium(orig_code, dest_code):
    url = f"https://www.brfares.com/#!fares?orig={orig_code}&dest={dest_code}"

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)

    try:
        buttons = driver.find_elements(By.TAG_NAME, "button")
        if buttons and len(buttons) >= 2:
            buttons[1].click()
            print("✅ Cookie consent clicked (second button).")
            time.sleep(2)
            driver.save_screenshot("after_consent.png")
        else:
            print("⚠️ Less than 2 buttons found, cannot click consent.")
    except Exception as e:
        print("ℹ️ Consent click failed:", e)


    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        print(f"Found {len(rows)} rows in table(s).\n")

        # # 🔍 Debug: print each row's raw text
        # for i, row in enumerate(rows):
        #     print(f"Row {i + 1}:")
        #     print(row.text)
        #     print("-" * 40)

        print(f"Found {len(rows)} rows in table(s).")

        fares = []

        for i, row in enumerate(rows):
            text = row.text.strip()
            if "£" in text:
                # Try to extract the price using regex
                import re
                match = re.search(r"£\d+\.\d{2}", text)
                if match:
                    price_str = match.group()
                    try:
                        price_val = float(price_str.replace("£", ""))
                        ticket_type = text.split("\n")[0].strip().upper()

                        # Skip irrelevant fare types
                        if "CHILD" in ticket_type or "UNDER" in ticket_type or "SEASON" in ticket_type:
                            continue

                        fares.append((ticket_type, price_val))
                    except:
                        continue

        if not fares:
            return {"error": "No fares were found on the page."}

        fares.sort(key=lambda x: x[1])

        print("\n🎟️ All filtered fares found:\n")
        for ticket_type, price in fares:
            print(f"{ticket_type:30} - £{price:.2f}")

        # Grab the cheapest
        cheapest = fares[0]

        return {
            "ticket_type": cheapest[0],
            "price": f"£{cheapest[1]:.2f}",
            "link": url
        }

    except Exception as e:
        result = {"error": str(e)}
    finally:
        driver.quit()

    return result



# === MAIN ===
print("=== Cheapest Train Ticket Finder ===")
# station_code_map = load_station_codes_from_csv("7028B-CW2-Task3-Specification-v2.csv")

origin_name = input("Enter origin station name: ").strip().lower()
dest_name = input("Enter destination station name: ").strip().lower()

# Check if user input matches name or is already a CRS code
orig_code = station_code_map.get(origin_name.lower())
if not orig_code and origin_name.upper() in station_code_map.values():
    orig_code = origin_name.upper()

dest_code = station_code_map.get(dest_name.lower())
if not dest_code and dest_name.upper() in station_code_map.values():
    dest_code = dest_name.upper()



if not orig_code or not dest_code:
    print("Unknown station name.")
else:
    result = get_cheapest_ticket_selenium(orig_code, dest_code)
    if "error" in result:
        print("Error:", result["error"])
    else:
        print(f"\nCheapest ticket from {origin_name.title()} to {dest_name.title()}:")
        print(f"Type: {result['ticket_type']}")
        print(f"Price: {result['price']}")
        print(f"Book here: {result['link']}")

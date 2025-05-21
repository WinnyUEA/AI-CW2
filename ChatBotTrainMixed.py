from NationalRailWebScraper import run_national_scraper
from ThamesWebScraper import run_thames_scraper

def compare_ticket_prices(
    origin, destination,
    depart_date, depart_time, depart_type,
    is_return=False, return_date=None, return_time=None, return_type="departing",
    adults="1", children="0"
):
    # Run National Rail scraper
    print("🔎 Running National Rail scraper...")
    national_result = run_national_scraper(
        origin, destination,
        depart_date, depart_time, depart_type,
        is_return, return_date, return_time, return_type,
        adults, children
    )

    # Run Thameslink scraper
    print("🔎 Running Thameslink scraper...")
    thames_result = run_thames_scraper(
        origin, destination,
        depart_date, depart_time, depart_type,
        is_return, return_date, return_time, return_type,
        adults, children
    )

    verdict = "Could not compare prices."

    if national_result["total_price"] and thames_result["total_price"]:
        if national_result["total_price"] < thames_result["total_price"]:
            verdict = "🏆 National Rail offers the cheapest ticket."
        elif thames_result["total_price"] < national_result["total_price"]:
            verdict = "🏆 Thameslink offers the cheapest ticket."
        else:
            verdict = "🤝 Both websites offer the same cheapest price."

    elif national_result["total_price"]:
        verdict = "⚠️ Only National Rail returned a valid price."
    elif thames_result["total_price"]:
        verdict = "⚠️ Only Thameslink returned a valid price."

    return {
        "national_rail": national_result,
        "thameslink": thames_result,
        "verdict": verdict
    }

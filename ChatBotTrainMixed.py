from NationalRailWebScraper import run_national_scraper
from ThamesWebScraper import run_thames_scraper

def compare_ticket_prices(
    origin, destination,
    depart_date, depart_time, depart_type,
    time_preference='at',
    is_return=False, return_date=None, return_time=None, return_type="departing",
    adults="1", children="0"
):
    """
    Compare ticket prices between National Rail and Thameslink scrapers,
    passing through user's time_preference ('at', 'by', 'after').
    """
    # Run National Rail scraper
    print("🔎 Running National Rail scraper...")
    national_result = run_national_scraper(
        origin, destination,
        depart_date, depart_time, depart_type,
        time_preference=time_preference,
        is_return=is_return, return_date=return_date,
        return_time=return_time, return_type=return_type,
        adults=adults, children=children
    )

    # Run Thameslink scraper
    print("🔎 Running Thameslink scraper...")
    thames_result = run_thames_scraper(
        origin, destination,
        depart_date, depart_time, depart_type,
        time_preference=time_preference,
        is_return=is_return, return_date=return_date,
        return_time=return_time, return_type=return_type,
        adults=adults, children=children
    )

    # Determine verdict
    verdict = "Could not compare prices."
    nr_price = national_result.get('total_price')
    tl_price = thames_result.get('total_price')

    if nr_price is not None and tl_price is not None:
        if nr_price < tl_price:
            verdict = "🏆 National Rail offers the cheapest ticket."
        elif tl_price < nr_price:
            verdict = "🏆 Thameslink offers the cheapest ticket."
        else:
            verdict = "🤝 Both websites offer the same cheapest price."
    elif nr_price is not None:
        verdict = "⚠️ Only National Rail returned a valid price."
    elif tl_price is not None:
        verdict = "⚠️ Only Thameslink returned a valid price."

    return {
        "national_rail": national_result,
        "thameslink": thames_result,
        "verdict": verdict
    }


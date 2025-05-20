# TicketScraperEngine.py

from NationalRailWebScraper import run_national_scraper
from ThamesWebScraper import run_thames_scraper

def compare_ticket_prices(origin, destination, date_str, time_str, adults, children,
                          return_date_str=None, return_time_str=None):
    try:
        national = run_national_scraper(origin, destination, date_str, time_str, adults, children,
                                        return_date_str, return_time_str)
        thames = run_thames_scraper(origin, destination, date_str, time_str, adults, children,
                                    return_date_str, return_time_str)

        results = [r for r in [national, thames] if r and r["total_price"] > 0]
        if not results:
            return "\u274c No tickets found on any website."

        if len(results) == 1:
            r = results[0]
            summary = f"\u2705 Cheapest ticket found on {r['provider']}:\n"
            summary += f"🕓 Outbound: {r['out_dep']} → {r['out_arr']} | £{r.get('out_price', r['total_price']):.2f}\n"
            if "ret_dep" in r:
                summary += f"🔁 Return: {r['ret_dep']} → {r['ret_arr']} | £{r['ret_price']:.2f}\n"
            summary += f"💷 Total: £{r['total_price']:.2f}\n"
            summary += f"🔗 {r['url']}"
            return summary

        if national and thames and national["total_price"] == thames["total_price"]:
            return (
                f"✅ Both websites offer the same cheapest price:\n"
                f"🎯 Total Price: £{national['total_price']:.2f}\n\n"
                f"🔹 National Rail:\n🕓 {national['out_dep']} → {national['out_arr']}\n"
                f"🔁 {national.get('ret_dep', '')} → {national.get('ret_arr', '')}\n"
                f"🔗 {national['url']}\n\n"
                f"🔴 Thameslink:\n🕓 {thames['out_dep']} → {thames['out_arr']}\n"
                f"🔁 {thames.get('ret_dep', '')} → {thames.get('ret_arr', '')}\n"
                f"🔗 {thames['url']}"
            )

        cheapest = min(results, key=lambda r: r["total_price"])
        summary = (
            f"✅ Cheapest ticket found on {cheapest['provider']}:\n"
            f"🕓 Outbound: {cheapest['out_dep']} → {cheapest['out_arr']} | £{cheapest.get('out_price', cheapest['total_price']):.2f}\n"
        )
        if "ret_dep" in cheapest:
            summary += f"🔁 Return: {cheapest['ret_dep']} → {cheapest['ret_arr']} | £{cheapest['ret_price']:.2f}\n"
        summary += f"💷 Total: £{cheapest['total_price']:.2f}\n"
        summary += f"🔗 {cheapest['url']}"
        return summary

    except Exception as e:
        return f"⚠️ Error comparing ticket prices: {e}"
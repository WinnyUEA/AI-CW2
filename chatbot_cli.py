import json

# Load the knowledge base once
def load_knowledge_base():
    with open("knowledge_base.json", "r") as f:
        return json.load(f)

def get_advice(location, blockage, kb):
    for item in kb:
        if location.lower() in item["location"].lower() and blockage.lower() == item["blockage"].lower():
            advice = f"\n🚦 {item['code']} – {item['location']}\n"
            advice += f"▶ Advice: {item['advice']}\n"
            if item.get("staff_notes"):
                advice += f"👷 Staff Notes: {item['staff_notes']}\n"
            if item.get("passenger_notes"):
                advice += f"🧳 Passenger Notes: {item['passenger_notes']}\n"
            if item.get("alt_transport"):
                advice += "🚍 Alternative Transport:\n"
                for mode in item["alt_transport"]:
                    advice += f"   • {mode}\n"
            return advice
    return "❌ No matching contingency found for the input."

def main():
    kb = load_knowledge_base()
    print("=== 🚂 Railway Contingency Chatbot (Staff CLI) ===\n")
    while True:
        location = input("Enter blockage location (e.g. Ipswich - Stowmarket): ")
        blockage = input("Enter blockage type (full or partial): ").lower()

        advice = get_advice(location, blockage, kb)
        print(advice)

        again = input("\nCheck another? (y/n): ").lower()
        if again != "y":
            break

if __name__ == "__main__":
    main()

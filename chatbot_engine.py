import json

def load_kb():
    with open("knowledge_base.json") as f:
        return json.load(f)

def get_advice(location, blockage):
    kb = load_kb()
    for rule in kb:
        if rule["location"].lower() == location.lower() and rule["blockage"].lower() == blockage.lower():
            return f"📘 Plan {rule['code']} in effect: {rule['advice']}"
    return "⚠️ No matching contingency found. Please contact the control center."

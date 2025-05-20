async function getContingencyAdvice(location, blockage) {
  try {
    const response = await fetch('/api/contingency');
    const data = await response.json();

    const match = data.find(item =>
      item.location.toLowerCase().includes(location.toLowerCase()) &&
      item.blockage.toLowerCase() === blockage.toLowerCase()
    );

    if (!match) {
      return '❌ No contingency plan found for the entered details.';
    }

    let advice = `<h3>🚦 ${match.code} – ${match.location}</h3>`;
    advice += `<p><strong>Advice:</strong> ${match.advice}</p>`;
    advice += match.staff_notes ? `<p><strong>Staff Notes:</strong> ${match.staff_notes}</p>` : '';
    advice += match.passenger_notes ? `<p><strong>Passenger Notes:</strong> ${match.passenger_notes}</p>` : '';

    if (match.alt_transport && match.alt_transport.length) {
      advice += `<p><strong>Alternative Transport:</strong><ul>`;
      for (const method of match.alt_transport) {
        advice += `<li>${method}</li>`;
      }
      advice += `</ul></p>`;
    }

    return advice;

  } catch (error) {
    console.error('Error fetching KB:', error);
    return '⚠️ Error retrieving data.';
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("contingency-form");
  const resultDiv = document.getElementById("contingency-result");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const location = document.getElementById("location-input").value.trim();
    const blockage = document.getElementById("blockage-input").value.trim();
    resultDiv.innerHTML = "🔍 Searching...";

    const advice = await getContingencyAdvice(location, blockage);
    resultDiv.innerHTML = advice;
  });
});
